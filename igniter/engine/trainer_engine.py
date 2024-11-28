#!/usr/bin/env python

import importlib
import os
from datetime import datetime
from typing import Any, Callable, Dict, Optional, Union

import ignite.distributed as idist
import torch
import torch.nn as nn
from ignite.engine import Engine as _Engine
from ignite.engine import Events
from ignite.handlers import Checkpoint
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

from igniter.logger import logger
from igniter.registry import engine_registry, io_registry
from igniter.utils import get_device, is_distributed, model_name

try:
    from ignite.handlers import ProgressBar
except ImportError:
    from ignite.contrib.handlers import ProgressBar


__all__ = ['TrainerEngine', 'EvaluationEngine']


def get_datetime(fmt: str = '%Y-%m-%dT%H-%M-%S') -> str:
    return str(datetime.now().strftime(fmt))


class Engine(_Engine):
    def add_persistent_logger(self) -> None:
        if hasattr(self, 'log_handler'):
            kwargs = getattr(self.log_handler, '_attach_kwargs', {})
            self.log_handler.attach(self, **kwargs)


@engine_registry('default_trainer')
class TrainerEngine(Engine):
    def __init__(
        self,
        cfg: DictConfig,
        process_func: Callable,
        model: nn.Module,
        dataloader: Union[DataLoader, Any],
        optimizer=None,
        io_ops: Optional[Dict[str, Callable]] = None,
        **kwargs,
    ) -> None:
        self._scheduler = kwargs.pop('scheduler', None)
        super(TrainerEngine, self).__init__(process_func, **kwargs)

        # TODO: Move this to each builder function
        if is_distributed(cfg):
            build_func = importlib.import_module('igniter.builder').build_func
            model = idist.auto_model(model)
            optimizer = idist.auto_optim(optimizer)
            attrs = dict(cfg.datasets.dataloader)
            dataloader = idist.auto_dataloader(
                dataloader.dataset, collate_fn=build_func(attrs.pop('collate_fn', 'collate_fn')), **attrs
            )
        else:
            model = model.to(get_device(cfg))

        self._cfg = cfg
        self._model = model
        self._optimizer = optimizer
        self._dataloader = dataloader

        self.log_handler = ProgressBar(persist=False)
        self.checkpoint = None
        if io_ops:
            self.__dict__.update(io_ops)

        if cfg.workdir.get('unique', False):
            name = 'run_' + get_datetime()
            self.log_dir = os.path.join(str(cfg.workdir.path), name)
        else:
            self.log_dir = str(cfg.workdir.path)

        self._writer = io_registry['summary_writer'](log_dir=self.log_dir)

        # TODO: better way to handle the event type
        scheduler_event = (
            Events.ITERATION_COMPLETED
            if isinstance(self._scheduler, torch.optim.lr_scheduler.OneCycleLR)
            else Events.EPOCH_COMPLETED
        )
        self.add_event_handler(scheduler_event, self.scheduler)
        self.add_event_handler(Events.ITERATION_COMPLETED, self.summary)

        self.checkpoint_handler()
        self.add_persistent_logger()

        OmegaConf.save(cfg, os.path.join(self.log_dir, 'config.yaml'))

    def __call__(self) -> None:
        train_cfg = self._cfg.build[model_name(self._cfg)].train
        epoch_length = train_cfg.get('iters_per_epoch', len(self._dataloader))
        self.run(self._dataloader, train_cfg.epochs, epoch_length=epoch_length)
        self._writer.close()

    def scheduler(self) -> None:
        if self._scheduler:
            self._scheduler.step()

    def summary(self) -> None:
        for key in self.state.metrics:
            if isinstance(self.state.metrics[key], str):
                continue

            value = self.state.metrics[key]
            value = torch.Tensor([value]) if isinstance(value, (float, int)) else value
            if torch.isnan(value.detach().cpu()).any():
                raise ValueError(f'{key} is NaN. Terminating on iteration {self.state.iteration}')

            self._writer.add_scalar(f'train/{key}', value, self.state.iteration)

    def checkpoint_handler(self, prefix: str = '%s') -> None:
        if self._cfg.solvers.snapshot <= 0:
            logger.warning('Not model checkpoint will be saved because snapshot <= 0')
            return

        def _checkpointer():
            filename = prefix % f'model_{str(self.state.epoch).zfill(7)}.pt'
            self.checkpoint(self.get_state_dict(), filename)

        if self.checkpoint is None:
            default_path = f'./logs/{self._cfg.build.model}/models/{get_datetime()}'
            logger.info(f'No checkpoint handler! Using default and saving {default_path}')

            _checkpointer = Checkpoint(
                {'model': self._model},
                default_path,
                n_saved=2,
                filename_prefix=prefix % '',
            )

        self.add_event_handler(
            Events.ITERATION_COMPLETED(every=self._cfg.solvers.snapshot) | Events.EPOCH_COMPLETED, _checkpointer
        )

    def get_lr(self) -> float:
        lr = self._optimizer.param_groups[0]['lr']
        return lr[0] if isinstance(lr, list) else lr

    def get_state_dict(self) -> Dict[str, Any]:
        state_dict = {
            'model': self._model.state_dict(),
            'cfg': self._cfg,
            'optimizer': self._optimizer.state_dict(),
            'scheduler': self._scheduler.state_dict() if self._scheduler is not None else None,
            'state': self.state,
        }

        save_options = self._cfg.io.checkpoint.get('save', 'all') if hasattr(self._cfg.io, 'checkpoint') else 'state'
        if save_options == 'all':
            return state_dict

        for key in ['optimizer', 'scheduler', 'state']:
            if key in save_options:
                continue
            state_dict.pop(key)

        return state_dict


@engine_registry('default_evaluation')
class EvaluationEngine(Engine):
    def __init__(
        self,
        cfg: DictConfig,
        process_func: Callable,
        model: nn.Module,
        dataloader: Union[DataLoader, Any],
        io_ops: Optional[Dict[str, Callable]] = None,
        **kwargs,
    ) -> None:
        self._scheduler = kwargs.pop('scheduler', None)
        super(EvaluationEngine, self).__init__(process_func)

        # TODO: Move this to each builder function
        if is_distributed(cfg):
            build_func = importlib.import_module('igniter.builder').build_func
            model = idist.auto_model(model)
            attrs = dict(cfg.datasets.dataloader)
            dataloader = idist.auto_dataloader(
                dataloader.dataset, collate_fn=build_func(attrs.pop('collate_fn', 'collate_fn')), **attrs
            )
        else:
            model = model.to(get_device(cfg))

        self._cfg = cfg
        self._model = model
        self._dataloader = dataloader

        if io_ops:
            self.__dict__.update(io_ops)

        self._iter = 0
        self.add_persistent_logger()

    def __call__(self):
        self._iter = 0
        val_cfg = self._cfg.build[model_name(self._cfg)].val
        epoch_length = int(val_cfg.get('iters_per_epoch', len(self._dataloader)))
        epochs = int(getattr(val_cfg, 'epochs', 1))
        if epochs < 1:
            raise ValueError(f'Epochs must be greater then 0 but got {epochs}')
        self.run(self._dataloader, epochs, epoch_length=epoch_length)
