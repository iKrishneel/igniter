#!/usr/bin/env python

from typing import Dict, Any, Callable, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import importlib
import os
from datetime import datetime
from omegaconf import OmegaConf, DictConfig

from ignite.engine import Engine, Events
from ignite.contrib.handlers import ProgressBar
import ignite.distributed as idist

from igniter.utils import is_distributed, model_name
from igniter.logger import logger
from igniter.registry import io_registry, engine_registry


__all__ = ['TrainerEngine', 'EvaluationEngine']


@engine_registry('default_trainer')
class TrainerEngine(Engine):
    def __init__(
        self,
        cfg: DictConfig,
        process_func: Callable,
        model: nn.Module,
        dataloader: DataLoader,
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

        self._cfg = cfg
        self._model = model
        self._optimizer = optimizer
        self._dataloader = dataloader

        self.checkpoint = None
        if io_ops:
            self.__dict__.update(io_ops)

        if cfg.workdir.get('unique', False):
            name = 'run_' + str(datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
            self.log_dir = os.path.join(str(cfg.workdir.path), name)
        else:
            self.log_dir = str(cfg.workdir.path)

        self._writer = io_registry['summary_writer'](log_dir=self.log_dir)

        self.add_event_handler(Events.EPOCH_COMPLETED, self.scheduler)
        self.add_event_handler(Events.ITERATION_COMPLETED, self.summary)

        self.checkpoint_handler()
        self.add_persistent_logger(self)

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

    def checkpoint_handler(self) -> None:
        if self._cfg.solvers.snapshot <= 0:
            return

        prefix = '%s'
        if self.checkpoint is None:
            logger.warning(f'Using default checkpoint saver to directory {self.log_dir}')
            self.checkpoint = importlib.import_module('torch').save
            prefix = os.path.join(self.log_dir, '%s')
            return

        def _checkpointer():
            filename = prefix % f'model_{str(self.state.epoch).zfill(7)}.pt'
            self.checkpoint(self.trainer_state_dict(), filename)

        self.add_event_handler(
            Events.ITERATION_COMPLETED(every=self._cfg.solvers.snapshot) | Events.EPOCH_COMPLETED, _checkpointer
        )

    def get_lr(self) -> float:
        lr = self._optimizer.param_groups[0]['lr']
        return lr[0] if isinstance(lr, list) else lr

    def trainer_state_dict(self) -> Dict[str, Any]:
        return {'model': self._model.state_dict(), 'optimizer': self._optimizer.state_dict()}

    @staticmethod
    def add_persistent_logger(engine, **kwargs) -> None:
        ProgressBar(persist=False).attach(engine, metric_names='all', output_transform=None)


@engine_registry('default_evaluation')
class EvaluationEngine(Engine):
    def __init__(
        self,
        cfg: DictConfig,
        process_func: Callable,
        model: nn.Module,
        dataloader: DataLoader,
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

        self._cfg = cfg
        self._model = model
        self._dataloader = dataloader

        if io_ops:
            self.__dict__.update(io_ops)

        self._iter = 0
        TrainerEngine.add_persistent_logger(self)

    def __call__(self):
        self._iter = 0
        self.run(self._dataloader)