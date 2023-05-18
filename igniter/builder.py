#!/usr/bin/env python

from typing import Dict, Any, Callable

import torch.nn as nn
from torch.utils.data import DataLoader

import importlib
import os
from datetime import datetime
from omegaconf import OmegaConf

from ignite.engine import Engine, Events
from ignite.handlers import Checkpoint, BasicTimeProfiler
from ignite.contrib.handlers import ProgressBar
import ignite.distributed as idist

from igniter.registry import model_registry, dataset_registry, io_registry, proc_registry
from igniter.utils import is_distributed, get_collate_fn
from igniter.logger import logger


def build_transforms(cfg) -> Dict[str, Any]:
    transforms = {}
    for key in cfg.transforms:
        module = importlib.import_module(cfg.transforms[key].engine)
        augs = cfg.transforms[key].augs
        if augs is None:
            transforms[key] = None
            continue
        transforms[key] = module.Compose([getattr(module, cls)(**(augs[cls]) if augs[cls] else {}) for cls in augs])
    return transforms


def build_train_dataloader(cfg) -> Dict[str, DataLoader]:
    name = cfg.build.dataset
    cls = dataset_registry[name]
    attrs = cfg.datasets
    transforms = build_transforms(cfg)
    dataloaders = {}
    collate_fn = get_collate_fn(cfg)

    for key in attrs[name]:
        if key not in attrs[name]:
            continue
        try:
            logger.info(f'Building {key} dataloader')
            dataset = cls(**{**dict(attrs[name][key]), 'transforms': transforms.get(key, None)})
            kwargs = dict(attrs.dataloader)
            kwargs.pop('collate_fn', None)
            dataloaders[key] = DataLoader(dataset, collate_fn=collate_fn, **kwargs)
        except TypeError as e:
            logger.warning(e)
            dataloaders[key] = None

    return dataloaders


def build_model(cfg) -> nn.Module:
    logger.info(f'Building network model {cfg.build.model}')
    cls_or_func = model_registry[cfg.build.model]
    try:
        return cls_or_func.build(cfg)
    except AttributeError as e:
        logger.debug(e)
        return cls_or_func(cfg)


def build_optim(cfg, model):
    module = importlib.import_module(cfg.solvers.engine)
    name = cfg.build.solver
    return getattr(module, name)(model.parameters(), **cfg.solvers[name])


def build_scheduler(cfg, optimizer):
    name = cfg.build.get('scheduler', None)
    if not name:
        return

    module = importlib.import_module('torch.optim.lr_scheduler')
    return getattr(module, name)(optimizer=optimizer, **cfg.solvers.schedulers[name])


def add_profiler(engine, cfg):
    profiler = BasicTimeProfiler()
    profiler.attach(engine)

    @engine.on(Events.ITERATION_COMPLETED(every=cfg.solvers.snapshot))
    def log_intermediate_results():
        profiler.print_results(profiler.get_results())

    return profiler


def build_io(cfg) -> Dict[str, Callable]:
    if not cfg.get('io'):
        return

    def _build(cfg):
        engine = cfg.engine
        cls = io_registry[engine]
        cls = importlib.import_module(engine) if cls is None else cls
        try:
            return cls.build(cfg)
        except AttributeError:
            return cls(cfg)

    return {key: _build(cfg.io[key]) for key in cfg.io}


def build_func(cfg):
    func_name = cfg.build.get('func', None) or 'default'
    func = proc_registry[func_name]
    if func is None:
        logger.info('Using default training function')
        func = proc_registry['default']
    assert func, 'Training forward function is not defined'
    return func


class TrainerEngine(Engine):
    def __init__(
        self,
        cfg,
        process_func,
        model,
        optimizer,
        dataloaders: Dict[str, DataLoader],
        io_ops: Dict[str, Callable] = None,
        **kwargs
    ) -> None:
        self._scheduler = kwargs.pop('scheduler', None)
        super(TrainerEngine, self).__init__(process_func, **kwargs)

        # TODO: Move this to each builder function
        if is_distributed(cfg):
            model = idist.auto_model(model)
            optimizer = idist.auto_optim(optimizer)
            for key in dataloaders:
                if dataloaders[key] is None:
                    continue
                attrs = dict(cfg.datasets.dataloader)
                attrs.pop('collate_fn', None)
                dataloaders[key] = idist.auto_dataloader(
                    dataloaders[key].dataset, collate_fn=get_collate_fn(cfg), **attrs
                )

        self._cfg = cfg
        self._model = model
        self._optimizer = optimizer
        self._train_dl, self._val = dataloaders['train'], dataloaders['val']

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
        self.add_persistent_logger()

        OmegaConf.save(cfg, os.path.join(self.log_dir, 'config.yaml'))

    @classmethod
    def build(cls, cfg) -> 'TrainerEngine':
        os.makedirs(cfg.workdir.path, exist_ok=True)
        model = build_model(cfg)
        optimizer = build_optim(cfg, model)
        io_ops = build_io(cfg)
        update_model = build_func(cfg)
        dls = build_train_dataloader(cfg)
        scheduler = build_scheduler(cfg, optimizer)
        return cls(cfg, update_model, model, optimizer, dataloaders=dls, io_ops=io_ops, scheduler=scheduler)

    def __call__(self):
        self.run(self._train_dl, self._cfg.solvers.epochs, epoch_length=len(self._train_dl))
        self._writer.close()

    def scheduler(self):
        if self._scheduler:
            self._scheduler.step()

    def summary(self):
        for key in self.state.metrics:
            self._writer.add_scalar(key, self.state.metrics[key], self.state.iteration)

    def checkpoint_handler(self):
        if self._cfg.solvers.snapshot == 0:
            return

        prefix = '%s'
        if self.checkpoint is None:
            logger.warning(f'Using default checkpoint saver to directory {self.log_dir}')
            self.checkpoint = importlib.import_module('torch').save
            prefix = os.path.join(self.log_dir, '%s')

        def _checkpointer():
            filename = prefix % f'model_{str(self.state.epoch).zfill(7)}.pt'
            self.checkpoint(self.trainer_state_dict(), filename)

        self.add_event_handler(
            Events.ITERATION_COMPLETED(every=self._cfg.solvers.snapshot) | Events.EPOCH_COMPLETED,
            _checkpointer
        )

    def add_persistent_logger(self, **kwargs) -> None:
        ProgressBar(persist=False).attach(self, metric_names='all', output_transform=None)

    def get_lr(self) -> float:
        lr = self._optimizer.param_groups[0]['lr']
        return lr[0] if isinstance(lr, list) else lr

    def trainer_state_dict(self) -> Dict[str, Any]:
        return {
            'model': self._model.state_dict(),
            'optimizer': self._optimizer.state_dict()
        }


def build_trainer(cfg) -> TrainerEngine:
    return TrainerEngine.build(cfg)


def _trainer(rank, cfg) -> None:
    trainer = build_trainer(cfg)
    trainer()


def trainer(cfg):
    if is_distributed(cfg):
        init_args = dict(cfg.distributed[cfg.distributed.type])
        with idist.Parallel(
            backend=cfg.distributed.backend, nproc_per_node=cfg.distributed.nproc_per_node, **init_args
        ) as parallel:
            parallel.run(_trainer, cfg)
    else:
        _trainer(None, cfg)
