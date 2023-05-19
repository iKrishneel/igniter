#!/usr/bin/env python

from typing import List, Dict, Any, Callable, Optional

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
from igniter.utils import is_distributed
from igniter.logger import logger

MODES: List[str] = ['train', 'val', 'test']


def model_name(cfg):
    return cfg.build.model


def build_transforms(cfg, mode: Optional[str] = None) -> Dict[str, Any]:
    transforms = {}
    for key in cfg.transforms:
        if mode and key != mode:
            continue
        module = importlib.import_module(cfg.transforms[key].engine)
        augs = cfg.transforms[key].augs
        if augs is None:
            transforms[key] = None
            continue
        transforms[key] = module.Compose([getattr(module, cls)(**(augs[cls]) if augs[cls] else {}) for cls in augs])

    if mode:
        transforms = transforms[mode]
    return transforms


def build_dataloader(cfg, mode: str) -> DataLoader:
    logger.info(f'Building {mode} dataloader')

    name = cfg.build[model_name(cfg)].dataset
    attrs = cfg.datasets[name].get(mode, None)
    kwargs = dict(cfg.datasets.dataloader)
    assert attrs, f'{mode} not found in datasets'

    cls = dataset_registry[name]
    transforms = build_transforms(cfg, mode)
    collate_fn = build_func(kwargs.pop('collate_fn', 'collate_fn'))
    dataset = cls(**{**dict(attrs), 'transforms': transforms})
    return DataLoader(dataset, collate_fn=collate_fn, **kwargs)


def build_model(cfg) -> nn.Module:
    name = model_name(cfg)
    logger.info(f'Building network model {name}')
    cls_or_func = model_registry[name]
    attrs = cfg.models[name]
    if attrs:
        return cls_or_func(**attrs)
    return cls_or_func()


def build_optim(cfg, model):
    name = cfg.build[model_name(cfg)].train.solver
    logger.info(f'Building optimizer {name}')
    module = importlib.import_module(cfg.solvers.engine)
    return getattr(module, name)(model.parameters(), **cfg.solvers[name])


def build_scheduler(cfg, optimizer):
    name = cfg.build[model_name(cfg)].train.get('scheduler', None)
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


def build_func(func_name: str = 'default'):
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
        dataloader: DataLoader,
        optimizer=None,
        io_ops: Optional[Dict[str, Callable]] = None,
        **kwargs,
    ) -> None:
        self._scheduler = kwargs.pop('scheduler', None)
        super(TrainerEngine, self).__init__(process_func, **kwargs)

        # TODO: Move this to each builder function
        if is_distributed(cfg):
            model = idist.auto_model(model)
            optimizer = idist.auto_optim(optimizer)
            attrs = dict(cfg.datasets.dataloader)
            dataloader = idist.auto_dataloader(
                dataloader.dataset, collate_fn=build_func(attrs.pop('collate_fn', 'collate_fin')), **attrs
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

    @classmethod
    def build(cls, cfg, mode: Optional[str] = 'train') -> 'TrainerEngine':
        assert mode in MODES, f'Mode must be one of {MODES} but got {mode}'
        os.makedirs(cfg.workdir.path, exist_ok=True)
        model = build_model(cfg)
        optimizer = build_optim(cfg, model)
        io_ops = build_io(cfg)
        update_model = build_func(cfg.build.get('func', 'default'))
        dataloader = build_dataloader(cfg, mode)
        scheduler = build_scheduler(cfg, optimizer)
        return cls(cfg, update_model, model, dataloader, optimizer=optimizer, io_ops=io_ops, scheduler=scheduler)

    def __call__(self):
        train_cfg = self._cfg.build[model_name(self._cfg)].train
        epoch_length = train_cfg.get('iters_per_epoch', len(self._dataloader))
        self.run(self._dataloader, train_cfg.epochs, epoch_length=epoch_length)
        self._writer.close()

    def scheduler(self):
        if self._scheduler:
            self._scheduler.step()

    def summary(self):
        for key in self.state.metrics:
            if isinstance(self.state.metrics[key], str):
                continue
            self._writer.add_scalar(f'train/{key}', self.state.metrics[key], self.state.iteration)

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


class EvaluationEngine(Engine):
    def __init__(
        self,
        cfg,
        process_func,
        model,
        dataloader: DataLoader,
        io_ops: Optional[Dict[str, Callable]] = None,
        **kwargs,
    ) -> None:
        self._scheduler = kwargs.pop('scheduler', None)
        super(EvaluationEngine, self).__init__(process_func)

        # TODO: Move this to each builder function
        if is_distributed(cfg):
            model = idist.auto_model(model)
            optimizer = idist.auto_optim(optimizer)
            attrs = dict(cfg.datasets.dataloader)
            dataloader = idist.auto_dataloader(
                dataloader.dataset, collate_fn=build_func(attrs.pop('collate_fn', 'collate_fin')), **attrs
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


def build_validation(cfg, trainer_engine) -> TrainerEngine:
    if not cfg.build[model_name(cfg)].get('val', None):
        logger.warning('Not validation config found. Validation will be skipped')
        return

    logger.info('Adding validation')
    val_attrs = cfg.build[model_name(cfg)].val
    process_func = build_func(val_attrs.get('func', 'default_val_forward'))
    dataloader = build_dataloader(cfg, 'val')
    val_engine = EvaluationEngine(cfg, process_func, trainer_engine._model, dataloader)

    # evaluation metric
    metric_name = val_attrs.get('metric', None)
    if metric_name:
        build_func(metric_name)(val_engine, metric_name)

    step = val_attrs.get('step', None)
    epoch = val_attrs.get('epoch', 1)

    # TODO: Check if step and epochs are valid
    event_name = Events.EPOCH_COMPLETED(every=epoch)
    event_name = event_name | Events.ITERATION_COMPLETED(every=step) if step else event_name

    @trainer_engine.on(event_name | Events.STARTED)
    def _run_eval():
        logger.info('Running validation')
        val_engine()

        iteration = trainer_engine.state.iteration

        for key, value in val_engine.state.metrics.items():
            if isinstance(value, str):
                continue
            trainer_engine._writer.add_scalar(f'val/{key}', value, iteration)

        if metric_name:
            accuracy = val_engine.state.metrics[metric_name]
            print(f'Accuracy: {accuracy:.2f}')


def build_engine(cfg) -> TrainerEngine:
    engine = TrainerEngine.build(cfg, mode='train')
    build_validation(cfg, engine)
    return engine


def _trainer(rank, cfg) -> None:
    trainer = build_engine(cfg)
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
