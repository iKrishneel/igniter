#!/usr/bin/env python

from typing import List, Dict, Any, Callable, Optional, Union
import inspect

import torch.nn as nn
from torch.utils.data import DataLoader

import importlib
import os
from omegaconf import DictConfig, OmegaConf, open_dict

from ignite.engine import Engine, Events
from ignite.handlers import BasicTimeProfiler
import ignite.distributed as idist

from igniter.utils import is_distributed, model_name, loggable_model_info
from igniter.logger import logger
from igniter.registry import (
    model_registry,
    dataset_registry,
    io_registry,
    func_registry,
    engine_registry,
    transform_registry,
)

MODES: List[str] = ['train', 'val', 'test']


def build_transforms(cfg: DictConfig, mode: Optional[str] = None) -> Union[List[Any], Dict[str, List[Any]]]:
    transforms = {mode: [] for mode in MODES}
    transforms_cfg = cfg.get('transforms', {})
    for key in transforms_cfg:
        attrs = dict(transforms_cfg[key])
        if mode and key != mode or attrs is None:
            continue

        engine = attrs.pop('engine', 'torchvision.transforms')
        module = importlib.import_module(engine)

        transform_list = []
        for obj, kwargs in attrs.items():
            transform = transform_registry[obj] if obj in transform_registry else getattr(module, obj)
            if inspect.isclass(transform):
                kwargs = kwargs or {}
                transform = transform(**kwargs)
            transform_list.append(transform)
        transforms[key] = module.Compose(transform_list)

    if mode:
        transforms = transforms[mode]
    return transforms


def build_dataloader(cfg: DictConfig, mode: str) -> DataLoader:
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


def build_model(cfg: DictConfig) -> nn.Module:
    name = model_name(cfg)
    logger.info(f'Building network model {name}')
    cls_or_func = model_registry[name]
    attrs = cfg.models[name] or {}
    return cls_or_func(**attrs)


def build_optim(cfg: DictConfig, model: nn.Module):
    name = cfg.build[model_name(cfg)].train.solver
    logger.info(f'Building optimizer {name}')
    engine = cfg.solvers.get('engine', 'torch.optim')
    module = importlib.import_module(engine)
    return getattr(module, name)(model.parameters(), **cfg.solvers[name])


def build_scheduler(cfg: DictConfig, optimizer: nn.Module, dataloader: DataLoader):
    name = cfg.build[model_name(cfg)].train.get('scheduler', None)
    if not name:
        return

    module = importlib.import_module('torch.optim.lr_scheduler')
    args = dict(cfg.solvers.schedulers[name])

    if name == 'OneCycleLR':
        args['steps_per_epoch'] = len(dataloader)

    return getattr(module, name)(optimizer=optimizer, **args)


def add_profiler(engine: Engine, cfg: DictConfig):
    profiler = BasicTimeProfiler()
    profiler.attach(engine)

    @engine.on(Events.ITERATION_COMPLETED(every=cfg.solvers.snapshot))
    def log_intermediate_results():
        profiler.print_results(profiler.get_results())

    return profiler


def build_io(cfg: DictConfig) -> Dict[str, Callable]:
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
    func = func_registry[func_name]
    assert func, f'Function {func_name} not found in registry \n{func_registry}'
    return func


def build_validation(cfg: DictConfig, trainer_engine: Engine) -> Engine:
    if not cfg.build[model_name(cfg)].get('val', None):
        logger.warning('Not validation config found. Validation will be skipped')
        return

    logger.info('Adding validation')
    val_attrs = cfg.build[model_name(cfg)].val
    process_func = build_func(val_attrs.get('func', 'default_val_forward'))
    dataloader = build_dataloader(cfg, 'val')
    val_engine = engine_registry['default_evaluation'](cfg, process_func, trainer_engine._model, dataloader)

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


def validate_config(cfg: DictConfig):
    # TODO: Validate all required fields and set defaults where missing
    with open_dict(cfg):
        if cfg.get('solvers', None):
            cfg.solvers.snapshot = cfg.solvers.get('snapshot', -1)

        trans_attrs = cfg.get('transforms', None)
        if trans_attrs:
            for key in trans_attrs:
                if 'engine' in key:
                    continue
                cfg.transforms[key].engine = 'torchvision.transforms'

    return cfg


def build_engine(cfg: DictConfig, mode: str = 'train') -> Callable:
    validate_config(cfg)

    assert mode in MODES, f'Mode must be one of {MODES} but got {mode}'
    os.makedirs(cfg.workdir.path, exist_ok=True)

    yaml_data = OmegaConf.to_yaml(cfg)
    logger.info(f'\033[32m\n{yaml_data} \033[0m')

    mode_attrs = cfg.build[model_name(cfg)].get(mode, None)
    func_name = mode_attrs.get('func', 'default') if mode_attrs else 'default'

    process_func = build_func(func_name)
    model = build_model(cfg)
    logger.info(f'\n{model}')
    logger.info('\n' + loggable_model_info(model))

    importlib.import_module('igniter.engine.utils').load_weights(model, cfg)

    # TODO: Remove hardcoded name and replace with registry based
    logger.warning('# TODO: Remove hardcoded name and replace with registry based')
    if mode == 'train':
        optimizer = build_optim(cfg, model)
        io_ops = build_io(cfg)
        dataloader = build_dataloader(cfg, mode)
        scheduler = build_scheduler(cfg, optimizer, dataloader)

        engine = engine_registry['default_trainer'](
            cfg, process_func, model, dataloader, optimizer=optimizer, io_ops=io_ops, scheduler=scheduler
        )
        build_validation(cfg, engine)
    else:
        attrs = cfg.build[model_name(cfg)].get('inference', None)
        name = attrs.get('engine', 'default_inference') if attrs else 'default_inference'
        engine = engine_registry[name](cfg)

    return engine


def _trainer(rank: int, cfg: DictConfig) -> None:
    trainer = build_engine(cfg)
    trainer()


def trainer(cfg: DictConfig):
    if is_distributed(cfg):
        init_args = dict(cfg.distributed[cfg.distributed.type])
        with idist.Parallel(
            backend=cfg.distributed.backend, nproc_per_node=cfg.distributed.nproc_per_node, **init_args
        ) as parallel:
            parallel.run(_trainer, cfg)
    else:
        _trainer(None, cfg)
