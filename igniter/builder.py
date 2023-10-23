#!/usr/bin/env python

import functools
import importlib
import inspect
import os
from typing import Any, Callable, Dict, List, Optional, Union

import ignite.distributed as idist
import torch.nn as nn
from ignite.engine import Engine, Events
from ignite.handlers import BasicTimeProfiler
from omegaconf import DictConfig, OmegaConf, open_dict
from torch.utils.data import DataLoader

from igniter.logger import logger
from igniter.registry import (
    dataset_registry,
    engine_registry,
    event_registry,
    func_registry,
    io_registry,
    model_registry,
    transform_registry,
)
from igniter.utils import is_distributed, loggable_model_info, model_name

MODES: List[str] = ['train', 'val', 'test']


def build_func(func_name: str = 'default'):
    func = func_registry[func_name]
    assert func, f'Function {func_name} not found in registry \n{func_registry}'
    return func


def configurable(func: Callable):
    @functools.wraps(func)
    def wrapper(cfg: DictConfig, *args, **kwargs):
        name = model_name(cfg)
        assert name, 'build model name is required'
        return func(name, cfg, *args, **kwargs)

    return wrapper


def build_transforms(cfg: DictConfig, mode: Optional[str] = None) -> Union[List[Any], Dict[str, List[Any]]]:
    transforms: Dict[str, List[Any]] = {mode: [] for mode in MODES}
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

    return transforms[mode] if mode else transforms


@configurable
def build_dataloader(model_name: str, cfg: DictConfig, mode: str) -> DataLoader:
    logger.info(f'Building {mode} dataloader')

    name = cfg.build[model_name].dataset
    attrs = cfg.datasets[name].get(mode, None)
    kwargs = dict(cfg.datasets.dataloader)
    assert attrs, f'{mode} not found in datasets'

    cls = dataset_registry[name]
    transforms = build_transforms(cfg, mode)
    collate_fn = build_func(kwargs.pop('collate_fn', 'collate_fn'))
    dataset = cls(**{**dict(attrs), 'transforms': transforms})
    return DataLoader(dataset, collate_fn=collate_fn, **kwargs)


@configurable
def build_model(name: str, cfg: DictConfig) -> nn.Module:
    logger.info(f'Building network model {name}')
    cls_or_func = model_registry[name]
    attrs = cfg.models[name] or {}
    return cls_or_func(**attrs)


@configurable
def build_optim(model_name: str, cfg: DictConfig, model: nn.Module):
    name = cfg.build[model_name].train.solver
    logger.info(f'Building optimizer {name}')
    engine = cfg.solvers.get('engine', 'torch.optim')
    module = importlib.import_module(engine)
    return getattr(module, name)(model.parameters(), **cfg.solvers[name])


@configurable
def build_scheduler(model_name: str, cfg: DictConfig, optimizer: nn.Module, dataloader: DataLoader):
    name = cfg.build[model_name].train.get('scheduler', None)
    if not name:
        return

    module = importlib.import_module('torch.optim.lr_scheduler')
    args = dict(cfg.solvers.schedulers[name])

    if name == 'OneCycleLR':
        args['steps_per_epoch'] = len(dataloader)

    return getattr(module, name)(optimizer=optimizer, **args)


@configurable
def build_event_handlers(model_name: str, cfg: DictConfig, engine: Engine) -> None:
    _cfg = cfg.build[model_name]
    attribut_name = 'event_handlers'

    def _build(events: List[Dict[str, str]]) -> None:
        
        for event in events:
            for func_name, event_args in event.items():
                # TODO: Handle chained event types
                event_type = event_args.pop('event_type')
                engine.add_event_handler(event_type, event_registry[func_name], **event_args)
    
    for mode in MODES:
        if mode not in _cfg:
            continue
        _build(_cfg[mode].get(attribut_name, []))


def add_profiler(engine: Engine, cfg: DictConfig):
    profiler = BasicTimeProfiler()
    profiler.attach(engine)

    @engine.on(Events.ITERATION_COMPLETED(every=cfg.solvers.snapshot))
    def log_intermediate_results():
        profiler.print_results(profiler.get_results())

    return profiler


def build_io(cfg: DictConfig) -> Union[Dict[str, Callable], None]:
    if not cfg.get('io'):
        return None

    def _build(cfg):
        engine = cfg.engine
        cls = io_registry[engine]
        cls = importlib.import_module(engine) if cls is None else cls
        try:
            return cls.build(cfg)
        except AttributeError:
            return cls(cfg)

    return {key: _build(cfg.io[key]) for key in cfg.io}


@configurable
def build_validation(model_name: str, cfg: DictConfig, trainer_engine: Engine) -> Union[Engine, None]:
    if not cfg.build[model_name].get('val', None):
        logger.warning('Not validation config found. Validation will be skipped')
        return None

    logger.info('Adding validation')
    val_attrs = cfg.build[model_name].val
    process_func = build_func(val_attrs.get('func', 'default_val_forward'))
    dataloader = build_dataloader(cfg, 'val')
    val_engine = engine_registry['default_evaluation'](cfg, process_func, getattr(trainer_engine, '_model'), dataloader)

    # evaluation metric
    metric_name = val_attrs.get('metric', None)
    if metric_name:
        build_func(metric_name)(val_engine, metric_name)

    step = val_attrs.get('step', None)
    epoch = val_attrs.get('epoch', 1)

    # TODO: Check if step and epochs are valid
    event_type = Events.EPOCH_COMPLETED(every=epoch)
    event_type = event_type | Events.ITERATION_COMPLETED(every=step) if step else event_type  # type: ignore

    @trainer_engine.on(event_type | Events.STARTED)
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

    return trainer_engine


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


@configurable
def build_engine(model_name, cfg: DictConfig, mode: str = 'train') -> Callable:
    logger.info(f'>>> Building Engine with mode {mode}')
    validate_config(cfg)

    assert mode in MODES, f'Mode must be one of {MODES} but got {mode}'
    if 'workdir' in cfg:
        os.makedirs(cfg.workdir.path, exist_ok=True)

    yaml_data = OmegaConf.to_yaml(cfg)
    logger.info(f'\033[32m\n{yaml_data} \033[0m')

    mode_attrs = cfg.build[model_name].get(mode, None)
    func_name = mode_attrs.get('func', 'default') if mode_attrs else 'default'

    process_func = build_func(func_name)
    model = build_model(cfg)
    logger.info(f'\n{model}')
    logger.info('\n' + loggable_model_info(model))

    # importlib.import_module('igniter.engine.utils').load_weights(model, cfg)

    # TODO: Remove hardcoded name and replace with registry based
    logger.warning('# TODO: Remove hardcoded name and replace with registry based')
    if mode in ['train', 'val']:
        io_ops = build_io(cfg)
        dataloader = build_dataloader(cfg, mode)

        if mode == 'train':
            optimizer = build_optim(cfg, model)
            scheduler = build_scheduler(cfg, optimizer, dataloader)

            engine_name = cfg.build[model_name]['train'].get('engine', 'default_trainer')
            logger.info(f'>>> Trainer engine: {engine_name}')
            engine = engine_registry[engine_name](
                cfg, process_func, model, dataloader, optimizer=optimizer, io_ops=io_ops, scheduler=scheduler
            )

            build_validation(cfg, engine)

            module = importlib.import_module('igniter.engine.utils')
            if cfg.get('options', {}).get('resume'):
                module.load_all(engine, cfg)
            else:
                module.load_weights(model, cfg)
        else:
            attrs = cfg.build[model_name].get('val', None)
            engine_name = attrs.get('engine') or 'default_evaluation'
            logger.info(f'>>> Evaluation engine: {engine_name}, {process_func}')
            engine = engine_registry[engine_name](cfg, process_func, model, dataloader, io_ops)
    elif mode in ['test', 'inference']:
        key = 'test' if 'test' in cfg.build[model_name] else 'inference'
        attrs = cfg.build[model_name].get(key, None)
        engine_name = attrs.get('engine', 'default_inference') if attrs else 'default_inference'
        logger.info(f'>>> Inference engine: {engine_name}')
        engine = engine_registry[engine_name](cfg)
    else:
        raise TypeError(f'Unknown mode {mode}')

    if engine:
        build_event_handlers(cfg, engine)

    return engine


def _trainer(rank: Union[int, None], cfg: DictConfig) -> None:
    trainer = build_engine(cfg)
    trainer()


def trainer(cfg: DictConfig) -> None:
    if is_distributed(cfg):
        init_args = dict(cfg.distributed[cfg.distributed.type])
        with idist.Parallel(
            backend=cfg.distributed.backend, nproc_per_node=cfg.distributed.nproc_per_node, **init_args
        ) as parallel:
            parallel.run(_trainer, cfg)
    else:
        _trainer(None, cfg)
