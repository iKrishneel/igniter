#!/usr/bin/env python

import functools
import importlib
import inspect
import os
from typing import Any, Callable, Dict, List, Optional, Union

import ignite.distributed as idist
import torch
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

MODES: List[str] = ['train', 'val', 'test', 'inference']


def build_func(func_name: Union[str, Dict[str, Any]] = 'default'):
    assert isinstance(func_name, (str, dict, DictConfig))
    if isinstance(func_name, (dict, DictConfig)):
        assert len(func_name) == 1, f'Current only supports one key but given {func_name.keys()}'
        for key, value in func_name.items():
            return functools.partial(func_registry[key], **(value or {}))
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


def build_transforms(cfg: DictConfig, name: Optional[str] = None) -> Union[List[Any], Dict[str, List[Any]]]:
    if name is not None:
        assert name in cfg.transforms, f'{name} not found. Available keys in transforms are: {cfg.transforms.keys()}'
    else:
        logger.warning('No data transformation!')
        return
    transforms_cfg = cfg.get('transforms', {})

    transforms: Dict[str, List[Any]] = {name: []}
    for key in transforms_cfg:
        attrs = dict(transforms_cfg[key])
        if name and key != name or attrs is None:
            continue

        engine = attrs.pop('engine', 'torchvision.transforms')
        module = importlib.import_module(engine)

        try:
            compose = module.v2.Compose
        except AttributeError:
            compose = module.Compose

        transform_list = []        
        for obj, kwargs in attrs.items():
            if 'compose' in obj.lower() and kwargs is not None:
                compose = transform_registry.get(kwargs) or compose
                continue
            transform = transform_registry[obj] if obj in transform_registry else getattr(module, obj)
            if inspect.isclass(transform):
                kwargs = kwargs or {}
                transform = transform(**kwargs)
            transform_list.append(transform)
        transforms[key] = compose(transform_list)

    return transforms[name] if name else transforms


@configurable
def build_dataloader(model_name: str, cfg: DictConfig, mode: str) -> DataLoader:
    logger.info(f'Building {mode} dataloader')

    key: str = 'transforms'
    build_kwargs = cfg.build[model_name]
    ds_name = build_kwargs.dataset
    logger.info(f'>>> Dataset: {ds_name}')

    cls = dataset_registry[ds_name]
    try:
        transforms = build_transforms(cfg, build_kwargs[mode].get(key) or build_kwargs.get(key) or mode)
    except AssertionError:
        logger.warning('No transforms for Dataloader')
        transforms = None

    dl_kwargs = dict(cfg.datasets.dataloader)

    collate_fn = build_func(dl_kwargs.pop('collate_fn', 'collate_fn'))
    attrs = cfg.datasets[ds_name].get(mode, None)
    dataset = cls(**{**dict(attrs), key: transforms})

    return DataLoader(dataset, collate_fn=collate_fn, **dl_kwargs)


@configurable
def build_model(name: str, cfg: DictConfig) -> nn.Module:
    logger.info(f'Building network model {name}')
    cls_or_func = model_registry[name]
    attrs = cfg.models[name] or {}
    dtype = getattr(torch, cfg.get('dtype', 'float32'))
    return cls_or_func(**attrs).to(dtype)


@configurable
def build_optim(model_name: str, cfg: DictConfig, model: nn.Module):
    name = cfg.build[model_name].train.solver
    logger.info(f'Building optimizer {name}')

    def global_lr_params(model, names: List[str]) -> List[torch.Tensor]:
        def has_name(name, names):
            has = False
            for n in names:
                has |= n in name
            return has
        return [param for name, param in model.named_parameters() if not has_name(name, names) and param.requires_grad]

    def legacy(cfg, name):
        engine = cfg.solvers.get('engine', 'torch.optim')
        module = importlib.import_module(engine)
        params = model.parameters()
        pp_args = dict(cfg.solvers[name])
        if 'per_parameter' in pp_args:
            per_parameter = pp_args.pop('per_parameter')
            layers = list(per_parameter.keys())
            params_dict = [{'params': global_lr_params(model, layers)}]
            for lname, value in per_parameter.items():
                pm = {'params': [p for n, p in model.named_parameters() if lname in n and p.requires_grad], **value}
                params_dict.append(pm)
            params = params_dict
        return getattr(module, name)(params, **pp_args)

    if name not in func_registry:
        return legacy(cfg, name)

    return func_registry[name](model, cfg.solvers[name])


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
    # TODO: iterate over entire config and enable adding event_handlers
    _cfg = cfg.build[model_name]
    attribut_name = 'event_handlers'

    def _build(events: List[Dict[str, str]]) -> None:
        for func_name in events:
            event_args = dict(events[func_name])
            event_type = event_args.pop('event_type')
            engine.add_event_handler(event_type, event_registry[func_name], **event_args)

    mode = cfg.build.get('mode', None)
    modes = [mode] if mode else MODES
    for mode in modes:
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

    def _build(io_cfg) -> Any:
        engine = io_cfg.engine
        cls = io_registry[engine]
        cls = importlib.import_module(engine) if cls is None else cls
        try:
            return cls.build(io_cfg, cfg)
        except AttributeError:
            return cls(io_cfg, cfg)
    return {key: _build(cfg.io[key]) for key in cfg.io}


def build_logger(cfg: DictConfig):
    pass


@configurable
def build_validation(model_name: str, cfg: DictConfig, trainer_engine: Engine) -> Union[Engine, None]:
    if not cfg.build[model_name].get('val', None):
        logger.warning('Not validation config found. Validation will be skipped')
        return None

    logger.info('Adding validation')
    val_attrs = cfg.build[model_name].val
    process_func = build_func(val_attrs.get('func', 'default_evaluation'))
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
def build_engine(model_name, cfg: DictConfig) -> Callable:
    mode = cfg.build.get('mode', 'train')
    assert mode in MODES, f'Invalid mode {mode}. Must be one of {MODES}'

    logger.info(f'>>> Building Engine with mode {mode}')
    validate_config(cfg)

    assert mode in MODES, f'Mode must be one of {MODES} but got {mode}'
    if 'workdir' in cfg:
        os.makedirs(cfg.workdir.path, exist_ok=True)

    yaml_data = OmegaConf.to_yaml(cfg)
    logger.info(f'\033[32m\n{yaml_data} \033[0m')

    mode_attrs = cfg.build[model_name].get(mode, None)

    key = 'forward' if 'forward' in mode_attrs else 'func'
    func_name = mode_attrs.get(key, 'default') if mode_attrs else 'default'

    process_func = build_func(func_name)
    model = build_model(cfg)
    logger.info(f'\n{model}')
    logger.info('\n' + loggable_model_info(model))

    # TODO: Remove hardcoded name and replace with registry based
    logger.warning('# TODO: Remove hardcoded name and replace with registry based')

    options = get_options(cfg)

    if mode in ['train', 'val']:
        io_ops = build_io(cfg)

        if options.train:
            dataloader = build_dataloader(cfg, mode)
            optimizer = build_optim(cfg, model)
            scheduler = build_scheduler(cfg, optimizer, dataloader)

            engine_name = cfg.build[model_name]['train'].get('engine', 'default_trainer')
            logger.info(f'>>> Trainer engine: {engine_name}')
            engine = engine_registry[engine_name](
                cfg, process_func, model, dataloader, optimizer=optimizer, io_ops=io_ops, scheduler=scheduler
            )

            if options.eval:
                build_validation(cfg, engine)
        elif options.eval:
            attrs = cfg.build[model_name].get('val', None)
            assert attrs, 'Validation attributes are required when options.eval=True'
            engine_name = attrs.get('engine') or 'default_evaluation'
            process_func = build_func(attrs.get('func', 'default_evaluation'))
            logger.info(f'>>> Evaluation engine: {engine_name}, {process_func}')
            dataloader = build_dataloader(cfg, 'val')
            engine = engine_registry[engine_name](cfg, process_func, model, dataloader, io_ops)
            engine._model.eval()
        else:
            logger.warning('Nothing to do as both eval and train are False')
            return
        module = importlib.import_module('igniter.engine.utils')
        if options.resume:
            module.load_all(engine, cfg)
        else:
            module.load_weights(model, cfg)
    elif mode in ['test', 'inference']:
        key = 'test' if 'test' in cfg.build[model_name] else 'inference'
        attrs = cfg.build[model_name].get(key, None)
        engine_name = attrs.get('engine', 'default_inference') if attrs else 'default_inference'

        engine_kwargs = {}
        if isinstance(engine_name, (dict, DictConfig)):
            assert len(engine_name) == 1, f'engine should have one key but got {len(engine_name)}'
            key = next(iter(engine_name))
            engine_kwargs = engine_name[key]
            engine_name = key
        
        logger.info(f'>>> Inference engine: {engine_name}')
        
        build_kwargs, key = cfg.build[model_name], 'transforms'
        transforms = build_transforms(cfg, build_kwargs[mode].get(key) or build_kwargs.get(key) or mode)
        engine = engine_registry[engine_name](cfg, model, transforms=transforms, **engine_kwargs)
    else:
        raise TypeError(f'Unknown mode {mode}')

    if engine:
        build_event_handlers(cfg, engine)

    return engine


def get_options(cfg: DictConfig) -> DictConfig:
    defaults = OmegaConf.create({'resume': False, 'eval': False, 'train': True, 'test': False})
    options = OmegaConf.merge(defaults, cfg.options)
    return options


def _trainer(rank: Union[int, None], cfg: DictConfig) -> None:
    trainer = build_engine(cfg)
    if trainer:
        trainer()
    else:
        logger.warning('Engine not available, Terminating execution!!')


def trainer(cfg: DictConfig) -> None:
    if is_distributed(cfg):
        init_args = dict(cfg.distributed[cfg.distributed.type])
        with idist.Parallel(
            backend=cfg.distributed.backend, nproc_per_node=cfg.distributed.nproc_per_node, **init_args
        ) as parallel:
            parallel.run(_trainer, cfg)
    else:
        _trainer(None, cfg)
