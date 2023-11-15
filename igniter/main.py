#!/usr/bin/env python

import functools
import inspect
import os
import subprocess
from copy import deepcopy
from typing import Any, Callable, Dict, Tuple

import hydra
from omegaconf import DictConfig, OmegaConf, open_dict

from .builder import trainer
from .logger import logger
from .utils import get_dir_and_file_name


def guard(func: Callable) -> Callable:
    @functools.wraps(func)
    def _wrapper(config_file: str = ''):
        caller_frame = getattr(inspect.currentframe(), 'f_back', None)
        assert caller_frame is not None
        caller_module = getattr(inspect.getmodule(caller_frame), '__name__', None)
        assert caller_module is not None
        caller_filename = getattr(inspect.getframeinfo(caller_frame), 'filename', None)
        assert caller_filename is not None
        absolute_path = os.path.abspath(caller_filename)

        if caller_module == '__main__':
            func(config_file, absolute_path)

    return _wrapper


def configure(func: Callable) -> Callable:
    @functools.wraps(func)
    def _wrapper(cfg: DictConfig, config_file: str, caller_path: str = '', **kwargs: Dict[str, Any]) -> Any:
        cfg = _full_config(cfg, config_file)
        return func(cfg, caller_path, **kwargs)

    return _wrapper


def _full_config(cfg: DictConfig, config_file: str, config_dir: str = None) -> DictConfig:
    OmegaConf.set_struct(cfg, False)

    config_dir = os.path.join(
        hydra.utils.get_original_cwd(), os.path.dirname(config_file)
    ) if config_dir is None else config_dir
    if '_base_' in cfg:
        filename = os.path.join(config_dir, cfg._base_)
        base_cfg = read_base_configs(filename)
        cfg = OmegaConf.merge(base_cfg, cfg)
    else:
        cfg = load_default(cfg)

    # TODO: load _file_ attributes for keys            
    # cfg = load_values_from_file(config_dir, cfg)

    OmegaConf.set_struct(cfg, True)
    return cfg


def load_values_from_file(config_dir: str, cfg: DictConfig) -> DictConfig:
    """
    Current implementation only supports top level keys to use this
    """
    assert os.path.isdir(config_dir), f'Invalid directory {config_dir}'

    for key in cfg:
        value = cfg[key]
        if isinstance(value, str) and '.yaml' in value:
            filename = value if os.path.isabs(value) else os.path.join(config_dir, value)
            assert os.path.isfile(filename), f'{value} file not found at {filename}'
            conf = OmegaConf.load(filename)
            cfg[key] = conf[key]
        if isinstance(value, DictConfig):
            load_values_from_file(config_dir, value)

    return cfg


def load_default(cfg: DictConfig) -> DictConfig:
    default_cfg = OmegaConf.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'configs/config.yaml'))
    remove_keys = [key for key in default_cfg if not default_cfg[key]]
    for key in remove_keys:
        default_cfg.pop(key)
    return OmegaConf.merge(default_cfg, cfg)


def read_base_configs(filename: str) -> DictConfig:
    assert os.path.isfile(filename), f'File not found {filename}'
    cfg = OmegaConf.load(filename)
    base_cfg = read_base_configs(
        os.path.join(os.path.dirname(filename), cfg._base_)
    ) if '_base_' in cfg else load_default({})
    cfg = OmegaConf.merge(base_cfg, cfg)
    return cfg


def get_full_config(config_file: str, caller_path: str = '') -> DictConfig:
    from hydra import compose, initialize_config_dir

    config_path, config_name = get_dir_and_file_name(config_file, True)
    with initialize_config_dir(version_base=None, config_dir=config_path, job_name='full_config'):
        cfg = compose(config_name=config_name)
        cfg = _full_config(cfg, config_file, config_dir=config_path)
    return cfg


@guard
def initiate(config_file: str, caller_path: str = '') -> None:
    assert os.path.isfile(config_file), f'Config file not found {config_file}'

    config_path, config_name = get_dir_and_file_name(config_file)
    kwargs = dict(version_base=None, config_path=config_path, config_name=config_name)
    if hydra.__version__ < '1.2':
        kwargs.pop('version_base', None)

    @hydra.main(**kwargs)
    def _initiate(cfg: DictConfig):
        run_flow(cfg, config_file, caller_path)

    _initiate()


@configure
def run_flow(cfg: DictConfig, caller_path: str = '') -> None:
    with open_dict(cfg):
        flows = cfg.pop('flow', None)

    if not flows:
        return _run(cfg)

    cfg_copy = deepcopy(cfg)

    directory = '/tmp/igniter/.flow/'
    os.makedirs(directory, exist_ok=True)
    for flow in flows:
        with open_dict(cfg_copy):
            cfg_copy.build.model = flow

        filename = f'{flow}.yaml'
        OmegaConf.save(cfg_copy, os.path.join(directory, filename))

        logger.info(f'Starting workflow for model {flow}')
        if not _exec(caller_path, directory, filename):
            raise RuntimeError(f'Process {flow} didnt complete successfully')
        logger.info(f'{"-" * 80}')


def _exec(caller_path: str, directory: str, filename: str) -> bool:
    assert os.path.isfile(caller_path)
    assert os.path.isdir(directory)
    assert os.path.isfile(os.path.join(directory, filename))

    config_name = filename.split('.')[0]
    process = subprocess.run(['python', caller_path, '--config-path', directory, '--config-name', config_name])

    return process.returncode == 0


def _run(cfg: DictConfig) -> None:
    mode = cfg.build.get('mode', 'train')
    if mode in ['train', 'val']:
        trainer(cfg)
    elif mode in ['test', 'inference']:
        from igniter.registry import func_registry
        from igniter.utils import model_name

        func_name = cfg.build.get(model_name(cfg)).inference.get('func', 'default_test')
        logger.info(f'Inference function {func_name}')

        func = func_registry[func_name]
        assert func is not None, f'{func_name} not found! Registerd are \n{func_registry}'
        func(cfg)
    else:
        raise TypeError(f'Unknown mode {mode}')
