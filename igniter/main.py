#!/usr/bin/env python

import functools
import inspect
import os
import subprocess
from copy import deepcopy
from typing import Callable

import hydra
from omegaconf import DictConfig, OmegaConf, open_dict

from .builder import trainer
from .logger import logger


def guard(func: Callable):
    @functools.wraps(func)
    def _wrapper(config_file: str = None):
        caller_frame = inspect.currentframe().f_back
        caller_module = inspect.getmodule(caller_frame).__name__

        caller_filename = inspect.getframeinfo(caller_frame).filename
        absolute_path = os.path.abspath(caller_filename)

        if caller_module == '__main__':
            func(config_file, absolute_path)

    return _wrapper


@guard
def initiate(config_file: str, caller_path: str = None) -> None:
    assert os.path.isfile(config_file), f'Config file not found {config_file}'
    config_name = config_file.split(os.sep)[-1]
    config_path = config_file.replace(config_name, '')
    config_name = config_name.split('.')[0]

    config_path = os.path.abspath(config_path) if not os.path.isabs(config_path) else config_path

    kwargs = dict(version_base=None, config_path=config_path, config_name=config_name)
    if hydra.__version__ < '1.2':
        kwargs.pop('version_base', None)

    @hydra.main(**kwargs)
    def _initiate(cfg: DictConfig):
        run_flow(cfg, caller_path)

    _initiate()


def run_flow(cfg: DictConfig, caller_path: str = None) -> None:
    with open_dict(cfg):
        flows = cfg.pop('flow', None)

    if not flows:
        return _run(cfg)

    cfg_copy = deepcopy(cfg)

    directory = '/tmp/igniter/flow/'
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
    if mode == 'train':
        trainer(cfg)
    elif mode in ['val', 'test', 'inference']:
        from igniter.registry import func_registry
        from igniter.utils import model_name

        func_name = cfg.build.get(model_name(cfg)).inference.get('func', 'default_test')
        logger.info(f'Inference function {func_name}')

        func = func_registry[func_name]
        assert func is not None, f'{func_name} not found! Registerd are \n{func_registry}'
        func(cfg)
    else:
        raise TypeError(f'Unknown mode {mode}')
