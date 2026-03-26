#!/usr/bin/env python

import importlib
import os.path as osp

from omegaconf import DictConfig

from igniter.logger import logger
from igniter.utils import find_pattern


def load_modules(cfg: DictConfig) -> None:
    if not hasattr(cfg, 'driver'):
        return
    if osp.isdir(cfg.driver):
        raise NotImplementedError
    elif _is_path(cfg.driver):
        logger.info(f'Loading: {cfg.driver}')
        load_script(cfg.driver)
    else:
        logger.info(f'Importing: {cfg.driver}')
        import_modules(cfg.driver)


def import_modules(module: str) -> bool:
    try:
        importlib.import_module(module)
        has_import = True
    except ImportError:
        has_import = False
    return has_import


def load_script(path: str) -> None:
    with open(path, 'r') as script:
        code = script.read()

    is_empty = lambda x: len(list(x)) == 0
    matches = find_pattern(code, r'from \.')
    if not is_empty(matches):
        raise TypeError(f'Relatively import is not supported! Found relative import in {path}')

    exec(code, globals())


def _is_path(string: str) -> bool:
    return osp.isabs(string) or osp.exists(string)
