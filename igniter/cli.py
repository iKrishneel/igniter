#!/usr/bin/env python

from typing import List, Dict
import argparse
import os
import importlib
import logging
from igniter.logger import logger
from igniter.main import get_full_config, _run


def _find_files(directory: str, ext: str = 'py') -> List[str]:
    valid_files = []
    for root, _, files in os.walk(directory):
        for filename in files:
            if ext != filename.split('.')[1]:
                continue
            valid_files.append(os.path.join(root, filename))
    return valid_files


def _is_path(string: str) -> bool:
    return os.path.isabs(string) or os.path.exists(string)


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
    exec(code, globals())


def import_from_script(script_path: str, module_name: str) -> None:
    spec = importlib.util.spec_from_file_location(module_name, script_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    names = [name for name in dir(module) if not name.startswith('_')]
    globals().update({name: getattr(module, name) for name in names})


def main() -> None:
    parser = argparse.ArgumentParser(description='Igniter Command Line Interface (CLI)')
    parser.add_argument('config', type=str, help='Configuration filename')
    parser.add_argument('--log-level', type=str, default='INFO')
    args = parser.parse_args()

    logger.setLevel(getattr(logging, args.log_level))
    
    config_path = os.path.abspath(args.config)
    assert config_path.split('.')[1] == 'yaml', f'Config must be a yaml file but got {config_path}'
    logger.info(f'Using config {config_path}')

    cfg = get_full_config(config_path)

    if os.path.isdir(cfg.driver):
        raise NotImplementedError
    elif _is_path(cfg.driver):
        logger.info(f'Loading: {cfg.driver}')
        load_script(cfg.driver)
    else:
        logger.info(f'Importing: {cfg.driver}')
        import_modules(cfg.driver)

    _run(cfg)


if __name__ == '__main__':
    main()
