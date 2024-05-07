#!/usr/bin/env python

import argparse
import importlib
import logging
import os
import os.path as osp
import sys
from collections import OrderedDict
from typing import List, Type

from omegaconf import DictConfig, open_dict

from igniter.builder import build_engine
from igniter.logger import logger
from igniter.main import _run as igniter_run
from igniter.main import get_full_config

Namespace = Type[argparse.Namespace]


def _find_files(directory: str, ext: str = 'py') -> List[str]:
    valid_files = []
    for root, _, files in os.walk(directory):
        for filename in files:
            if ext != filename.split('.')[1]:
                continue
            valid_files.append(osp.join(root, filename))
    return valid_files


def _is_path(string: str) -> bool:
    return osp.isabs(string) or osp.exists(string)


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


def get_config(args: Namespace) -> DictConfig:
    config_path = osp.abspath(args.config)
    assert config_path.split('.')[1] == 'yaml', f'Config must be a yaml file but got {config_path}'
    logger.info(f'Using config {config_path}')
    return get_full_config(config_path)


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


def train_val_run(args: Namespace, is_train: bool) -> None:
    cfg = get_config(args)

    with open_dict(cfg):
        cfg.build.mode = 'train' if is_train else 'val'
        cfg.options.train = is_train
        cfg.options.eval = not is_train

    load_modules(cfg)
    igniter_run(cfg)


def test_run(args: Namespace) -> None:
    from igniter.registry import func_registry

    from .defaults.image_io import Inference

    cfg = get_config(args)
    with open_dict(cfg):
        cfg.build.mode = 'inference'        
        if args.weights:
            cfg.build[cfg.build.model].weights = args.weights

    load_modules(cfg)
    engine = build_engine(cfg)
    visualizer = func_registry[args.visualizer] if args.visualizer else None
    inference = Inference(args.input, engine, visualizer, input_fmt=args.format, save=args.save, save_dir=args.save_dir)
    inference.run()


def export(args: Namespace) -> None:
    weights, output = args.weights, args.output
    utils = importlib.import_module('igniter.engine.utils')
    state_dict = utils.get_weights_util(weights)

    assert osp.isfile(weights), f'Invalid model path {weights}'
    filename = osp.join(osp.dirname(weights), f'exported_{osp.basename(weights)}')

    if output:
        directory = osp.dirname(filename)
        _, ext = osp.splitext(output)
        filename = output if bool(ext) else filename.replace(directory, _)

    filename = osp.normpath(filename)
    os.makedirs(osp.dirname(filename), exist_ok=True)

    if isinstance(state_dict, dict):
        for _, value in state_dict.items():
            if not isinstance(value, OrderedDict):
                continue
            state_dict = value
            break

    assert isinstance(state_dict, OrderedDict)
    logger.info(f'Saving weights to {filename}')
    utils.save_weights(state_dict, filename)


def main() -> None:
    parser = argparse.ArgumentParser(description='Igniter Command Line Interface (CLI)')
    # parser.add_argument('config', type=str, help='Configuration filename')
    parser.add_argument('--log-level', type=str, default='INFO')

    sub_parsers = parser.add_subparsers(dest='options', help='Options')

    train_parser = sub_parsers.add_parser('train', help='Description for training args')
    train_parser.add_argument('config', type=str, help='Configuration filename')

    eval_parser = sub_parsers.add_parser('eval', help='Description for evaluation args')
    eval_parser.add_argument('config', type=str, help='Configuration filename')
    eval_parser.add_argument('--weights', type=str, required=False, help='Path to weight file')

    test_parser = sub_parsers.add_parser('test', help='Description for test args')
    test_parser.add_argument('config', type=str, help='Configuration filename')
    test_parser.add_argument('input', type=str, help='Path to the input file')
    test_parser.add_argument('--weights', type=str, required=False)    
    test_parser.add_argument('--visualizer', type=str, required=False)
    test_parser.add_argument('--format', type=str, default='RGB')
    test_parser.add_argument('--save', type=str, required=False)
    test_parser.add_argument('--save_dir', type=str, required=False, default='output/')
    test_parser.add_argument('--thresh', type=float, default=0.1)

    export_parser = sub_parsers.add_parser('export', help='Exports the train model for inference')
    # train_parser.add_argument('config', type=str, help='Configuration filename')
    export_parser.add_argument('weights', type=str, help='Path to the trained model file with extension .pt/.pth')
    export_parser.add_argument('--output', type=str, required=False, help='Output name or directory')

    args = parser.parse_args()

    logger.setLevel(getattr(logging, args.log_level))

    if args.options is None:
        print('\033[32m')
        parser.print_help()
        print('\033[0m')
        sys.exit()

    if args.options in ['train', 'eval']:
        train_val_run(args, args.options == 'train')
    elif args.options == 'export':
        export(args)
    elif args.options == 'test':
        test_run(args)


if __name__ == '__main__':
    main()
