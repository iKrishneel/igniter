#!/usr/bin/env python

import os
import argparse
import inspect
import functools

import hydra
from omegaconf import DictConfig

from .builder import trainer


"""
from .registry import func_registry


@func_registry('default_argument_parser')
def get_argument_parser(cfg: str = None) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument('--config-file', type=str, default=cfg, required=cfg is None)
    parser.add_argument('--weights', type=str, required=False, default=None)
    parser.add_argument('--log_dir', type=str, required=False, default=None)
    parser.add_argument('--test', action='store_true', default=False)
    parser.add_argument('--image', type=str, required=False)
    parser.add_argument('--viz', action='store_true', default=True)
    return parser
"""


def guard(func):
    @functools.wraps(func)
    def _wrapper(config_file: str = None):
        caller_frame = inspect.currentframe().f_back
        caller_module = inspect.getmodule(caller_frame).__name__
        if caller_module == '__main__':
            # args = func_registry['default_argument_parser'](config_file).parse_args()
            # if args.test:
            #     _test(args)
            # else:
            #     func(config_file, weights=args.weights)
            func(config_file)

    return _wrapper


@guard
def initiate(config_file: str):
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
        _run(cfg)

    _initiate()


def _run(cfg: DictConfig):
    mode = cfg.build.get('mode', 'train')
    if mode == 'train':
        trainer(cfg)
    elif mode in ['val', 'test', 'inference']:
        _test(cfg)
    else:
        raise TypeError(f'Unknown mode {mode}')


def _test(cfg: DictConfig) -> None:
    import matplotlib.pyplot as plt
    import cv2 as cv
    from igniter.builder import build_engine
    from igniter.visualizer import make_square_grid

    engine = build_engine(cfg, mode='inference')

    image = cv.imread(cfg.image, cv.IMREAD_ANYCOLOR)

    pred = engine(image)

    print(pred.min(), pred.max())

    im_grid = make_square_grid(pred.numpy())
    plt.imshow(im_grid)
    plt.show()
