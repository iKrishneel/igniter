#!/usr/bin/env python

import os
import argparse
import inspect
import functools

import hydra
from omegaconf import DictConfig

from .builder import trainer


def get_argparser(cfg: str = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--config-file', type=str, default=cfg, required=cfg is None)
    parser.add_argument('--weights', type=str, required=False, default=None)
    parser.add_argument('--log_dir', type=str, required=False, default=None)
    parser.add_argument('--test', action='store_true', default=False)
    parser.add_argument('--image', type=str, required=False)
    parser.add_argument('--viz', action='store_true', default=True)

    return parser.parse_args()


def guard(func):
    @functools.wraps(func)
    def _wrapper(cfg):
        caller_frame = inspect.currentframe().f_back
        caller_name = inspect.getframeinfo(caller_frame).function
        caller_module = inspect.getmodule(caller_frame).__name__
        if caller_module == '__main__':
            print(cfg)
            args = get_argparser(cfg)
            if args.test:
                initiate_test(args)
            else:
                return func(cfg)

        return func

    return _wrapper


@guard
def initiate(config_file: str):
    assert os.path.isfile(config_file), f'Config file not found {config_file}'
    config_name = config_file.split(os.sep)[-1]
    config_path = config_file.replace(config_name, '')
    config_name = config_name.split('.')[0]

    config_path = os.path.abspath(config_path) if not os.path.isabs(config_path) else config_path

    @hydra.main(version_base=None, config_path=config_path, config_name=config_name)
    def _initiate(cfg: DictConfig):
        trainer(cfg)

    _initiate()


def initiate_test(args: argparse.Namespace) -> None:
    import cv2 as cv
    from igniter.defaults import build_inference_engine
    from igniter.visualizer import make_square_grid

    engine = build_inference_engine(args=args)

    image = cv.imread(args.image, cv.IMREAD_ANYCOLOR)

    pred = engine(image)

    if args.viz:
        im_grid = make_square_grid(pred.numpy())

        import matplotlib.pyplot as plt

        plt.imshow(im_grid)
        plt.show()
