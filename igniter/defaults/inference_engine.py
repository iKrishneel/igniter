#!/usr/bin/env python

from glob import glob
import os.path as osp
from typing import Union

import numpy as np
from PIL import Image

import torch
from torchvision import transforms as T
from omegaconf import DictConfig, OmegaConf

from igniter.builder import build_model, build_transforms
from igniter.logger import logger

__all__ = ['InferenceEngine']


class InferenceEngine(object):
    def __init__(self, log_dir: str = None, config_file: str = None, **kwargs):
        assert log_dir or config_file, 'Must provide either the log_dir or the config file'

        if log_dir and not osp.isdir(log_dir):
            raise TypeError(f'Invalid log_dir {log_dir}')

        if config_file and not osp.isfile(config_file):
            raise TypeError(f'Invalid config_file {log_dir}')

        weights = kwargs.get('weights', None)
        if log_dir:
            extension = kwargs.get('extension', '.pt')
            config_file = config_file or osp.join(log_dir, 'config.yaml')
            weights = weights or sorted(glob(osp.join(log_dir, f'*{extension}')), reverse=True)[0]

        assert osp.isfile(config_file), f'Not Found: {config_file}'
        cfg: DictConfig = OmegaConf.load(config_file)

        self.model = build_model(cfg)
        self.transforms = kwargs.get('transforms', T.Compose([T.ToTensor()]))

        if cfg.build.get('inference', None):
            weights = weights or cfg.build.inference['weights']
            if cfg.build.inference.get('transforms'):
                self.transforms = build_transforms(cfg)[cfg.build.inference.transforms]

        if weights:
            logger.info(f'Weights: {weights}')
            weights = torch.load(weights, map_location=torch.device('cpu'))
            weights = weights['model'] if 'model' in weights else weights
            self.model.load_state_dict(weights)
        else:
            logger.warning('Weight is empty')

        if torch.cuda.is_available() and cfg.device.lower() != 'cpu':
            self.model.to(cfg.device)

        self.model.eval()
        self._device = cfg.device

    @torch.no_grad()
    def __call__(self, image: Union[np.ndarray, Image.Image]):
        image = Image.fromarray(image) if not isinstance(image, Image.Image) else image
        image = self.transforms(image)

        image = image[None, :] if len(image.shape) == 3 else image
        return self.model(image[:])
