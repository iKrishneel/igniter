#!/usr/bin/env python

from glob import glob
import os
import os.path as osp
from typing import Union, Any, Dict, Optional
from collections import OrderedDict

import numpy as np
from PIL import Image

import torch
from torchvision import transforms as T
from omegaconf import DictConfig, OmegaConf, open_dict

from igniter.builder import build_model, build_transforms, model_name
from igniter.registry import engine_registry
from igniter.io import S3Client
from igniter.logger import logger

__all__ = [
    'InferenceEngine',
]


@engine_registry('default_inference')
class InferenceEngine(object):
    def __init__(
        self, config_file: Optional[Union[str, DictConfig]] = None, log_dir: Optional[str] = None, **kwargs
    ) -> None:
        assert log_dir or config_file, 'Must provide either the log_dir or the config file'

        if log_dir and not osp.isdir(log_dir):
            raise TypeError(f'Invalid log_dir {log_dir}')

        if config_file and not isinstance(config_file, (str, DictConfig)):
            raise TypeError(f'Invalid config_file {config_file}')

        weights = kwargs.get('weights', None)

        if log_dir:
            extension = kwargs.get('extension', '.pt')
            filename = kwargs.get('config_filename', 'config.yaml')
            config_file = config_file or osp.join(log_dir, filename)
            weights = weights or sorted(glob(osp.join(log_dir, f'*{extension}')), reverse=True)[0]

        if isinstance(config_file, DictConfig):
            cfg = config_file
        else:
            assert osp.isfile(config_file), f'Not Found: {config_file}'
            cfg = OmegaConf.load(config_file)

        if weights:
            with open_dict(cfg):
                cfg.build[model_name(cfg)].weights = weights

        self.transforms = kwargs.get('transforms', T.Compose([T.ToTensor()]))
        inference_attrs = cfg.build[model_name(cfg)].get('inference', None)
        if inference_attrs:
            if inference_attrs.get('transforms', None):
                self.transforms = build_transforms(cfg)[inference_attrs.transforms]

        self.device = cfg.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f'Using device: {self.device}')

        self.model = build_model(cfg)
        self.load_weights(cfg)
        self.model.to(self.device)
        self.model.eval()

        logger.info('Inference Engine is Ready!')

    @torch.no_grad()
    def __call__(self, image: Union[np.ndarray, Image.Image]):
        assert image is not None, 'Input image is required'
        image = Image.fromarray(image) if not isinstance(image, Image.Image) else image
        image = self.transforms(image)

        image = image[None, :] if len(image.shape) == 3 else image
        return self.model(image.to(self.device))  # .squeeze(0).cpu()

    def load_weights(self, cfg: DictConfig, weight_key: str = 'model'):
        weight_path = cfg.build[model_name(cfg)].get('weights')
        if not weight_path or len(weight_path) == 0:
            logger.warning('Weight is empty!'.upper())
            return

        if 's3://' in weight_path:
            weight_dict = self._load_weights_from_s3(weight_path)
        else:
            weight_dict = self._load_weights_from_file(weight_path)

        assert weight_dict is not None

        def _remap_keys(weight_dict):
            new_wpth = OrderedDict()
            for key in weight_dict:
                new_key = key.replace('module.', '') if 'module.' in key else key
                new_wpth[new_key] = weight_dict[key]
            return new_wpth

        wpth = _remap_keys(weight_dict[weight_key])
        self.model.load_state_dict(wpth, strict=False)

    def _load_weights_from_s3(self, path: str) -> Dict[str, Any]:
        bucket_name = path[5:].split('/')[0]
        assert len(bucket_name) > 0, 'Invalid bucket name'

        path = path[5 + len(bucket_name) + 1 :]
        # check if weight is in cache
        root = osp.join(os.environ['HOME'], f'.cache/torch/{path}')
        if osp.isfile(root):
            logger.info(f'Cache found in cache, loading from {root}')
            return self._load_weights_from_file(root)

        s3_client = S3Client(bucket_name=bucket_name, decoder_func='decode_torch_weights')

        logger.info(f'Loading weights from {path}')
        weights = s3_client(path)

        # save weights to cache
        os.makedirs('/'.join(root.split('/')[:-1]), exist_ok=True)
        torch.save(weights, root)
        logger.info(f'Saved model weight to cache: {root}')

        return weights

    def _load_weights_from_file(self, path: str) -> Dict[str, torch.Tensor]:
        return torch.load(path, map_location='cpu')
