#!/usr/bin/env python

from glob import glob
import os
import os.path as osp
from typing import Union, Any, Dict

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
        weights = kwargs.get('weights', None)
        assert log_dir or config_file or weights, 'Must provide either the log_dir or the config file'

        if log_dir and not osp.isdir(log_dir):
            raise TypeError(f'Invalid log_dir {log_dir}')

        if config_file and not osp.isfile(config_file):
            raise TypeError(f'Invalid config_file {config_file}')

        if weights and 's3://' in weights:
            weights = self._load_weights_from_s3(weights)

        if log_dir:
            extension: str = kwargs.get('extension', '.pt')
            config_file: str = config_file or osp.join(log_dir, 'config.yaml')
            weights: str = weights or sorted(glob(osp.join(log_dir, f'*{extension}')), reverse=True)[0]

        assert osp.isfile(config_file), f'Not Found: {config_file}'
        cfg: DictConfig = OmegaConf.load(config_file)

        self.device = cfg.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f'Using device: {self.device}')

        model_name = cfg.build.model

        self.model = build_model(cfg)
        self.transforms = kwargs.get('transforms', T.Compose([T.ToTensor()]))

        inference_attrs = cfg.build[model_name].get('inference', None)
        if inference_attrs:
            weights = weights or inference_attrs['weights']
            if inference_attrs.get('transforms', None):
                self.transforms = build_transforms(cfg)[inference_attrs.transforms]

        if weights and isinstance(weights, str):
            logger.info(f'Weights: {weights}')
            weights = self._load_weights_from_file(weights)
            # weights = weights[weight_key] if weight_key and len(weight_key) > 0 else weights

        if weights:
            from collections import OrderedDict

            weight_key = kwargs.get('weight_key', 'model')
            wpth = weights[weight_key]
            new_wpth = OrderedDict()
            for key in wpth:
                new_key = key.replace('module.', '') if 'module.' in key else key
                new_wpth[new_key] = wpth[key]

            self.model.load_state_dict(new_wpth, strict=True)
        else:
            logger.warning('Weight is empty!'.upper())

        self.model.to(self.device)
        self.model.eval()

        logger.info('Inference Engine is Ready!')

    @torch.no_grad()
    def __call__(self, image: Union[np.ndarray, Image.Image]):
        image = Image.fromarray(image) if not isinstance(image, Image.Image) else image
        image = self.transforms(image)

        image = image[None, :] if len(image.shape) == 3 else image
        return self.model(image.to(self.device)).squeeze(0).cpu()

    def _load_weights_from_s3(self, path: str) -> Dict[str, Any]:
        from igniter.io import S3Client

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


from argparse import Namespace


def build_inference_engine(cfg: DictConfig = None, args: Namespace = None) -> InferenceEngine:
    if cfg:
        raise NotImplementedError('Building inference engine using cfg is not yet implemented')

    if args:
        return InferenceEngine(config_file=args.config_file, weights=args.weights, log_dir=args.log_dir)
