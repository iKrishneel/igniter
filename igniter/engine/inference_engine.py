#!/usr/bin/env python

import os.path as osp
from glob import glob
from typing import Any, Dict, Optional, Union

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf, open_dict
from PIL import Image
from torchvision import transforms as T

from igniter.builder import build_model, build_transforms, model_name
from igniter.logger import logger
from igniter.registry import engine_registry

from .utils import load_weights

__all__ = [
    'InferenceEngine',
]


@engine_registry('default_inference')
class InferenceEngine(object):
    def __init__(
        self,
        config_file: Optional[Union[str, DictConfig]] = None,
        model: torch.nn.Module = None,
        log_dir: Optional[str] = None,
        **kwargs,
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
            assert config_file and osp.isfile(config_file), f'Not Found: {config_file}'
            cfg: DictConfig = OmegaConf.load(config_file)  # type: ignore

        if weights:
            with open_dict(cfg):
                cfg.build[model_name(cfg)].weights = weights

        self.transforms = kwargs.get('transforms', T.Compose([T.ToTensor()]))
        inference_attrs = cfg.build[model_name(cfg)].get('inference', None)
        if inference_attrs:
            if inference_attrs.get('transforms', None):
                transforms = build_transforms(cfg, inference_attrs.transforms)
                self.transforms = transforms if isinstance(transforms, T.Compose) else self.transforms

        self.device = cfg.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f'Using device: {self.device}')

        model = build_model(cfg) if model is None else model
        load_weights(model, cfg, strict=kwargs.get('strict', True))
        model.to(self.device)
        model.to(getattr(torch, cfg.dtype))
        model.eval()
        # self._model = torch.compile(model) if hasattr(torch, 'compile') else model
        self._model = model

        logger.info('Inference Engine is Ready!')

    @torch.inference_mode()
    def __call__(self, image: Union[np.ndarray, Image.Image, torch.Tensor], **kwargs: Dict[str, Any]):
        assert image is not None, 'Input image is required'

        if not isinstance(image, torch.Tensor):
            image = Image.fromarray(image) if not isinstance(image, Image.Image) else image

        if self.transforms:
            image = self.transforms(image)

        image = image[None, :] if len(image.shape) == 3 else image  # type: ignore
        return self.model(image.to(self.device), **kwargs)  # type: ignore

    @property
    def model(self) -> torch.nn.Module:
        return self._model
