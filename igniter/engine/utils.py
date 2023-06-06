#!/usr/bin/env python

import os
import os.path as osp

from typing import Any, Dict
from collections import OrderedDict

import torch
import torch.nn as nn
from omegaconf import DictConfig

from ..logger import logger
from ..io import S3Client
from ..utils import model_name


__all__ = ['load_weights', 'load_weights_from_s3', 'load_weights_from_file']


def load_weights(model: nn.Module, cfg: DictConfig, **kwargs):
    if isinstance(cfg, DictConfig):
        weight_path = cfg.build[model_name(cfg)].get('weights', None)
    else:
        weight_path = cfg

    if not weight_path or len(weight_path) == 0:
        logger.warning('Weight is empty!'.upper())
        return

    if 's3://' in weight_path:
        weight_dict = load_weights_from_s3(weight_path)
    else:
        weight_dict = load_weights_from_file(weight_path)

    assert weight_dict is not None

    def _remap_keys(weight_dict):
        new_wpth = OrderedDict()
        for key in weight_dict:
            new_key = key.replace('module.', '') if 'module.' in key else key
            new_wpth[new_key] = weight_dict[key]
        return new_wpth

    state_dict = model.state_dict()
    for key in weight_dict:
        if any([k in weight_dict[key] for k in ['state', 'param_groups']]):
            continue
        # TODO: check if current key has keys similar to state_dict
        weight_key = key
        break

    wpth = _remap_keys(weight_dict[weight_key])

    for key in state_dict:
        if key not in wpth or state_dict[key].shape == wpth[key].shape:
            continue
        logger.info(f'Removing shape missmatch key {key}')
        wpth.pop(key)

    load_status = model.load_state_dict(wpth, strict=kwargs.get('strict', False))
    logger.info(f'{load_status}')


def load_weights_from_s3(path: str) -> Dict[str, Any]:
    bucket_name = path[5:].split('/')[0]
    assert len(bucket_name) > 0, 'Invalid bucket name'

    path = path[5 + len(bucket_name) + 1 :]
    # check if weight is in cache
    root = osp.join(os.environ['HOME'], f'.cache/torch/{path}')
    if osp.isfile(root):
        logger.info(f'Cache found in cache, loading from {root}')
        return load_weights_from_file(root)

    s3_client = S3Client(bucket_name=bucket_name, decoder_func='decode_torch_weights')

    logger.info(f'Loading weights from {path}')
    weights = s3_client(path)

    # save weights to cache
    os.makedirs('/'.join(root.split('/')[:-1]), exist_ok=True)
    torch.save(weights, root)
    logger.info(f'Saved model weight to cache: {root}')

    return weights


def load_weights_from_file(path: str) -> Dict[str, torch.Tensor]:
    return torch.load(path, map_location='cpu')
