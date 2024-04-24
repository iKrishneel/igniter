#!/usr/bin/env python

import os
import os.path as osp
from collections import OrderedDict
from typing import Any, Callable, Dict, Union

import torch
import torch.nn as nn
from omegaconf import DictConfig

from ..io import S3Client
from ..logger import logger
from ..utils import model_name

__all__ = ['load_weights', 'load_weights_from_s3', 'load_weights_from_file']


def _remap_keys(weight_dict) -> OrderedDict:
    new_wpth = OrderedDict()
    for key in weight_dict:
        new_key = key.replace('module.', '') if 'module.' in key else key
        new_wpth[new_key] = weight_dict[key]
    return new_wpth


def get_weights(cfg: DictConfig, **kwargs: Dict[str, Any]) -> Union[Dict[str, Any], None]:
    weight_path = cfg.build[model_name(cfg)].get('weights', None) if isinstance(cfg, DictConfig) else cfg
    return get_weights_util(weight_path, **kwargs)


def get_weights_util(weight_path: str, **kwargs: Dict[str, Any]):
    if not weight_path or len(weight_path) == 0:
        logger.warning('Weight is empty!'.upper())
        return None

    state_dict = (
        load_weights_from_s3(weight_path, kwargs.get('decoder', None))  # type: ignore
        if 's3://' in weight_path
        else load_weights_from_file(weight_path)
    )
    assert state_dict is not None, 'Weight dict is None'
    return state_dict


def load_all(engine, cfg: DictConfig, **kwargs: Dict[str, Any]):
    state_dict = get_weights(cfg, **kwargs)
    if not state_dict:
        return

    engine._model.load_state_dict(state_dict['model'])

    if 'optimizer' in state_dict and engine._optimizer:
        engine._optimizer.load_state_dict(state_dict['optimizer'])

    if 'scheduler' in state_dict and engine._scheduler:
        engine._scheduler.load_state_dict(state_dict['scheduler'])

    if 'state' in state_dict:
        engine.state = state_dict['state']


def load_weights(model: nn.Module, cfg: DictConfig, **kwargs):
    weight_dict = get_weights(cfg, **kwargs)
    if not weight_dict:
        return

    state_dict = model.state_dict()
    wpth = _remap_keys(weight_dict['model'] if not isinstance(weight_dict, OrderedDict) else weight_dict)

    for key in state_dict:
        if key not in wpth or state_dict[key].shape == wpth[key].shape:
            continue
        logger.warning(f'Shape missmatch key {key} {state_dict[key].shape} != {wpth[key].shape}')
        # wpth.pop(key)

    load_status = model.load_state_dict(wpth, strict=kwargs.get('strict', False))
    logger.info(f'{load_status}')


def load_weights_from_s3(path: str, decoder: Union[Callable[..., Any], str, None] = None) -> Dict[str, Any]:
    bucket_name = path[5:].split('/')[0]
    assert len(bucket_name) > 0, 'Invalid bucket name'

    path = path[5 + len(bucket_name) + 1 :]
    # check if weight is in cache
    root = osp.join(os.environ['HOME'], f'.cache/torch/{path}')
    if osp.isfile(root):
        logger.info(f'Cache found in cache, loading from {root}')
        return load_weights_from_file(root)

    s3_client = S3Client(bucket_name=bucket_name)
    os.makedirs('/'.join(root.split('/')[:-1]), exist_ok=True)

    logger.info(f'Loading weights from {path}')
    if decoder:
        weights = s3_client(path, decoder=decoder)
        save_weights(weights, root)
    else:
        s3_client.download(path, root)
        weights = load_weights_from_file(root)  # type: ignore

    logger.info(f'Saved model weight to cache: {root}')

    return weights  # type: ignore


def load_weights_from_file(path: str) -> Dict[str, torch.Tensor]:
    assert osp.isfile(path), f'Not weight found {path}'
    return torch.load(path, map_location='cpu')


def save_weights(weights: Dict[str, Any], root: str) -> None:
    torch.save(weights, root)
