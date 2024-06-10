#!/usr/bin/env python

import importlib
import os
import os.path as osp
from collections import OrderedDict
from typing import Any, Callable, Dict, Tuple, Union

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

    if 's3://' in weight_path:
        state_dict = load_weights_from_s3(weight_path, kwargs.get('decoder', None))
    elif weight_path.startswith('https://drive.google.com/'):
        state_dict = load_weights_from_gdrive(weight_path)
    elif weight_path.startswith('http://') or weight_path.startswith('https://'):
        state_dict = load_weights_from_url(weight_path)
    else:
        state_dict = load_weights_from_file(weight_path)

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


def get_path_or_load(filename: str) -> Tuple[str, Dict[str, Any]]:
    root = osp.join(os.environ['HOME'], f'.cache/torch/{filename}') if not osp.isfile(filename) else filename
    if osp.isfile(root):
        logger.info(f'Cache found in cache, loading from {root}')
        return root, load_weights_from_file(root)

    os.makedirs(osp.dirname(root), exist_ok=True)
    return root, None


def load_weights_from_s3(path: str, decoder: Union[Callable[..., Any], str, None] = None) -> Dict[str, Any]:
    bucket_name = path[5:].split('/')[0]
    assert len(bucket_name) > 0, 'Invalid bucket name'

    path = path[5 + len(bucket_name) + 1 :]
    root, weights = get_path_or_load(path)

    if weights is not None:
        return weights

    s3_client = S3Client(bucket_name=bucket_name)

    logger.info(f'Loading weights from {path}')
    if decoder:
        weights = s3_client(path, decoder=decoder)
        save_weights(weights, root)
    else:
        s3_client.download(path, root)
        weights = load_weights_from_file(root)  # type: ignore

    logger.info(f'Saved model weight to cache: {root}')

    return weights  # type: ignore


def load_weights_from_url(url: str) -> Dict[str, torch.Tensor]:
    import requests

    logger.info(f'Loading weights from {url}')

    filename = osp.basename(url)
    root, weights = get_path_or_load(filename)

    if weights is not None:
        return weights

    response = requests.get(url)
    with open(root, 'wb') as f:
        f.write(response.content)

    return load_weights_from_file(root)


def load_weights_from_gdrive(url: str):
    import gdown
    import requests
    from bs4 import BeautifulSoup

    def get_id(url):
        parts = url.split('/')
        try:
            index = parts.index('d')
        except ValueError:
            try:
                index = parts.index('file') + 1
            except ValueError:
                return None
        return parts[index + 1]

    def get_filename(html_response: str) -> str:
        soup = BeautifulSoup(html_response, 'html.parser')
        return soup.find('span', class_='uc-name-size').find('a').text

    url = 'https://drive.google.com/uc?id=' + get_id(url)
    session = requests.session()
    res = session.get(url, stream=True, verify=False)

    filename = get_filename(res.content)

    save_path, weights = get_path_or_load(filename)
    if weights is not None:
        return weights

    gdown.download(url, save_path, quiet=False)
    return get_path_or_load(save_path)[1]


def load_weights_from_file(path: str) -> Dict[str, torch.Tensor]:
    assert osp.isfile(path), f'Not weight found {path}'

    if use_safe_tensors(path):
        from safetensors import safe_open

        weights = OrderedDict()
        with safe_open(path, framework='pt', device='cpu') as f:
            for key in f.keys():
                weights[key] = f.get_tensor(key)
        assert len(weights) > 0, f'No weights loaded from {path}'
    else:
        weights = torch.load(path, map_location='cpu')
    return weights


def save_weights(weights: Dict[str, Any], root: str) -> None:
    save_func = torch.save
    if use_safe_tensors(root):
        from safetensors.torch import save_file

        save_func = save_file

    save_func(weights, root)


def use_safe_tensors(path: str) -> bool:
    is_safe_tensor = '.safetensors' in path

    try:
        importlib.import_module('safetensors.torch').save_file
    except ModuleNotFoundError as e:
        logger.warning(f'{e}')
        is_safe_tensor = False

    return is_safe_tensor
