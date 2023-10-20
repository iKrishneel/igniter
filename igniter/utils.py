#!/usr/bin/env python

import os.path as osp
from enum import Enum
from typing import Union

import torch
from omegaconf import DictConfig, ListConfig, OmegaConf

from .logger import logger


class Node(Enum):
    SINGLE = 'single'
    MULTI = 'multi'


def get_world_size(cfg: DictConfig) -> int:
    nproc = cfg.distributed.nproc_per_node
    if cfg.distributed.type == Node.SINGLE.value:
        world_size = nproc
    else:
        world_size = nproc * cfg.distributed.dist_config.nnodes
    return world_size


def is_distributed(cfg: DictConfig) -> bool:
    if not torch.cuda.is_available():
        logger.warning('No CUDA Available!')
        return False
    if 'distributed' not in cfg:
        return False
    return get_world_size(cfg) > 0 and torch.cuda.device_count() > 1 and cfg.distributed.nproc_per_node > 1


def get_device(cfg: DictConfig) -> torch.device:
    device = cfg.get('device', 'cuda')
    if device == 'cuda' and not torch.cuda.is_available():
        logger.warning(f'{device} not available')
        device = 'cpu'
    return torch.device(device)


def check_str(string: str, msg: str = 'String is empty!'):
    assert len(string) > 0, msg


def convert_bytes_to_human_readable(nbytes: float) -> str:
    suffixes = ['B', 'KB', 'MB', 'GB', 'TB', 'PB']
    i = 0
    while nbytes >= 1024 and i < len(suffixes) - 1:
        nbytes /= 1024
        i += 1
    return f'{nbytes:.2f} {suffixes[i]}'


def model_name(cfg: DictConfig) -> str:
    return cfg.build.model


def get_config(filename: str) -> Union[DictConfig, ListConfig]:
    assert osp.isfile(filename), f'Config file {filename} not found!'
    return OmegaConf.load(filename)


def loggable_model_info(model: torch.nn.Module) -> str:
    from tabulate import tabulate

    total_params, trainable_params = 0, 0
    for param in model.parameters():
        total_params += param.shape.numel()
        trainable_params += param.shape.numel() if param.requires_grad else 0

    header = ['Parameters', '#']
    table = [
        ['Non-Trainable', f'{total_params - trainable_params: ,}'],
        ['Trainable', f'{trainable_params: ,}'],
        ['Total', f'{total_params: ,}'],
    ]
    return tabulate(table, header, tablefmt='grid')
