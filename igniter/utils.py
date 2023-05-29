#!/usr/bin/env python

from enum import Enum

import torch
from omegaconf import DictConfig

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
    return get_world_size(cfg) > 0 and torch.cuda.device_count() > 1 and cfg.distributed.nproc_per_node > 1


def check_str(string: str, msg: str = 'String is empty!'):
    assert len(string) > 0, msg


def convert_bytes_to_human_readable(nbytes: int) -> str:
    suffixes = ['B', 'KB', 'MB', 'GB', 'TB', 'PB']
    i = 0
    while nbytes >= 1024 and i < len(suffixes) - 1:
        nbytes /= 1024
        i += 1
    return f'{nbytes:.2f} {suffixes[i]}'


def model_name(cfg: DictConfig) -> str:
    return cfg.build.model
