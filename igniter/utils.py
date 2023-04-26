#!/usr/bin/env python

from enum import Enum

import torch

from omegaconf import DictConfig


class Node(Enum):
    SINGLE = 'single'
    MULTI = 'multi'


def get_world_size(cfg: DictConfig) -> int:
    if cfg.distributed.type == Node.SINGLE.value:
        world_size = cfg.distributed.single.nproc_per_node
    else:
        world_size = cfg.distributed.dist_config.nproc_per_node * cfg.distributed.dist_config.nnodes
    return world_size


def is_distributed(cfg: DictConfig) -> bool:
    return get_world_size(cfg) > 0 and torch.cuda.device_count() > 1


def check_str(string: str, msg: str = 'String is empty!'):
    assert len(string) > 0, msg
