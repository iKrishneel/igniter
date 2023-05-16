#!/usr/bin/env python

from typing import Callable
from enum import Enum

import torch
from omegaconf import DictConfig


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
        return False
    return get_world_size(cfg) > 0 and torch.cuda.device_count() > 1 and cfg.distributed.nproc_per_node > 1


def check_str(string: str, msg: str = 'String is empty!'):
    assert len(string) > 0, msg


def get_collate_fn(cfg: DictConfig) -> Callable:
    try:
        collate_fn = cfg.datasets.dataloader.collate_fn
    except AttributeError:
        collate_fn = 'collate_fn'

    from igniter.registry import proc_registry

    return proc_registry.get(collate_fn)
