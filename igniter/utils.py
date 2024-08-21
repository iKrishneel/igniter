#!/usr/bin/env python

import os.path as osp
import re
from enum import Enum
from typing import Tuple, Union

import torch
from omegaconf import DictConfig, ListConfig, OmegaConf

from .logger import logger


class Node(Enum):
    SINGLE = 'single'
    MULTI = 'multi'


def _get_world_size(cfg: DictConfig) -> int:
    nproc = cfg.distributed.nproc_per_node
    if cfg.distributed.type == Node.SINGLE.value:
        world_size = nproc
    else:
        world_size = nproc * cfg.distributed.dist_config.nnodes
    return world_size


def get_world_size(cfg: DictConfig = None) -> int:
    if cfg is not None:
        return _get_world_size(cfg)
    if not is_dist_avail_and_initalized():
        return 0
    return torch.distributed.get_world_size()


def is_dist_avail_and_initalized() -> bool:
    if not torch.distributed.is_available():
        return False
    if not torch.distributed.is_initialized():
        return False
    return True


def get_world_size_and_rank() -> Tuple[int, ...]:
    world_size = get_world_size()
    return world_size, get_rank()


def get_rank() -> int:
    if not is_dist_avail_and_initalized():
        return 0
    return torch.distributed.get_rank()


def is_main_process() -> bool:
    return get_rank() == 0


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


def get_dir_and_file_name(path: str, abs_path: bool = True, remove_ext: bool = True) -> Tuple[str, str]:
    dirname, filename = osp.dirname(path), osp.basename(path)
    filename = osp.splitext(filename)[0] if remove_ext else filename
    dirname = osp.abspath(dirname) if abs_path and not osp.isabs(dirname) else dirname
    return dirname, filename


def find_pattern(text: str, pattern: str):
    assert all(isinstance(arg, str) for arg in (text, pattern))

    matches = re.finditer(pattern, text)
    return matches


def find_replace_pattern(text: str, replacement: str, pattern: str):
    assert all(isinstance(arg, str) for arg in (text, pattern, replacement))

    return re.sub(pattern, replacement, text)


def resolve_env_vars(string: str, pattern: str = r'\$(\w+)|\$\{(\w+)\}') -> str:
    import os

    if '$' not in string:
        return string

    def env_var_replacer(match: str):
        var_name = match.group(1)
        try:
            return os.environ.get(var_name, match.group(0))
        except TypeError:
            logger.warning(f'{var_name} not found')

    resolved_string = re.sub(pattern, env_var_replacer, string)
    return osp.normpath(resolved_string)
