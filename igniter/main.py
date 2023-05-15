#!/usr/bin/env python

import os
import hydra
from omegaconf import DictConfig

from .builder import trainer


def initiate(config_file: str):
    assert os.path.isfile(config_file), f'Config file not found {config_file}'
    config_name = config_file.split(os.sep)[-1]
    config_path = config_file.replace(config_name, '')
    config_name = config_name.split('.')[0]

    config_path = os.path.abspath(config_path) if not os.path.isabs(config_path) else config_path

    @hydra.main(version_base=None, config_path=config_path, config_name=config_name)
    def _initiate(cfg: DictConfig):
        trainer(cfg)

    _initiate()
