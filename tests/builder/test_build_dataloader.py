#!/usr/bin/env python

import os.path as osp

import pytest

# import torchvision.transforms as TF
from omegaconf import OmegaConf

from igniter.builder import build_dataloader


@pytest.fixture(scope='session')
def cfg():
    config_path = osp.join(osp.dirname(osp.abspath(__file__)), '../data/configs/datasets.yaml')
    assert osp.isfile(config_path)
    return OmegaConf.load(config_path)


def test_build_dataloader(cfg):
    for mode in ['train', 'val']:
        dataloader = build_dataloader(cfg, mode='train')
        assert dataloader
    # import IPython; IPython.embed()
