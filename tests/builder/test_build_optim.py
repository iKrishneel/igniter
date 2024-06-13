#!/usr/bin/env python

import os.path as osp

import pytest
import torch
from omegaconf import OmegaConf

from igniter.builder import build_model, build_optim


@pytest.fixture(scope='session')
def cfg():
    config_path = osp.join(osp.dirname(osp.abspath(__file__)), '../data/configs/config.yaml')
    assert osp.isfile(config_path)
    cfg = OmegaConf.load(config_path)
    OmegaConf.set_struct(cfg, False)
    return cfg


@pytest.fixture(scope='session')
def model(cfg):
    return build_model(cfg)


def test_build_optim_torch(model, cfg):
    optim = build_optim(cfg, model)
    assert isinstance(optim, torch.optim.SGD)


def test_build_optim_per_param(model, cfg):
    name = cfg.build.model
    cfg.build[name].train.solver = 'AdamW'
    optim = build_optim(cfg, model)
    assert isinstance(optim, torch.optim.AdamW)
