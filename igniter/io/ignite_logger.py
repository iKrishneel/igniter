#!/usr/bin/env python

from typing import Any, Dict

from omegaconf import DictConfig

from ..logger import logger
from ..registry import io_registry


@io_registry('tqdm')
def tqdm_logger(io_cfg: DictConfig, cfg: DictConfig) -> Any:
    from ignite.handlers import ProgressBar

    attrs = dict(io_cfg)
    attrs.pop('engine')

    attach = attrs.pop('attach') if hasattr(attrs, 'attach') else {}
    attach['metric_names'] = attach.get('metric_names', 'all')

    p_logger = ProgressBar(**attrs)
    p_logger._attach_kwargs = attach
    return p_logger


@io_registry('fair')
def fair_logger(io_cfg: DictConfig, cfg: DictConfig) -> Any:
    from .logger_handles import FBResearchLogger

    attrs = dict(io_cfg)
    attrs.pop('engine')
    attach = dict(attrs.pop('attach')) if hasattr(attrs, 'attach') else {}
    attach['name'] = cfg.build.model

    fb_logger = FBResearchLogger(logger=logger, **attrs)
    fb_logger._attach_kwargs = attach
    return fb_logger