#!/usr/bin/env python

from typing import Any, Dict

from omegaconf import DictConfig

from ..logger import logger
from ..registry import io_registry


@io_registry('tqdm')
def tqdm_logger(**kwargs) -> Any:  
    try:
        from ignite.handlers import ProgressBar
    except ImportError:
        from ignite.contrib.handlers import ProgressBar

    return ProgressBar(**kwargs)


@io_registry('fair')
def fair_logger(**kwargs) -> Any:
    from .logger_handles import FBResearchLogger

    return FBResearchLogger(logger=logger, **kwargs)
