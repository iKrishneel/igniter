#!/usr/bin/env python

from torch.utils.tensorboard import SummaryWriter

from ..registry import io_registry


@io_registry('summary_writer')
def summary_writer(cfg=None, **kwargs) -> SummaryWriter:
    kwargs = dict(cfg.io.args) if cfg else kwargs
    return SummaryWriter(**kwargs)
