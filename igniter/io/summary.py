#!/usr/bin/env python

from torch.utils.tensorboard import SummaryWriter

from ..registry import io_registry


@io_registry('summary_writer')
def summary_writer(cfg) -> SummaryWriter:
    return SummaryWriter(**dict(cfg.io.args))
