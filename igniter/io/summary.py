#!/usr/bin/env python

from torch.utils.tensorboard import SummaryWriter

from ..registry import io_registry


@io_registry('summary_writer')
def summary_writer(*args, **kwargs) -> SummaryWriter:
    return SummaryWriter(*args, **kwargs)
