#!/usr/bin/env python

import torch
from igniter.utils import convert_bytes_to_human_readable
from igniter.registry import func_registry

__all__ = ['default_forward', 'default_val_forward', 'default_collate_fn']


@func_registry('default')
def default_forward(engine, batch) -> None:
    engine._model.train()
    inputs, targets = batch
    engine._optimizer.zero_grad()
    losses = engine._model(inputs, targets)

    for key in losses:
        losses[key].backward()

    engine._optimizer.step()
    losses['lr'] = engine.get_lr()

    if torch.cuda.is_available():
        free, total = torch.cuda.mem_get_info()
        losses['gpu_mem'] = convert_bytes_to_human_readable(total - free)

    engine.state.metrics = losses


@func_registry('default_val_forward')
def default_val_forward(engine, batch) -> None:
    engine._model.eval()
    inputs, targets = batch

    with torch.no_grad():
        output, losses = engine._model(inputs, targets)

    if torch.cuda.is_available():
        free, total = torch.cuda.mem_get_info()
        losses['gpu_mem'] = convert_bytes_to_human_readable(total - free)

    engine.state.metrics = losses

    return {'y_pred': output, 'y_true': targets}


@func_registry('collate_fn')
def default_collate_fn(data):
    return data
