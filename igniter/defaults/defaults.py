#!/usr/bin/env python

import torch
from igniter.utils import convert_bytes_to_human_readable
from igniter.registry import proc_registry


@proc_registry('default')
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
        _, usage = torch.cuda.mem_get_info()
        losses['gpu_mem'] = convert_bytes_to_human_readable(usage)

    engine.state.metrics = losses


@proc_registry('collate_fn')
def default_collate_fn(data):
    return data