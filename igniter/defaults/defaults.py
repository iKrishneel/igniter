#!/usr/bin/env python

from typing import Dict
import torch
from igniter.registry import proc_registry


@proc_registry('default')
def default_forward(engine, batch) -> Dict[str, torch.Tensor]:
    engine._model.train()
    inputs, targets = batch
    engine._optimizer.zero_grad()
    losses = engine._model(inputs, targets)

    for key in losses:
        engine._io_ops.add_scalar(f'{key}', losses[key], engine.state.iteration)
        losses[key].backward()

    engine._optimizer.step()
    return losses['loss'].item()


@proc_registry('collate_fn')
def default_collate_fn(data):
    return data
