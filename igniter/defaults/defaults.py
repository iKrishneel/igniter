#!/usr/bin/env python

from typing import Dict
import torch
from igniter.registry import proc_registry


@proc_registry('default')
def default_forward(engine, batch) -> Dict[str, torch.Tensor]:
    engine._model.train()
    inputs, targets = batch
    engine._optimizer.zero_grad()
    loss = engine._model(inputs)
    loss.backward()
    engine._optimizer.step()
    return loss.item()
