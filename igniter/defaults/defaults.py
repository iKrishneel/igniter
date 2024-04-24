#!/usr/bin/env python

from typing import Any, Dict

import torch
from omegaconf import DictConfig

from igniter.registry import func_registry
from igniter.utils import convert_bytes_to_human_readable

__all__ = ['default_forward', 'default_val_forward', 'default_collate_fn']


@func_registry('default')
def default_forward(engine, batch) -> None:
    engine._model.train()
    inputs, targets = batch
    losses = engine._model(inputs, targets)
    n_size = len(losses)

    if isinstance(losses, dict):
        loss_sum = sum(losses.values())
        losses['total_loss'] = loss_sum
    else:
        loss_sum = losses
        losses = {'total_loss': loss_sum}

    engine._optimizer.zero_grad()
    loss_sum.backward()
    engine._optimizer.step()
    losses['lr'] = engine.get_lr()

    if torch.cuda.is_available():
        free, total = torch.cuda.mem_get_info()
        losses['gpu_mem'] = convert_bytes_to_human_readable(total - free)

    if n_size == 1:
        losses.pop('total_loss')
    engine.state.metrics = losses


@func_registry('default_evaluation')
@func_registry('default_val_forward')
def default_val_forward(engine, batch) -> Dict[str, Any]:
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


@func_registry('default_test')
def default_test(cfg: DictConfig) -> None:
    import cv2 as cv
    import matplotlib.pyplot as plt

    from igniter.builder import build_engine
    from igniter.visualizer import make_square_grid

    engine = build_engine(cfg, mode='test')

    image = cv.imread(cfg.image, cv.IMREAD_ANYCOLOR)

    pred = engine(image)
    pred = pred.cpu().numpy()
    pred = pred[0] if len(pred.shape) == 4 else pred

    if pred.shape[0] > 3:
        im_grid = make_square_grid(pred)
        plt.imshow(im_grid, cmap='jet')
    elif pred.shape[0] == 3:
        plt.imshow(pred.transpose((1, 2, 0)))
    else:
        plt.imshow(pred.transpose((1, 2, 0)), cmap='jet')
    plt.show()
