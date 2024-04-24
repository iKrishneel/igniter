#!/usr/bin/env python

import datetime
from typing import Any, Optional

import torch
from ignite.engine import Engine
from ignite.handlers.fbresearch_logger import MB
from ignite.handlers.fbresearch_logger import FBResearchLogger as _FBResearchLogger


class FBResearchLogger(_FBResearchLogger):
    def __init__(self, logger: Any, show_output: bool = True, **kwargs):
        super(FBResearchLogger, self).__init__(logger, show_output=show_output, **kwargs)

    def log_every(self, engine: Engine, optimizer: Optional[torch.optim.Optimizer] = None) -> None:
        """
        Logs the training progress at regular intervals.

        Args:
            engine: The training engine.
            optimizer: The optimizer used for training. Defaults to None.
        """
        assert engine.state.epoch_length is not None
        # cuda_max_mem = ""
        # if torch.cuda.is_available():
        #     cuda_max_mem = f"GPU Max Mem: {torch.cuda.max_memory_allocated() / MB:.0f} MB"

        current_iter = engine.state.iteration % (engine.state.epoch_length + 1)
        iter_avg_time = self.iter_timer.value()

        eta_seconds = iter_avg_time * (engine.state.epoch_length - current_iter)

        outputs = []
        if self.show_output and engine.state.metrics is not None:
            output = engine.state.metrics
            if isinstance(output, dict):
                outputs = [f"{k}: {v:.4f}" if not isinstance(v, str) else f"{k}: {v}" for k, v in output.items()]
            else:
                outputs = [f"{v:.4f}" if isinstance(v, float) else f"{v}" for v in output]  # type: ignore

        lrs = ""
        if optimizer is not None:
            if len(optimizer.param_groups) == 1:
                lrs += f"lr: {optimizer.param_groups[0]['lr']:.5f}"
            else:
                for i, g in enumerate(optimizer.param_groups):
                    lrs += f"lr [g{i}]: {g['lr']:.5f}"

        msg = self.delimiter.join(
            [
                f"Epoch [{engine.state.epoch}/{engine.state.max_epochs}]",
                f"[{current_iter}/{engine.state.epoch_length}]:",
                f"ETA: {datetime.timedelta(seconds=int(eta_seconds))}",
                f"{lrs}",
            ]
            + outputs
            + [
                f"Iter time: {iter_avg_time:.4f} s",
                f"Data prep time: {self.data_timer.value():.4f} s",
                # cuda_max_mem,
            ]
        )
        self.logger.info(msg)
