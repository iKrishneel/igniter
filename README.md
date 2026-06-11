# 🔥 Igniter

**A config-driven deep learning training framework built on [PyTorch Ignite](https://pytorch-ignite.ai/).**

Igniter lets you go from a model definition to a fully working training, evaluation, and inference pipeline with almost no boilerplate. You register your model, dataset, and any custom functions with simple decorators, describe the experiment in a single YAML file, and Igniter builds and runs everything — dataloaders, optimizers, schedulers, checkpointing, logging, distributed training, and inference.

[![Python](https://img.shields.io/badge/python-3.9%20%7C%203.10%20%7C%203.11-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)

---

## Table of Contents

- [Key Features](#key-features)
- [Installation](#installation)
- [How It Works](#how-it-works)
- [Quick Start: MNIST](#quick-start-mnist)
  - [1. Define and register components](#1-define-and-register-components)
  - [2. Write the config](#2-write-the-config)
  - [3. Train](#3-train)
- [Command Line Interface](#command-line-interface)
- [Configuration Reference](#configuration-reference)
- [Registries](#registries)
- [Checkpointing and S3 Support](#checkpointing-and-s3-support)
- [Inference](#inference)
- [Distributed Training](#distributed-training)
- [Project Structure](#project-structure)
- [Development](#development)
- [License](#license)

---

## Key Features

- **Config-driven workflows** — the entire experiment (model, data, transforms, solver, schedulers, logging, checkpointing) is described in one YAML file, powered by [Hydra](https://hydra.cc/) and [OmegaConf](https://omegaconf.readthedocs.io/).
- **Registry system** — register models, datasets, transforms, functions, event handlers, and engines with one-line decorators; reference them by name in the config.
- **Built on PyTorch Ignite** — training/validation engines, events, and metrics come from the battle-tested `pytorch-ignite` ecosystem.
- **Zero-boilerplate defaults** — sensible default forward passes, collate functions, checkpoint handlers, and loggers (`tqdm`, FAIR-style console logging, TensorBoard summaries) are provided out of the box and can be overridden by name.
- **Config inheritance** — compose configs with the `_base_` key; child configs are merged on top of parents and the built-in defaults.
- **Multi-stage flows** — chain multiple training stages (e.g., pre-train → fine-tune → distill) using the `flow` key; each stage runs as its own process.
- **CLI included** — `igniter train / eval / test / export` works on any project that has a config file with a `driver` entry.
- **S3 I/O** — read datasets from and write checkpoints directly to Amazon S3 (`s3://bucket/path`).
- **Distributed training** — single- and multi-node training via `ignite.distributed` with a few config lines.
- **Model export** — export trained checkpoints to plain weight files or [safetensors](https://github.com/huggingface/safetensors).

## Installation

Requires **Python ≥ 3.9** and PyTorch.

```bash
# From source
git clone https://github.com/iKrishneel/igniter.git
cd igniter
pip install -e .
```

Key dependencies (installed automatically): `pytorch-ignite>=0.4.12`, `hydra-core>=1.2`, `omegaconf`, `tensorboard`, `boto3`, `safetensors`, `opencv-python`, `pycocotools`.

## How It Works

Igniter follows a three-part pattern:

```
┌──────────────────┐     ┌──────────────────┐     ┌─────────────────────┐
│  Your Python     │     │   YAML Config     │     │   Igniter Builder   │
│  (driver script) │ ──▶ │  (experiment      │ ──▶ │  builds & runs the  │
│                  │     │   description)    │     │  Ignite engines     │
│ @model_registry  │     │ models, datasets, │     │ train / val / test  │
│ @dataset_registry│     │ solvers, io, ...  │     │                     │
└──────────────────┘     └──────────────────┘     └─────────────────────┘
```

1. **Register** your components (model, dataset, collate function, metrics…) using decorators from `igniter.registry`.
2. **Describe** the experiment in YAML: which registered components to use and with what parameters.
3. **Run** with `initiate('<config>.yaml')` in your script, or via the `igniter` CLI. The builder reads the config, looks everything up in the registries, wires it together, and launches the Ignite training loop.

## Quick Start: MNIST

The complete example lives in [`example/mnist.py`](example/mnist.py) with its config in [`example/configs/mnist.yaml`](example/configs/mnist.yaml).

### 1. Define and register components

```python
#!/usr/bin/env python

import torch
import torch.nn as nn
import torch.nn.functional as F

from igniter import initiate
from igniter.registry import dataset_registry, model_registry, proc_registry


# ---- Model -------------------------------------------------------------
# Register the model under the name "mnist" so the config can refer to it.
@model_registry('mnist')
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x, targets=None):
        device = self.conv1.weight.device
        x = x.to(device)

        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        x = F.log_softmax(x, dim=-1)

        # Training: return loss dict | Validation: return (output, losses)
        if targets is not None:
            losses = self.losses(x, targets.to(device))
            if self.training:
                return losses
            return x, losses

        return x

    def losses(self, x, targets):
        return {'loss': F.nll_loss(x, targets)}


# ---- Dataset -----------------------------------------------------------
# Any callable that returns a torch Dataset can be registered.
@dataset_registry('mnist')
def mnist_dataset(**kwargs):
    from torchvision.datasets import MNIST

    transform = kwargs.pop('transforms')
    return MNIST(transform=transform, **kwargs)


# ---- Collate function ---------------------------------------------------
@proc_registry('mnist_collate_fn')
def collate_fn(data):
    images, targets = [], []
    for d in data:
        images.append(d[0])
        targets.append(d[1])

    images = torch.stack(images)
    targets = torch.Tensor(targets).long()
    return images, targets


# ---- Validation metric ---------------------------------------------------
# Attach any ignite.metrics metric to the evaluation engine.
@proc_registry('accuracy')
def metric(engine, name):
    from ignite.metrics import Accuracy

    def _output_transform(data):
        data['y'] = data.pop('y_true')
        data['y'] = data['y'].to(data['y_pred'].device)
        return data

    Accuracy(output_transform=_output_transform).attach(engine, name)


# ---- Launch --------------------------------------------------------------
initiate('./configs/mnist.yaml')
```

A few conventions worth noting:

- **Model contract.** With the built-in `default` training step, `model(inputs, targets)` should return a **dict of losses** in training mode (they are summed, backpropagated, and logged automatically). In eval mode it should return `(outputs, losses)`. With `targets=None` (inference), it returns raw outputs. You can replace this behavior entirely by registering your own forward function (see [`func` in the config](#configuration-reference)).
- **Dataset contract.** The registered callable receives all the keyword arguments from the config's `datasets.<name>.<mode>` block, plus the composed `transforms`.
- **Metric contract.** A metric function receives `(engine, name)` and attaches an Ignite metric to the evaluation engine.

### 2. Write the config

`example/configs/mnist.yaml`:

```yaml
driver: "../mnist.py"        # script whose registrations should be loaded (used by the CLI)
device: "cuda"               # cpu | cuda

workdir:
  path: "/tmp/dist/mnist"    # working directory for logs/artifacts

transforms:
  train: &trans
    engine: "torchvision.transforms"   # module to import transforms from
    ToTensor:                          # class name → kwargs (empty = defaults)
    Normalize:
      mean: [0.1307]
      std: [0.3081]
  val:
    <<: *trans                         # plain YAML anchors work for reuse

datasets:
  dataloader:                          # kwargs forwarded to torch DataLoader
    batch_size: 128
    num_workers: 4
    collate_fn: 'mnist_collate_fn'     # name in the function registry
  mnist:                               # must match the dataset registry name
    val:
      root: '/tmp/data/'
      train: False
      download: True
    train:
      root: '/tmp/data/'
      train: True
      download: True

models:
  mnist:                               # name in model registry → constructor kwargs

solvers:
  snapshot: 10000
  SGD:                                 # any optimizer from torch.optim, by class name
    lr: 0.01
    momentum: 0.5
  schedulers:
    StepLR:                            # any scheduler from torch.optim.lr_scheduler
      step_size: 20
      gamma: 0.1

io:
  checkpoint:
    engine: 'file_writer'              # or 's3_writer' for S3 (see below)
    root: "/tmp/mnist/models/"
  log_handler:
    engine: 'fair'                     # 'fair' (console) or 'tqdm' progress logging
    attach:
      every: 10                        # log every N iterations

build:
  mnist:                               # build spec for the model named "mnist"
    dataset: "mnist"                   # which registered dataset to use
    train:
      solver: "SGD"                    # references solvers.SGD above
      scheduler: "StepLR"
      epochs: 10
      func: "default"                  # training step function (registry name)
      transforms: "train"              # which transforms block to apply
      event_handlers:
        default_checkpoint_handler:    # save weights on every epoch
          event_type: EPOCH_COMPLETED
          root: /tmp/weights/
          prefix: mnist_
    val:
      epoch: 1                         # run validation every N epochs
      metric: "accuracy"               # registered metric to attach
      transforms: "val"

  model: "mnist"                       # which build entry to run

options:
  train: True
  eval: True

flow:                                  # optional: ordered list of build stages
  - mnist
```

Any key you omit falls back to Igniter's [built-in defaults](igniter/configs/config.yaml) (CPU device, batch size 1, `tqdm` logging, checkpoints to `./weights/`, etc.). You can also inherit from another YAML file by adding `_base_: path/to/base.yaml`.

### 3. Train

Run the driver script directly:

```bash
cd example
python mnist.py
```

Or use the CLI from anywhere (it loads the script referenced by the config's `driver` key):

```bash
igniter train example/configs/mnist.yaml
```

You'll see the FAIR-style logger printing loss, learning rate, and GPU memory every 10 iterations, validation accuracy at the end of each epoch, and checkpoints written as `mnist_0000001.pt`, `mnist_0000002.pt`, … to the checkpoint root.

Since Igniter uses Hydra under the hood, you can override any config value from the command line when running the driver script:

```bash
python mnist.py solvers.SGD.lr=0.001 build.mnist.train.epochs=20
```

## Command Line Interface

Installing the package provides an `igniter` command:

| Command | Description |
|---|---|
| `igniter train <config.yaml> [--bs N] [--workers N]` | Train using the given config; optionally override batch size and dataloader workers. |
| `igniter eval <config.yaml> [--weights PATH]` | Run evaluation only. |
| `igniter test <config.yaml> <input> [--weights PATH] [--format RGB] [--save NAME] [--save_dir DIR]` | Run inference on an input (e.g., image or folder) using the configured inference runner. |
| `igniter export <weights.pt> [--output PATH] [--safe-tensor]` | Extract model weights from a training checkpoint into a standalone file, optionally as safetensors. |
| `igniter --log-level DEBUG ...` | Set logging verbosity. |

The CLI uses the config's `driver` field to import your script (so all your registrations are loaded) before building anything. The `driver` can be either a path to a `.py` file or an importable module name.

## Configuration Reference

| Section | Purpose |
|---|---|
| `driver` | Path or module containing your registered components. Required for the CLI. |
| `device` | `cpu` or `cuda`. |
| `dtype` | Model dtype, e.g. `float32` (default). |
| `workdir.path` | Working/log directory. |
| `transforms.<name>` | Named transform pipelines. `engine` selects the module (default `torchvision.transforms.v2`); each subsequent key is a transform class name with its kwargs. Custom transforms registered in `transform_registry` take precedence over module lookups. |
| `datasets.dataloader` | Kwargs for `torch.utils.data.DataLoader` plus `collate_fn` (a function-registry name). |
| `datasets.<name>.<mode>` | Kwargs passed to the registered dataset callable for each mode (`train` / `val` / `test`). |
| `models.<name>` | Constructor kwargs for the registered model. |
| `solvers.<OptimName>` | Optimizer kwargs; the key is a `torch.optim` class name (or a name in `solver_registry`). |
| `solvers.schedulers.<SchedName>` | LR scheduler kwargs; the key is a `torch.optim.lr_scheduler` class name. |
| `io.checkpoint` | Checkpoint writer: `engine: file_writer` with a local `root`, or `engine: s3_writer` with `bucket_name` and `root`. |
| `io.log_handler` | `engine: fair` (rich console logging) or `engine: tqdm` (progress bar); `attach.every` controls frequency. TensorBoard summary writers are also available via the io registry. |
| `build.<model>` | Per-model build spec: `dataset`, `train` (solver, scheduler, epochs, step `func`, transforms, `event_handlers`), `val` (frequency, `metric`, transforms), `inference` (engine, transforms, hooks), optional `weights` to initialize from a checkpoint. |
| `build.model` | Which build entry to run. |
| `options.train / options.eval` | Enable or disable the train/eval phases. |
| `flow` | Ordered list of build entries to run sequentially as separate processes (multi-stage pipelines). |
| `distributed` | Distributed settings: `backend` (e.g. `nccl`), `type` (`single`/`multiple`), `nproc_per_node`, init method. |
| `_base_` | Path to a parent config to inherit from (merged recursively). |

## Registries

All registries live in `igniter.registry` and share the same decorator API:

```python
from igniter.registry import model_registry

@model_registry('my_model')      # explicit name
class MyModel(nn.Module): ...

@model_registry                  # or implicit: registered as "MyOtherModel"
class MyOtherModel(nn.Module): ...
```

| Registry | What goes in it |
|---|---|
| `model_registry` | `nn.Module` classes or factory functions. |
| `dataset_registry` | Dataset classes or factory functions. |
| `func_registry` (alias `proc_registry`) | Training/validation step functions, collate functions, metrics, inference hooks. |
| `transform_registry` | Custom data transforms (referenced by name inside `transforms`). |
| `event_registry` | Ignite event handlers (e.g., `default_checkpoint_handler`). |
| `engine_registry` | Custom trainer/inference engines. |
| `io_registry` | Checkpoint/log writers (`file_writer`, `s3_writer`, summary writers). |
| `solver_registry` | Custom optimizers (falls back to `torch.optim` by class name). |
| `runner_registry` | Inference runners used by `igniter test`. |

Useful built-ins already registered in `func_registry`: `default` (training step), `default_val_forward` / `default_evaluation` (validation step), `collate_fn` (identity collate), `default_test` (simple image inference + visualization).

## Checkpointing and S3 Support

Checkpoints are written by the `default_checkpoint_handler` event (configured under `build.<model>.train.event_handlers`) and/or the `io.checkpoint` writer.

Write directly to S3 by switching the writer:

```yaml
io:
  checkpoint:
    engine: 's3_writer'
    bucket_name: "my-bucket"
    root: "models/mnist/"
```

An `s3://bucket/path` value in `root` or `weights` is also detected automatically — weights can be loaded straight from S3 (or an HTTP URL) at build time. Datasets stored on S3 can be consumed via the bundled `S3Client` / `S3Dataset` utilities (`igniter.io`, `igniter.datasets`). AWS credentials are resolved by `boto3`'s standard mechanisms. If an S3 write fails, Igniter falls back to saving locally so training progress is never lost.

## Inference

For programmatic inference, use the `InferenceEngine` directly:

```python
from igniter.engine import InferenceEngine

# From a config file (uses build.<model>.weights), or
engine = InferenceEngine(config_file='configs/mnist.yaml', weights='/path/to/mnist_0000010.pt')

# From a training log directory (picks config.yaml + the latest checkpoint)
engine = InferenceEngine(log_dir='/tmp/dist/mnist/')

pred = engine(image)   # applies the configured transforms and runs the model
```

Or from the command line with optional pre/post-processing hooks defined under `build.<model>.inference`:

```bash
igniter test configs/mnist.yaml path/to/image.png --weights /path/to/weights.pt
```

## Distributed Training

Distributed training is handled by `ignite.distributed`. Enable it in the config:

```yaml
distributed:
  backend: nccl
  type: single          # single-node
  nproc_per_node: 4     # number of GPUs/processes
  single:
    init_method: tcp://127.0.0.1:23456
```

Igniter automatically wraps the dataloader and model with `idist.auto_*` helpers when more than one process is configured.

## Project Structure

```
igniter/
├── igniter/
│   ├── main.py              # initiate(), config loading, flow orchestration
│   ├── builder.py           # builds dataloaders, models, optimizers, engines from config
│   ├── registry.py          # the Registry class and all global registries
│   ├── cli.py               # `igniter` command line interface
│   ├── events.py            # built-in event handlers (checkpointing, …)
│   ├── defaults/            # default train/val steps, inference runner
│   ├── engine/              # TrainerEngine, EvaluationEngine, InferenceEngine
│   ├── datasets/            # S3 dataset, COCO dataset helpers
│   ├── io/                  # loggers (tqdm/fair), TensorBoard summaries, S3 client/IO
│   └── configs/config.yaml  # built-in default configuration
├── example/
│   ├── mnist.py             # ← start here: end-to-end MNIST example
│   ├── distill.py           # knowledge distillation example
│   ├── sam_image_features.py# SAM feature extraction example
│   └── configs/             # YAML configs for the examples
└── tests/                   # pytest test suite
```

## Development

```bash
pip install -e .
pre-commit install          # flake8 / isort hooks
pytest tests/               # run the test suite
```

## License

MIT © [Krishneel](https://github.com/iKrishneel)
