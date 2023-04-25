#!/usr/bin/env python

import hydra
from omegaconf import DictConfig

from igniter.builder import trainer


@hydra.main(version_base=None, config_path='./configs', config_name='config')
def main(cfg: DictConfig):
    trainer(cfg)


if __name__ == '__main__':
    main()
