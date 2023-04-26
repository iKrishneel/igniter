#!/usr/bin/env python

import logging
import colorlog

logger = logging.getLogger()

for handler in logger.handlers.copy():
    if not isinstance(handler.formatter, colorlog.ColoredFormatter):
        logger.removeHandler(handler)


formatter = colorlog.ColoredFormatter(
    '%(log_color)s%(asctime)s [%(filename)s:%(lineno)d] %(levelname)s: %(message)s%(reset)s',
    log_colors={'DEBUG': 'cyan', 'INFO': 'green', 'WARNING': 'yellow', 'ERROR': 'red', 'CRITICAL': 'bold_red'},
)

console = logging.StreamHandler()
console.setFormatter(formatter)
logger.addHandler(console)
