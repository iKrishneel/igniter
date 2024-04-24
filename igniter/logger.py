#!/usr/bin/env python

import logging

import colorlog

logger = logging.getLogger()

for handler in logger.handlers.copy():
    if not isinstance(handler.formatter, colorlog.ColoredFormatter):
        logger.removeHandler(handler)


level = logging.INFO

formatter = colorlog.ColoredFormatter(
    '%(log_color)s%(asctime)s [%(filename)s:%(lineno)d] %(levelname)s: %(message)s%(reset)s',
    log_colors={'DEBUG': 'cyan', 'INFO': 'green', 'WARNING': 'yellow', 'ERROR': 'red', 'CRITICAL': 'bold_red'},
)

console_handler = logging.StreamHandler()
console_handler.setLevel(level)
console_handler.setFormatter(formatter)

# file_handler = logging.FileHandler('/tmp/logs.log')
# file_handler.setLevel(level)
# file_handler.setFormatter(logging.Formatter(format_style))

logger.addHandler(console_handler)
logger.setLevel(level)
# logger.addHandler(file_handler)
