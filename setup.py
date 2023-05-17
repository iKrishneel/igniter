#! /usr/bin/env python

from setuptools import setup
from setuptools import find_packages


try:
    with open('README.md', 'r') as f:
        readme = f.read()
except Exception:
    readme = str('')


install_requires = [
    'einops',
    'numpy >= 1.2',
    'matplotlib',
    'opencv-python',
    'tqdm',
    'pytest',
    'hydra-core',
    'omegaconf',
    'colorlog',
    'boto3',
    'pytorch-ignite @ git+https://github.com/pytorch/ignite@master',
]


setup(
    name='igniter',
    version='0.0.1',
    long_description=readme,
    packages=find_packages(),
    zip_safe=False,
    install_requires=install_requires,
    test_suite='tests',
)
