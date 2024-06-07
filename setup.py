#! /usr/bin/env python

from setuptools import find_packages, setup

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
    'hydra-core >= 1.2',
    'tabulate',
    'omegaconf',
    'colorlog',
    'boto3',
    'pytest',
    'pytest-mock',
    'pytorch-ignite>=0.4.12',
    'tensorboard',
    'argcomplete',
    'safetensors',
    'requests',
]


__name__ = 'igniter'

with open(f'{__name__}/__init__.py', 'r') as init_file:
    for line in init_file:
        if line.startswith("__version__"):
            exec(line)


setup(
    name=__name__,
    author='Krishneel',
    email='krishneel@krishneel',
    license='MIT',
    url='https://github.com/iKrishneel/igniter',
    version=f'{__version__}',  # NOQA: F821
    long_description=readme,
    packages=find_packages(),
    zip_safe=True,
    install_requires=install_requires,
    test_suite='tests',
    include_package_data=True,
    package_data={__name__: [f'{__name__}/configs/config.yaml']},
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Topic :: Software Development :: Build Tools',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
    ],
    entry_points={
        'console_scripts': {
            'igniter=igniter.cli:main',
        },
    },
)
