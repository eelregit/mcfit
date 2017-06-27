#!/usr/bin/env python
from setuptools import setup

setup(
    name = 'mcfit',
    version = '0.0.4',
    description = 'Multiplicatively convolutional fast integral transforms',
    url = 'https://github.com/eelregit/mcfit',
    author = 'Yin Li',
    author_email = 'eelregit@gmail.com',
    license = 'GPLv3',
    keywords = 'numerical integral transform FFTLog cosmology',
    packages = ['mcfit', 'mcfit.tests'],
    install_requires = ['numpy', 'scipy', 'mpmath'],
)
