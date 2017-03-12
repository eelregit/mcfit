#!/usr/bin/env python
from distutils.core import setup

setup(
    name = 'mcfit',
    version = '0.0.1',
    author = 'Yin Li',
    author_email = 'eelregit@gmail.com',
    description = 'Multiplicative convolutional fast integral transforms',
    url = 'https://github.com/eelregit/mcfit',
    packages = ['mcfit', 'mcfit.tests'],
    zip_safe = True,
    license = 'GPLv3',
    install_requires = ['numpy', 'scipy'],
)
