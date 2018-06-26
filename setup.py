#!/usr/bin/env python
from setuptools import setup

def find_version(path):
    with open(path, 'r') as fp:
        file = fp.read()
    import re
    match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]",
                            file, re.M)
    if match:
        return match.group(1)
    raise RuntimeError("Version not found")

setup(
    name = 'mcfit',
    version = find_version("mcfit/__init__.py"),
    description = 'Multiplicatively convolutional fast integral transforms',
    url = 'https://github.com/eelregit/mcfit',
    author = 'Yin Li',
    author_email = 'eelregit@gmail.com',
    license = 'GPLv3',
    keywords = 'numerical integral transform FFTLog cosmology',
    packages = ['mcfit', 'mcfit.tests'],
    install_requires = ['numpy', 'scipy', 'mpmath'],
)
