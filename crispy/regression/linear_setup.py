#!/usr/bin/env python
# Copyright (C) 2018 Emanuel Goncalves

from distutils.core import setup
from Cython.Build import cythonize

# python linear_setup.py build_ext --inplace

setup(
  name='SS GSEA',
  ext_modules=cythonize('linear_setup.py'),
)
