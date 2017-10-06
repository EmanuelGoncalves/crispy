#!/usr/bin/env python
# Copyright (C) 2017 Emanuel Goncalves

from distutils.core import setup
from Cython.Build import cythonize

# python gsea_setup.py build_ext --inplace

setup(
  name='SS GSEA',
  ext_modules=cythonize('gsea.pyx'),
)
