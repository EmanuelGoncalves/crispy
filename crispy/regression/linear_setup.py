#!/usr/bin/env python
# Copyright (C) 2018 Emanuel Goncalves

import numpy
from distutils.core import setup, Extension
from Cython.Build import cythonize


setup(
  name='LR',
  ext_modules=cythonize('linear.pyx'),
  include_dirs=[numpy.get_include()]
)
