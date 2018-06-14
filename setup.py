#!/usr/bin/env python
# Copyright (C) 2018 Emanuel Goncalves

import setuptools

with open('README.md', 'r') as fh:
    long_description = fh.read()

setuptools.setup(
    name='cy',
    version='0.1.4',
    author='Emanuel Goncalves',
    author_email='eg14@sanger.ac.uk',
    description='Modelling CRISPR dropout data',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/EmanuelGoncalves/crispy',
    packages=setuptools.find_packages(),
    classifiers=(
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: BSD License',
        'Operating System :: OS Independent',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Bio-Informatics',
        'Topic :: Scientific/Engineering :: Information Analysis'
    ),
)