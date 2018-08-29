#!/usr/bin/env python
# Copyright (C) 2018 Emanuel Goncalves

import setuptools

with open('README.md', 'r') as fh:
    long_description = fh.read()

included_files = {
    'crispy': [
        'data/association_example_data.csv',
    ]
}

requirements = [
    'numpy>=1.13',
    'scipy>=0.19',
    'pandas>=0.20',
    'scikit-learn>=0.18',
    'matplotlib>=2.0',
    'seaborn>=0.7',
    'natsort>=5.1.0',
    'statsmodels>=0.8.0'
]

setuptools.setup(
    name='cy',
    version='0.2.1',

    author='Emanuel Goncalves',
    author_email='eg14@sanger.ac.uk',

    long_description=long_description,
    description='Modelling CRISPR dropout data',
    long_description_content_type='text/markdown',
    url='https://github.com/EmanuelGoncalves/crispy',

    packages=setuptools.find_packages(),

    include_package_data=True,
    package_data=included_files,
    install_requires=requirements,

    classifiers=(
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: BSD License',
        'Operating System :: OS Independent',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Bio-Informatics',
        'Topic :: Scientific/Engineering :: Information Analysis'
    ),
)