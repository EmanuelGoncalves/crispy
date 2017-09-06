#!/usr/bin/env python
# Copyright (C) 2017 Emanuel Goncalves

from setuptools import setup, find_packages


setup(
    name='crispy',
    version='0.0.1',
    package_dir={'': 'src'},
    packages=find_packages('src'),
    install_requires=['numpy', 'scipy', 'pandas', 'scikit-learn', 'limix', 'matplotlib', 'seaborn'],
    scripts=['scripts/framed-smetana-pipeline'],  # TODO: CHANGE
    author='Emanuel Goncalves',
    author_email='eg14@sanger.ac.uk',
    description='Crispy - modelling CRISPR dropout data',
    license='BSD',
    keywords=['CRISPR', 'loss-of-function', 'gaussian'],
    url='https://github.com/EmanuelGoncalves/crispy',
    long_description=open('README.rst').read(),
    classifiers=[
        'Development Status :: 1 - Planning',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: BSD License',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3.5',
        'Topic :: Software Development :: Libraries :: Python Modules',
        'Topic :: Scientific/Engineering :: Bio-Informatics',
        'Topic :: Scientific/Engineering :: Information Analysis'
    ],
)
