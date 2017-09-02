#!/usr/bin/env python
# Copyright (C) 2017 Emanuel Goncalves

# -*- coding: utf-8 -*-
import os
from distutils.core import setup
from setuptools import find_packages


_MAJOR, _MINOR, _MICRO = 0, 1, 0
__version__ = '%d.%d.%d' % (_MAJOR, _MINOR, _MICRO)


metainfo = {
    'authors': {
        'Goncalves': ('Emanuel Goncalves', 'eg14@sanger.ac.uk'),
    },

    'version': __version__,

    'license': 'BSD',

    'url': ['https://github.com/EmanuelGoncalves/crispy'],
    'download_url': [],

    'description': 'CRISPR-Cas9 drop-out screen Python toolbox',

    'platforms': ['Linux', 'Unix', 'MacOsX', 'Windows'],

    'keywords': ['CRISPR', 'loss-of-function', 'gaussian'],

    'classifiers': [
          'Development Status :: 1 - Planning',
          'Intended Audience :: Developers',
          'Intended Audience :: Science/Research',
          'License :: OSI Approved :: BSD License',
          'Operating System :: OS Independent',
          'Programming Language :: Python :: 2.7',
          'Programming Language :: Python :: 3.5',
          'Topic :: Software Development :: Libraries :: Python Modules',
          'Topic :: Scientific/Engineering :: Bio-Informatics',
          'Topic :: Scientific/Engineering :: Information Analysis',
          'Topic :: Scientific/Engineering :: Mathematics',
          'Topic :: Scientific/Engineering :: Physics'
    ]
}


with open('README.md') as f:
    readme = f.read()


setup(
    name='crispy', version=__version__,

    maintainer=metainfo['authors']['Goncalves'][0], maintainer_email=metainfo['authors']['Goncalves'][1],
    author=metainfo['authors']['Goncalves'][0], author_email=metainfo['authors']['Goncalves'][1],

    description=metainfo['description'], long_description=readme, keywords=metainfo['keywords'],

    license=metainfo['license'],

    platforms=metainfo['platforms'],

    url=metainfo['url'], download_url=metainfo['download_url'],

    classifiers=metainfo['classifiers'],

    packages=find_packages(),

    package_data={
        '': ['*.py', '*.txt', '*.csv'],
    },

    install_requires=[
        'numpy', 'scipy', 'pandas', 'scikit-learn', 'limix', 'matplotlib', 'seaborn'
    ],

    zip_safe=False,

    entry_points={
        'console_scripts': [
            'gdsctools_anova=gdsctools.scripts.anova:main',
            'gdsctools_regression=gdsctools.scripts.regression:main'
        ]
    },
)

