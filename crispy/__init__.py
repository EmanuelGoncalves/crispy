#!/usr/bin/env python
# Copyright (C) 2017 Emanuel Goncalves

import pandas as pd
import pkg_resources
import seaborn as sns
from crispy.association import CRISPRCorrection
from crispy.utils import get_example_data

name = 'crispy'

__version__ = '0.1.5'

# Set plotting aesthetics
sns_rc = {
    'axes.linewidth': .3,
    'xtick.major.width': .3, 'ytick.major.width': .3,
    'xtick.major.size': 2.5, 'ytick.major.size': 2.5,
    'xtick.direction': 'in', 'ytick.direction': 'in'
}
sns.set(style='ticks', context='paper', rc=sns_rc)

# Define color palettes
bipal_gray = {1: '#FF8200', 0: '#58595B'}
bipal_dbgd = {1: '#F2C500', 0: '#37454B'}

__all__ = [
    'name',
    'CRISPRCorrection',
    'get_example_data',
    'sns_rc',
    'bipal_gray',
    'bipal_dbgd'
]
