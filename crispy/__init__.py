#!/usr/bin/env python
# Copyright (C) 2017 Emanuel Goncalves

import pandas as pd
import pkg_resources
import seaborn as sns
from crispy.association import CRISPRCorrection

name = 'crispy'

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


def get_example_data(dfile='association_example_data.csv'):
    DATA_PATH = pkg_resources.resource_filename('crispy', 'extdata/')
    return pd.read_csv('{}/{}'.format(DATA_PATH, dfile), index_col=0)

