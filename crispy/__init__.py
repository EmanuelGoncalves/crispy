#!/usr/bin/env python
# Copyright (C) 2017 Emanuel Goncalves

import pandas as pd
import pkg_resources
import seaborn as sns
from crispy.association import CRISPRCorrection


__version__ = '0.1.7'

PAL_DBGD = {1: '#F2C500', 0: '#656565'}

# PLOT AESTHETICS
sns_rc = {
    'axes.linewidth': .3,
    'xtick.major.width': .3, 'ytick.major.width': .3,
    'xtick.major.size': 2.5, 'ytick.major.size': 2.5,
    'xtick.direction': 'in', 'ytick.direction': 'in'
}
sns.set(style='ticks', context='paper', rc=sns_rc)


# GENE-SETS
HART_ESSENTIAL = 'data/gene_sets/curated_BAGEL_essential.csv'
HART_NON_ESSENTIAL = 'data/gene_sets/curated_BAGEL_nonEssential.csv'

# CRISPR LIBRARIES
CRISPR_LIBS = ['data/crispr_libs/KY_Library_v1.1_updated.csv']


# - METHODS
def get_example_data(dfile='association_example_data.csv'):
    dpath = pkg_resources.resource_filename('crispy', 'data/')
    return pd.read_csv('{}/{}'.format(dpath, dfile), index_col=0)


def get_essential_genes():
    return set(pd.read_csv(HART_ESSENTIAL, sep='\t')['gene'].rename('essential'))


def get_non_essential_genes():
    return set(pd.read_csv(HART_NON_ESSENTIAL, sep='\t')['gene'].rename('non-essential'))


def get_crispr_lib(flib):
    return pd.read_csv(CRISPR_LIBS[0] if flib is None else flib, index_col=0)


# - HANDLES
__all__ = [
    'CRISPRCorrection',
    'get_example_data',
    'PAL_DBGD',
    'get_essential_genes',
    'get_non_essential_genes',
    'get_crispr_lib'
]
