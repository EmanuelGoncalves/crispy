#!/usr/bin/env python
# Copyright (C) 2017 Emanuel Goncalves

import pandas as pd
import pkg_resources
import seaborn as sns
from crispy.association import CRISPRCorrection
from crispy.benchmark_plot import plot_cumsum_auc


__version__ = '0.1.7'

# PLOT AESTHETICS
PAL_DBGD = {0: '#656565', 1: '#F2C500', 2: '#E1E1E1'}

SNS_RC = {
    'axes.linewidth': .3,
    'xtick.major.width': .3, 'ytick.major.width': .3,
    'xtick.major.size': 2.5, 'ytick.major.size': 2.5,
    'xtick.direction': 'in', 'ytick.direction': 'in'
}

sns.set(style='ticks', context='paper', rc=SNS_RC)


# - PACKAGE DATA METHODS
DPATH = pkg_resources.resource_filename('crispy', 'data/')


def get_example_data(dfile='association_example_data.csv'):
    return pd.read_csv('{}/{}'.format(DPATH, dfile), index_col=0)


def get_essential_genes(dfile='gene_sets/curated_BAGEL_essential.csv'):
    return set(pd.read_csv('{}/{}'.format(DPATH, dfile), sep='\t')['gene'].rename('essential'))


def get_non_essential_genes(dfile='gene_sets/curated_BAGEL_nonEssential.csv'):
    return set(pd.read_csv('{}/{}'.format(DPATH, dfile), sep='\t')['gene'].rename('non-essential'))


def get_crispr_lib(dfile='crispr_libs/KY_Library_v1.1_updated.csv'):
    return pd.read_csv('{}/{}'.format(DPATH, dfile), index_col=0)


# -
def get_cytobands(dfile='cytoBand.txt', chrm=None):
    cytobands = pd.read_csv('{}/{}'.format(DPATH, dfile), sep='\t')

    if chrm is not None:
        cytobands = cytobands[cytobands['chr'] == chrm]

    assert cytobands.shape[0] > 0, '{} not found in cytobands file'

    return cytobands


# - HANDLES
__all__ = [
    'CRISPRCorrection',
    'get_example_data',
    'PAL_DBGD',
    'get_essential_genes',
    'get_non_essential_genes',
    'get_crispr_lib',
    'plot_cumsum_auc'
]
