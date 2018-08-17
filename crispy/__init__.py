#!/usr/bin/env python
# Copyright (C) 2017 Emanuel Goncalves

import numpy as np
import pandas as pd
import pkg_resources
import seaborn as sns
from crispy.utils import Utils
from crispy.crispy import Crispy
from crispy.benchmark_plot import plot_cumsum_auc
from crispy.qc_plot import QCplot, SNS_RC, FLIERPROPS, PAL_DBGD, MEDIANPROPS, BOXPROPS, WHISKERPROPS


__version__ = '0.2.0'

# - SET STYLE
sns.set(style='ticks', context='paper', rc=SNS_RC)


# - PACKAGE DATA METHODS
DPATH = pkg_resources.resource_filename('crispy', 'data/')


def get_example_data(dfile='association_example_data.csv'):
    return pd.read_csv('{}/{}'.format(DPATH, dfile), index_col=0)


def get_essential_genes(dfile='gene_sets/curated_BAGEL_essential.csv', return_series=True):
    geneset = set(pd.read_csv('{}/{}'.format(DPATH, dfile), sep='\t')['gene'])

    if return_series:
        geneset = pd.Series(list(geneset)).rename('essential')

    return geneset


def get_non_essential_genes(dfile='gene_sets/curated_BAGEL_nonEssential.csv', return_series=True):
    geneset = set(pd.read_csv('{}/{}'.format(DPATH, dfile), sep='\t')['gene'])

    if return_series:
        geneset = pd.Series(list(geneset)).rename('non-essential')

    return geneset


def get_crispr_lib(dfile='crispr_libs/KY_Library_v1.1_annotated.csv'):
    r_cols = dict(
        index='sgrna', CHRM='chr', STARTpos='start', ENDpos='end', GENES='gene', EXONE='exon', CODE='code', STRAND='strand'
    )

    lib = pd.read_csv('{}/{}'.format(DPATH, dfile), index_col=0).reset_index()

    lib = lib.rename(columns=r_cols)

    lib['chr'] = lib['chr'].apply(lambda v: f'chr{v}' if str(v) != 'nan' else np.nan)

    return lib


def get_adam_core_essential(dfile='gene_sets/pancan_core.csv'):
    return set(pd.read_csv('{}/{}'.format(DPATH, dfile))['ADAM PanCancer Core-Fitness genes'].rename('adam_essential'))


def get_cytobands(dfile='cytoBand.txt', chrm=None):
    cytobands = pd.read_csv('{}/{}'.format(DPATH, dfile), sep='\t')

    if chrm is not None:
        cytobands = cytobands[cytobands['chr'] == chrm]

    assert cytobands.shape[0] > 0, '{} not found in cytobands file'

    return cytobands


CHR_SIZES_HG19 = {
    'chr1': 249250621, 'chr2': 243199373, 'chr3': 198022430, 'chr4': 191154276, 'chr5': 180915260, 'chr6': 171115067, 'chr7': 159138663, 'chr8': 146364022,
    'chr9': 141213431, 'chr10': 135534747, 'chr11': 135006516, 'chr12': 133851895, 'chr13': 115169878, 'chr14': 107349540, 'chr15': 102531392, 'chr16': 90354753,
    'chr17': 81195210, 'chr18': 78077248, 'chr19': 59128983, 'chr20': 63025520, 'chr21': 48129895, 'chr22': 51304566, 'chrX': 155270560, 'chrY': 59373566
}


# - HANDLES
__all__ = [
    'Crispy',

    'Utils',

    'QCplot',
    'PAL_DBGD',
    'FLIERPROPS',
    'SNS_RC',
    'MEDIANPROPS',
    'BOXPROPS',
    'WHISKERPROPS',

    'get_example_data',
    'get_essential_genes',
    'get_non_essential_genes',
    'get_crispr_lib',
    'get_adam_core_essential',
    'plot_cumsum_auc',

    'CHR_SIZES_HG19'
]
