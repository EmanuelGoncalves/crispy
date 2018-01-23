#!/usr/bin/env python
# Copyright (C) 2018 Emanuel Goncalves

import numpy as np
import pandas as pd
from crispy.data_matrix import DataMatrix


def foldchange_gdsc(
        counts_file='data/gdsc/crispr/gdsc_crispr_rawcounts.csv', manifest_file='data/gdsc/crispr/canapps_manifest.csv', lib_file='data/crispr_libs/KY_Library_v1.1_updated.csv'
):
    # - Imports
    # sgRNA library
    sgrna_lib = pd.read_csv(lib_file, index_col=0).dropna(subset=['gene'])

    # Manifest
    manifest = pd.read_csv(manifest_file, index_col='header').query('to_use == True')

    # Raw Counts
    counts = DataMatrix(pd.read_csv(counts_file, index_col=0))

    # - Processing
    # Build counts data-frame (discard sgRNAs with problems)
    qc_failed_sgrnas = [
        'DHRSX_CCDS35195.1_ex1_X:2161152-2161175:+_3-1', 'DHRSX_CCDS35195.1_ex6_Y:2368915-2368938:+_3-3', 'DHRSX_CCDS35195.1_ex4_X:2326777-2326800:+_3-2', 'sgPOLR2K_1'
    ]
    counts = counts.drop(qc_failed_sgrnas, errors='ignore')

    # Add constant
    counts = counts + 1

    # Normalise by library size
    size_factors = counts.estimate_size_factors()
    counts = counts.divide(size_factors)

    # Calculate replicates correlation (pearson)
    corr_coefs = []
    for c in set(manifest['name']):
        df = np.log2(counts[list(manifest.query("name == '%s'" % c).index)]).corr()
        corr_coefs.append(
            df.where(np.triu(np.ones(df.shape), k=1).astype(np.bool))
                .unstack()
                .dropna()
                .reset_index()
                .rename(columns={'level_0': 's1', 'level_1': 's2', 0: 'pearson'})
                .assign(cell_line=c)
        )
    corr_coefs = pd.concat(corr_coefs).sort_values('pearson')

    # Remove plasmid low count sgRNAs
    for c in set(manifest['plasmid']):
        counts.loc[counts[c] < 30, c] = np.nan

    # Calculate log2 fold-changes: log2(sample / plasmid)
    fc = pd.DataFrame({c: np.log2(counts[c] / counts[manifest.loc[c, 'plasmid']]) for c in manifest.index})

    # Remove lowly correlated replicates (< 0.7 pearson coef)
    fc = fc.loc[:, set(corr_coefs.query('pearson > .7')[['s1', 's2']].unstack())]

    # Average replicates
    fc = fc.groupby(manifest.loc[fc.columns, 'name'], axis=1).mean()

    # Average gene level
    fc = fc.groupby(sgrna_lib.reindex(fc.index)['gene']).mean().dropna()

    # Export
    fc.round(5).to_csv('data/crispr_gdsc_logfc.csv')


def foldchange_ccle(
        manifest_file='data/ccle/crispr_ceres/replicatemap.csv', counts_file='data/ccle/crispr_ceres/sgrna_raw_readcounts.csv', lib_file='data/crispr_libs/Achiles_Avana_Library.csv'
):
    # sgRNA library
    sgrna_lib = pd.read_csv(lib_file).dropna(subset=['Gene'])

    # Manifest
    manifest = pd.read_csv(manifest_file, index_col=0)
    manifest = manifest.assign(plasmid=['pDNA_%d' % i for i in manifest['Batch']]).dropna()

    # Raw Counts
    counts = DataMatrix(pd.read_csv(counts_file, index_col=0))

    # - Process
    # Build counts data-frame (discard sgRNAs with problems)
    qc_failed_sgrnas = list(counts[counts.sum(1) < 30].index)
    counts = counts.drop(qc_failed_sgrnas, errors='ignore')

    # Add constant
    counts = counts + 1

    # Normalise by library size
    size_factors = counts.estimate_size_factors()
    counts = counts.divide(size_factors)

    # Calculate replicates correlation (pearson)
    corr_coefs = []
    for c in set(manifest['CellLine']):
        df = np.log2(counts[list(manifest.query("CellLine == '%s'" % c).index)]).corr()
        corr_coefs.append(
            df.where(np.triu(np.ones(df.shape), k=1).astype(np.bool))
                .unstack()
                .dropna()
                .reset_index()
                .rename(columns={'level_0': 's1', 'level_1': 's2', 0: 'pearson'})
                .assign(cell_line=c)
        )
    corr_coefs = pd.concat(corr_coefs).sort_values('pearson')

    # Remove plasmid low count sgRNAs
    for c in set(manifest['plasmid']):
        counts.loc[counts[c] < 30, c] = np.nan

    # Calculate log2 fold-changes: log2(sample / plasmid)
    fc = pd.DataFrame({c: np.log2(counts[c] / counts[manifest.loc[c, 'plasmid']]) for c in manifest.index})

    # Remove lowly correlated replicates (< 0.7 pearson coef)
    fc = fc.loc[:, set(corr_coefs.query('pearson > .7')[['s1', 's2']].unstack())]

    # Average replicates
    fc = fc.groupby(manifest.loc[fc.columns, 'CellLine'], axis=1).mean()

    # Average gene level
    fc = fc.loc[sgrna_lib['Guide']].groupby(sgrna_lib['Gene'].values).mean().dropna()

    # Export
    fc.round(5).to_csv('data/crispr_achiles_logfc.csv')


if __name__ == '__main__':
    # Fold-change GDSC CRISPR
    foldchange_gdsc()

    # Fold-change CCLE CRISPR
    foldchange_ccle()
