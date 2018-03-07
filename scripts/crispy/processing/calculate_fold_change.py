#!/usr/bin/env python
# Copyright (C) 2018 Emanuel Goncalves

import numpy as np
import pandas as pd
from crispy.data_matrix import DataMatrix


def foldchange_gdsc(counts_file, manifest_file, lib_file):
    # - Imports
    # sgRNA library
    sgrna_lib = pd.read_csv(lib_file, index_col=0).dropna(subset=['gene'])

    # Manifest
    manifest = pd.read_csv(manifest_file, index_col='header').query('to_use')

    # Raw Counts
    counts = DataMatrix(pd.read_csv(counts_file, index_col=0))
    counts.columns = [i.split('.')[0] for i in counts]

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

    # Remove plasmid low count sgRNAs
    for c in set(manifest['plasmid']):
        counts.loc[counts[c] < 30, c] = np.nan

    # Calculate log2 fold-changes: log2(sample / plasmid)
    fc = pd.DataFrame({c: np.log2(counts[c] / counts[manifest.loc[c, 'plasmid']]) for c in manifest.index})

    # Average replicates
    fc = fc.groupby(manifest.loc[fc.columns, 'name'], axis=1).mean()

    # Export - sgRNA level
    fc.round(5).to_csv('data/crispr_gdsc_sgrna_logfc.csv')

    # Average gene level
    fc = fc.groupby(sgrna_lib.reindex(fc.index)['gene']).mean().dropna()

    # Export - gene level
    fc.round(5).to_csv('data/crispr_gdsc_logfc.csv')


if __name__ == '__main__':
    # Fold-change GDSC CRISPR
    foldchange_gdsc(
        counts_file='data/gdsc/crispr/gdsc_crispr_rawcounts.csv',
        manifest_file='data/gdsc/crispr/manifest.csv',
        lib_file='data/crispr_libs/KY_Library_v1.1_updated.csv'
    )
