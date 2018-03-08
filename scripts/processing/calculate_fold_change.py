#!/usr/bin/env python
# Copyright (C) 2018 Emanuel Goncalves

import numpy as np
import pandas as pd
import scripts as mp
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
    counts = counts.drop(mp.QC_FAILED_SGRNAS, errors='ignore')

    # Add constant
    counts = counts + 1

    # Normalise by library size
    size_factors = counts.estimate_size_factors()
    counts = counts.divide(size_factors)

    # Remove plasmid low count sgRNAs
    for c in set(manifest['plasmid']):
        counts.loc[counts[c] < mp.LOW_COUNT_THRES, c] = np.nan

    # Calculate log2 fold-changes: log2(sample / plasmid)
    fc = pd.DataFrame({c: np.log2(counts[c] / counts[manifest.loc[c, 'plasmid']]) for c in manifest.index})

    # Average replicates
    fc = fc.groupby(manifest.loc[fc.columns, 'name'], axis=1).mean()

    # Export - sgRNA level
    fc.round(5).to_csv(mp.CRISPR_SGRNA_FC)

    # Average gene level
    fc = fc.groupby(sgrna_lib.reindex(fc.index)['gene']).mean().dropna()

    # Export - gene level
    fc.round(5).to_csv(mp.CRISPR_GENE_FC)


if __name__ == '__main__':
    # Fold-change GDSC CRISPR
    foldchange_gdsc(
        counts_file=mp.CRISPR_RAW_COUNTS, manifest_file=mp.MANIFEST, lib_file=mp.LIBRARY
    )
