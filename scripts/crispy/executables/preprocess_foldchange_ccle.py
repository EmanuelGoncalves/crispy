#!/usr/bin/env python
# Copyright (C) 2017 Emanuel Goncalves

import numpy as np
import pandas as pd
from crispy.data_matrix import DataMatrix


# - Imports
# Essential genes
essential = pd.read_csv('data/resources/curated_BAGEL_essential.csv', sep='\t')['gene'].rename('essential')

# Non-essential genes
nessential = pd.read_csv('data/resources/curated_BAGEL_nonEssential.csv', sep='\t')['gene'].rename('non-essential')

# sgRNA library
sgrna_lib = pd.read_csv('data/ccle/crispr_ceres/sgrnamapping.csv').dropna(subset=['Gene'])

# Manifest
manifest = pd.read_csv('data/ccle/crispr_ceres/replicatemap.csv', index_col=0)
manifest = manifest.assign(plasmid=['pDNA_%d' % i for i in manifest['Batch']]).dropna()

# Raw Counts
counts = DataMatrix(pd.read_csv('data/ccle/crispr_ceres/sgrna_raw_readcounts.csv', index_col=0))


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
fc.round(5).to_csv('data/crispr_ccle_logfc.csv')
