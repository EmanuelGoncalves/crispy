#!/usr/bin/env python
# Copyright (C) 2017 Emanuel Goncalves

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from crispy.data_matrix import DataMatrix
from crispy.benchmark_plot import plot_cumsum_auc


# - Imports
# Essential genes
essential = pd.read_csv('data/resources/curated_BAGEL_essential.csv', sep='\t')['gene'].rename('essential')

# Non-essential genes
nessential = pd.read_csv('data/resources/curated_BAGEL_nonEssential.csv', sep='\t')['gene'].rename('non-essential')

# sgRNA library
sgrna_lib = pd.read_csv('data/gdsc/crispr/KY_Library_v1.1_annotated.csv', index_col=0).dropna(subset=['GENES'])

# Manifest
manifest = pd.read_csv('data/gdsc/crispr/canapps_manifest.csv', index_col='header').query('to_use == True')

# Raw Counts
counts = DataMatrix(pd.read_csv('data/gdsc/crispr/gdsc_crispr_rawcounts.csv', index_col=0))


# - Processing
# Build counts data-frame (discard sgRNAs with problems)
qc_failed_sgrnas = [
    'DHRSX_CCDS35195.1_ex1_X:2161152-2161175:+_3-1', 'DHRSX_CCDS35195.1_ex6_Y:2368915-2368938:+_3-3', 'DHRSX_CCDS35195.1_ex4_X:2326777-2326800:+_3-2', 'sgPOLR2K_1'
]
counts = DataMatrix(counts).drop(qc_failed_sgrnas, errors='ignore')

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
fc = fc.groupby(sgrna_lib.reindex(fc.index)['GENES']).mean().dropna()

# Export
fc.round(5).to_csv('data/crispr_gdsc_logfc.csv')


# - Benchmark
# Essential/Non-essential genes AUC
ax, ax_stats = plot_cumsum_auc(fc, essential, plot_mean=True, legend=False)
plt.gcf().set_size_inches(3, 3)
plt.savefig('reports/processing_gdsc_logfc_essential_auc.png', bbox_inches='tight', dpi=600)
plt.close('all')

ax, ax_stats = plot_cumsum_auc(fc, nessential, plot_mean=True, legend=False)
plt.gcf().set_size_inches(3, 3)
plt.savefig('reports/processing_gdsc_logfc_nonessential_auc.png', bbox_inches='tight', dpi=600)
plt.close('all')
