#!/usr/bin/env python
# Copyright (C) 2017 Emanuel Goncalves

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import crispy
from crispy import bipal_dbgd
from crispy.data_matrix import DataMatrix


# - Imports
# Samplesheet
ss = pd.read_csv('data/gdsc/samplesheet.csv', index_col=0)

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

# Copy-number
copynumber = pd.read_csv('data/gdsc/copynumber/Gene_level_CN.txt', sep='\t', index_col=0).dropna()


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
fc = fc.groupby(sgrna_lib.reindex(fc.index)['GENES']).mean().dropna()

# Overlap with copy-number data
fc = fc.loc[:, fc.columns.isin(set(copynumber))]

# Export
fc.round(5).to_csv('data/crispr_gdsc_logfc.csv')


# - Tissue distribution
plot_df = ss.loc[list(fc), 'Cancer Type'].value_counts(ascending=True).reset_index().rename(columns={'index': 'Cancer Type', 'Cancer Type': 'Counts'})
plot_df = plot_df.assign(ypos=np.arange(plot_df.shape[0]))

plt.barh(plot_df['ypos'], plot_df['Counts'], .8, color=bipal_dbgd[0], align='center')

for x, y in plot_df[['ypos', 'Counts']].values:
    plt.text(y - .5, x, str(y), color='white', ha='right' if y != 1 else 'center', va='center', fontsize=6)

plt.yticks(plot_df['ypos'])
plt.yticks(plot_df['ypos'], plot_df['Cancer Type'])

plt.xlabel('# cell lines')
plt.title('CRISPR/Cas9 screen\n(%d cell lines)' % plot_df['Counts'].sum())

# plt.show()
plt.gcf().set_size_inches(2, 3)
plt.savefig('reports/screen_samples.png', bbox_inches='tight', dpi=600)
plt.close('all')
