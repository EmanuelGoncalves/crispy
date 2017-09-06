#!/usr/bin/env python
# Copyright (C) 2017 Emanuel Goncalves

import numpy as np
import pandas as pd
import seaborn as sns
import scipy.stats as st
import matplotlib.pyplot as plt
from datetime import datetime as dt
from sklearn.linear_model.base import LinearRegression

# - Imports
# Essential genes
essential = list(pd.read_csv('data/resources/curated_BAGEL_essential.csv', sep='\t')['gene'])

# sgRNA library
sgrna_lib = pd.read_csv('data/gdsc/crispr/KY_Library_v1.1_annotated.csv', index_col=0).dropna(subset=['STARTpos'])

# Manifest
manifest = pd.read_csv('data/gdsc/crispr/manifest.txt', sep='\t')
manifest['CGaP_Sample_Name'] = [i.split('_')[0] for i in manifest['CGaP_Sample_Name']]
manifest = manifest.groupby('CGaP_Sample_Name')['Cell_Line_Name'].first()

# Cell lines samplesheet
css = pd.read_csv('data/gdsc/samplesheet.csv', index_col=0)

# Samplesheet
ss = pd.read_csv('data/gdsc/crispr/crispr_raw_counts_files_vFB.csv', sep=',', index_col=0).query("to_use == 'Yes'")
ss = ss.assign(name=list(manifest.loc[[i.split('_')[0] for i in ss.index]]))
ss = ss.assign(tissue=list(css.loc[ss['name'], 'GDSC1'].replace(np.nan, 'other')))

# sgRNA library
sgrna_lib = pd.read_csv('data/gdsc/crispr/KY_Library_v1.1_annotated.csv', index_col=0).dropna(subset=['STARTpos'])

# Controls
controls = set(ss['control'])


# - Controls read counts histogram
df_controls = pd.concat([
    pd.read_csv('data/gdsc/crispr/raw_counts/%s' % f, sep='\t', index_col=0).drop(['gene'], axis=1)[s] for s, f in ss.reset_index().groupby('control')['file'].first().items()
], axis=1).dropna()

sns.set(style='ticks', context='paper', rc={'axes.linewidth': .3, 'xtick.major.width': .3, 'ytick.major.width': .3, 'xtick.major.size': 2.5, 'ytick.major.size': 2.5, 'xtick.direction': 'in', 'ytick.direction': 'in'})
(f, axs), pos = plt.subplots(1, 2), 0

for s, c in zip(*(controls, ['#37454B', '#F2C500'])):
    sns.distplot(df_controls[s].dropna(), kde=False, bins=30, color=c, label=s, ax=axs[pos])
    axs[pos].set_xlabel('Read counts')
    axs[pos].set_ylabel('Counts')
    axs[pos].set_title('%s (%d samples)' % (s, ss['control'].value_counts()[s]))
    pos += 1

plt.gcf().set_size_inches(8, 2)
plt.savefig('reports/controls_counts_histogram.png', bbox_inches='tight', dpi=600)
plt.close('all')


# -
fold_changes = {}
# Assemble counts matrix
for c_sample in manifest.index:
    # c_sample = 'HT29'
    print('[%s] Preprocess Sample: %s' % (dt.now().strftime('%Y-%m-%d %H:%M:%S'), c_sample))

    # Sample files
    c_files = set(ss[[i.split('_')[0] == c_sample for i in ss.index]].index)

    # Read counts
    c_counts = {}
    for f in c_files:
        df = pd.read_csv('data/gdsc/crispr/raw_counts/%s' % f, sep='\t', index_col=0).drop(['gene'], axis=1).to_dict()
        c_counts.update(df)

    c_counts = pd.DataFrame(c_counts).drop('sgPOLR2K_1', errors='ignore')

    # Add dummy count for 0 counts
    c_counts += 1

    # Normalisation coefficients
    c_counts_gmean = pd.Series(st.gmean(c_counts, axis=1), index=c_counts.index)
    c_counts_gmean = c_counts.T.divide(c_counts_gmean).T
    c_counts_gmean = c_counts_gmean.median()

    c_counts = c_counts.divide(c_counts_gmean)

    # Log2 transform
    c_counts = np.log2(c_counts)

    # Calculate fold-changes
    c_control = [i for i in c_counts if i in controls][0]
    c_counts = c_counts.subtract(c_counts[c_control], axis=0).drop(c_control, axis=1)

    # Zscore
    c_counts = pd.DataFrame(st.zscore(c_counts), index=c_counts.index, columns=c_counts.columns)

    # Average fold-changes
    c_counts = c_counts.mean(1)

    # Store
    fold_changes[manifest[c_sample]] = c_counts

fold_changes = pd.DataFrame(fold_changes)
print(fold_changes.shape)

# - Export
fold_changes.to_csv('data/gdsc/crispr/crispy_fold_change.csv')
