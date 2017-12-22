#!/usr/bin/env python
# Copyright (C) 2017 Emanuel Goncalves

import crispy
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from crispy.data_matrix import DataMatrix
from crispy.benchmark_plot import plot_cumsum_auc


# - Imports
# Color palette
color = '#34495e'

# Essential genes
essential = pd.read_csv('data/resources/curated_BAGEL_essential.csv', sep='\t')['gene'].rename('essential')

# Non-essential genes
nessential = pd.read_csv('data/resources/curated_BAGEL_nonEssential.csv', sep='\t')['gene'].rename('non-essential')

# sgRNA library
sgrna_lib = pd.read_csv('data/gdsc/crispr/KY_Library_v1.1_annotated.csv', index_col=0)

# Crispr
qc_failed_sgrnas = [
    'DHRSX_CCDS35195.1_ex1_X:2161152-2161175:+_3-1',
    'DHRSX_CCDS35195.1_ex6_Y:2368915-2368938:+_3-3',
    'DHRSX_CCDS35195.1_ex4_X:2326777-2326800:+_3-2',
    'sgPOLR2K_1'
]

crispr = DataMatrix(pd.read_csv('data/HaCaT_Essential.read_count.tsv', sep='\t', index_col=0))
crispr = crispr.drop(qc_failed_sgrnas).drop('gene', axis=1)

# - Preprocess
# Filter sgRNAs with low counts in the plasmid
crispr = crispr[crispr['Plasmid_v1.1'] >= 30]

# Add constant
crispr = crispr + 1

# Normalise by library size
size_factors = crispr.estimate_size_factors()
crispr = crispr.divide(size_factors)


# - Library counts
# EGAN00001611743.sample    216004621
# Plasmid_v1.1              57856157

# CRISPR_C7020593.sample    42104795
# ERS717283.plasmid         38445785

# Raw counts
g = sns.jointplot('Plasmid_v1.1', 'EGAN00001611743.sample', data=np.log2(crispr), space=0, color=color, joint_kws={'s': 2, 'lw': .1, 'edgecolor': 'white', 'alpha': .4})
g.annotate(spearmanr, template='{stat}: {val:.2f}', loc='upper left', stat="Spearman's rho", fontsize=5)
g.set_axis_labels('Plasmid_v1.1 (log2)', 'EGAN00001611743.sample (log2)')
plt.gcf().set_size_inches(3, 3)
plt.savefig('reports/HaCaTs_raw_counts.png', bbox_inches='tight', dpi=600)
plt.close('all')


# - Fold-change
# Calculate log2 fold-changes: log2(sample / plasmid)
crispr_fc = np.log2(crispr['EGAN00001611743.sample'] / crispr['Plasmid_v1.1'])

# Average gene level
crispr_fc = crispr_fc.loc[sgrna_lib.index].groupby(sgrna_lib['GENES'].values).mean().dropna().sort_values()

# Export
crispr_fc.to_csv('data/HaCaT_Essential.fold_change.csv')


# - AUCs
# Essential AUCs corrected fold-changes
ax, stats_ess = plot_cumsum_auc(crispr_fc.to_frame(), essential, plot_mean=True, legend=False, palette=[color])
plt.gcf().set_size_inches(3, 3)
plt.savefig('reports/HaCaTs_essential_auc.png', bbox_inches='tight', dpi=600)
plt.close('all')

# Non-essential AUCs corrected fold-changes
ax, stats_ess = plot_cumsum_auc(crispr_fc.to_frame(), nessential, plot_mean=True, legend=False, palette=[color])
plt.gcf().set_size_inches(3, 3)
plt.savefig('reports/HaCaTs_non_essential_auc.png', bbox_inches='tight', dpi=600)
plt.close('all')
