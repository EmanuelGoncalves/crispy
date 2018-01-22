#!/usr/bin/env python
# Copyright (C) 2017 Emanuel Goncalves

import crispy
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from crispy import bipal_dbgd
from scipy.stats import spearmanr
from crispy.data_matrix import DataMatrix


# - Import
# CRISPR raw counts
raw_counts = DataMatrix(pd.read_csv('data/crm/hnsc_rawcounts.csv', index_col=0).drop(['gene'], axis=1))

# sgRNA library
sgrna_lib = pd.read_csv('data/crm/KY_Library_v1.0.csv', index_col=0)

# Essential genes
essential = pd.read_csv('data/resources/curated_BAGEL_essential.csv', sep='\t')['gene'].rename('essential')

# Samplesheet
samplesheet = pd.read_csv('data/crm/samplesheet.csv', index_col=0)


# -
# Remove QC failed
qc_failed_sgrnas = [
    'DHRSX_CCDS35195.1_ex1_X:2161152-2161175:+_3-1', 'DHRSX_CCDS35195.1_ex6_Y:2368915-2368938:+_3-3', 'DHRSX_CCDS35195.1_ex4_X:2326777-2326800:+_3-2', 'sgPOLR2K_1',
    'MOG_CCDS34366.1_ex0_6:29625062-29625085:-_13-1'
]
raw_counts = raw_counts.drop(qc_failed_sgrnas, errors='ignore')

# Add constant
counts = raw_counts + 1

# Normalise by library size
size_factors = raw_counts.estimate_size_factors()
raw_counts = raw_counts.divide(size_factors)


# -
y_sample = 'ORL48-D10-KOR2'
x_sample = samplesheet.loc[y_sample, 'plasmid_id']

g = sns.jointplot(x_sample, y_sample, data=raw_counts, space=0, color=bipal_dbgd[0], joint_kws={'s': 2, 'lw': .1, 'edgecolor': 'white', 'alpha': .4})

g.annotate(spearmanr, template='{stat}: {val:.2f}', loc='upper left', stat="Spearman's rho", fontsize=8)

g.set_axis_labels('Plasmid (log2)', 'MDA-MB-361 (log2)')
plt.gcf().set_size_inches(3, 3)
plt.savefig('reports/raw_counts_metric.png', bbox_inches='tight', dpi=600)
plt.close('all')

# #
# plot_df = data['CRISPR_C7020593.sample'] - data['ERS717283.plasmid']
# plot_df = plot_df.groupby(sgrna_lib.loc[plot_df.index, 'GENES']).mean().dropna()
# plot_df = pd.concat([
#     plot_df.rename('fc'),
#     pd.Series({g: 'Essential' if g in essential.values else 'All' for g in plot_df.index}).rename('Type')
# ], axis=1)
# plot_df = plot_df.assign(dummy=1)
#
# sns.distplot(plot_df.query("Type == 'All'")['fc'], color=cpal[0], label='All')
# sns.distplot(plot_df.query("Type == 'Essential'")['fc'], color=cpal[1], label='Essential')
# plt.title('MDA-MB-361')
# plt.xlabel('Fold-change (log2)')
# plt.ylabel('Density')
# plt.legend(loc='upper left')
# plt.gcf().set_size_inches(3, 2)
# plt.savefig('reports/raw_counts_metric_hist.png', bbox_inches='tight', dpi=600)
# plt.close('all')
#
#
# sns.boxplot('Type', 'fc', data=plot_df, palette={'All': cpal[0], 'Essential': cpal[1]}, linewidth=.3, notch=True, fliersize=2)
# plt.title('MDA-MB-361')
# plt.ylabel('Fold-change (log2)')
# plt.xlabel('')
# plt.gcf().set_size_inches(1, 3)
# plt.savefig('reports/raw_counts_metric_boxplots.png', bbox_inches='tight', dpi=600)
# plt.close('all')
