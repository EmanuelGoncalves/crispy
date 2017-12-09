#!/usr/bin/env python
# Copyright (C) 2017 Emanuel Goncalves

import crispy
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from scipy.stats import spearmanr
from sklearn.mixture import GaussianMixture

# Utils
cpal = {1: '#FF8200', 0: '#58595B'}
controls = {'CRISPR_C6596666.sample', 'ERS717283.plasmid'}

# Impor
# # sgRNA library
sgrna_lib = pd.read_csv('data/gdsc/crispr/KY_Library_v1.1_annotated.csv', index_col=0).dropna(subset=['STARTpos', 'GENES'])

# Essential genes
essential = pd.read_csv('data/resources/curated_BAGEL_essential.csv', sep='\t')['gene'].rename('essential')

# data = pd.read_csv('data/gdsc/crispr/raw_counts/HT29_C902R3_F2C50L30.read_count.tsv', sep='\t', index_col=0)\
data = pd.read_csv('data/gdsc/crispr/raw_counts/MB361_c904R1.read_count.tsv', sep='\t', index_col=0)\
    .drop(['sgPOLR2K_1'], errors='ignore').drop('gene', axis=1)
data = np.log2(data + 1)

# GMM
df = pd.concat([data.iloc[:, [0]].subtract(data.iloc[:, 1], axis=0)], axis=1)

gmm = GaussianMixture(n_components=2, covariance_type='full').fit(df)
gmm_label = pd.Series(dict(zip(*(df.index, gmm.predict(df)))))

svc = SVC().fit(data)
gmm_label = pd.Series(dict(zip(*(data.index, gmm.predict(data)))))

#
# x, y = 'ERS717283.plasmid', '3980STDY6310326.sample'
x, y = 'ERS717283.plasmid', 'CRISPR_C7020593.sample'

g = sns.jointplot(x, y, data=data, space=0, color=cpal[0], joint_kws={'s': 2, 'lw': .1, 'edgecolor': 'white', 'alpha': .4})
# sns.regplot(
#     x, y, data=data.loc[gmm_label == 1], color=cpal[0], truncate=True, fit_reg=False,
#     scatter_kws={'s': 2, 'edgecolor': cpal[1], 'linewidth': .5, 'alpha': .8}, ax=g.ax_joint
# )
g.annotate(spearmanr, template='{stat}: {val:.2f}', loc='upper left', stat="Spearman's rho", fontsize=5)
g.set_axis_labels('Plasmid (log2)', 'MDA-MB-361 (log2)')
xlim, ylim = g.ax_joint.get_xlim(), g.ax_joint.get_ylim()
g.ax_joint.plot(xlim, ylim, 'k--', lw=.3)
g.ax_joint.set_xlim(xlim)
g.ax_joint.set_ylim(ylim)
plt.gcf().set_size_inches(3, 3)
plt.savefig('reports/raw_counts_metric.png', bbox_inches='tight', dpi=600)
plt.close('all')

#
plot_df = data['CRISPR_C7020593.sample'] - data['ERS717283.plasmid']
plot_df = plot_df.groupby(sgrna_lib.loc[plot_df.index, 'GENES']).mean().dropna()
plot_df = pd.concat([
    plot_df.rename('fc'),
    pd.Series({g: 'Essential' if g in essential.values else 'All' for g in plot_df.index}).rename('Type')
], axis=1)
plot_df = plot_df.assign(dummy=1)

sns.distplot(plot_df.query("Type == 'All'")['fc'], color=cpal[0], label='All')
sns.distplot(plot_df.query("Type == 'Essential'")['fc'], color=cpal[1], label='Essential')
plt.title('MDA-MB-361')
plt.xlabel('Fold-change (log2)')
plt.ylabel('Density')
plt.legend(loc='upper left')
plt.gcf().set_size_inches(3, 2)
plt.savefig('reports/raw_counts_metric_hist.png', bbox_inches='tight', dpi=600)
plt.close('all')


sns.boxplot('Type', 'fc', data=plot_df, palette={'All': cpal[0], 'Essential': cpal[1]}, linewidth=.3, notch=True, fliersize=2)
plt.title('MDA-MB-361')
plt.ylabel('Fold-change (log2)')
plt.xlabel('')
plt.gcf().set_size_inches(1, 3)
plt.savefig('reports/raw_counts_metric_boxplots.png', bbox_inches='tight', dpi=600)
plt.close('all')
