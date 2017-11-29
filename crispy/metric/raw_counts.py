#!/usr/bin/env python
# Copyright (C) 2017 Emanuel Goncalves

import crispy
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture

# Utils
cpal = {1: '#FF8200', 0: '#58595B'}
controls = {'CRISPR_C6596666.sample', 'ERS717283.plasmid'}

# Import
# data = pd.read_csv('data/gdsc/crispr/raw_counts/HT29_C902R3_F2C50L30.read_count.tsv', sep='\t', index_col=0)\
data = pd.read_csv('data/gdsc/crispr/raw_counts/AU565_c903R1.read_count.tsv', sep='\t', index_col=0)\
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
x, y = 'ERS717283.plasmid', 'CRISPR_C6394964.sample'

g = sns.jointplot(x, y, data=data, space=0, color=cpal[0], joint_kws={'s': 2, 'lw': .1, 'edgecolor': 'white', 'alpha': .4})
sns.regplot(
    x, y, data=data.loc[gmm_label == 1], color=cpal[0], truncate=True, fit_reg=False,
    scatter_kws={'s': 2, 'edgecolor': cpal[1], 'linewidth': .5, 'alpha': .8}, ax=g.ax_joint
)
xlim, ylim = g.ax_joint.get_xlim(), g.ax_joint.get_ylim()
g.ax_joint.plot(xlim, ylim, 'k--', lw=.3)
g.ax_joint.set_xlim(xlim)
g.ax_joint.set_ylim(ylim)
plt.gcf().set_size_inches(4, 4)
plt.savefig('reports/raw_counts_metric.png', bbox_inches='tight', dpi=600)
plt.close('all')
