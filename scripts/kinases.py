#!/usr/bin/env python
# Copyright (C) 2017 Emanuel Goncalves

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from limix.qtl import qtl_test_lmm
from sklearn.linear_model import LinearRegression

# - Imports
ccrispy = pd.read_csv('data/crispr_gdsc_crispy.csv', index_col=0)

samplesheet = pd.read_csv('data/gdsc/samplesheet.csv', index_col=0).loc[ccrispy.columns]

kfam = pd.read_csv('data/resources/pkinfam_parsed.txt')
kfam = kfam[kfam['kinase'].isin(ccrispy.index)]

# -
y = ccrispy.loc[kfam['kinase']].T
x = pd.get_dummies(samplesheet.loc[y.index, 'Cancer Type'])

x = x.loc[:, x.sum() > 3]

lm = LinearRegression().fit(x, y)
lm_coef = pd.DataFrame(lm.coef_, index=y.columns, columns=x.columns)


# -
sns.clustermap(lm_coef.T, cmap='RdGy', center=0)
plt.gcf().set_size_inches(15, 4)
plt.savefig('reports/kinases_essentiality_clustermap.png', bbox_inches='tight', dpi=600)
plt.close('all')


# -
plot_df = lm_coef.T.corr()
sns.clustermap(plot_df, cmap='RdGy', center=0)
plt.gcf().set_size_inches(15, 15)
plt.savefig('reports/kinases_essentiality_clustermap_corr.png', bbox_inches='tight', dpi=600)
plt.close('all')
