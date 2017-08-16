#!/usr/bin/env python
# Copyright (C) 2017 Emanuel Goncalves


import numpy as np
import pandas as pd
import scipy.stats as st
import seaborn as sns
import matplotlib.pyplot as plt

plasmids = ['ERS717283.plasmid', 'CRISPR_C6596666.sample']

# samplesheet
ss = pd.read_csv('data/gdsc/crispr/crispr_raw_counts_files_vFB.csv', sep=',', index_col=0)
ss = ss[ss['to_use'] == 'Yes']


# Assemble counts matrix
c_files = 'HT29'
c_files = set(ss[[i.split('_')[0] == c_files for i in ss.index]].index)

c_counts = {}
for f in c_files:
    df = pd.read_csv('data/gdsc/crispr/raw_counts/%s' % f, sep='\t', index_col=0).drop(['gene'], axis=1).to_dict()
    c_counts.update(df)

c_counts = pd.DataFrame(c_counts)

#
c_control = [i for i in c_counts if i in plasmids][0]

# Estimate samples median-of-ratios
c_counts_gmean = pd.Series(st.gmean(c_counts, axis=1), index=c_counts.index)
c_counts_gmean = c_counts.T.divide(c_counts_gmean).T
c_counts_gmean = c_counts_gmean.median()


#
plot_df = c_counts.copy().drop('sgPOLR2K_1')
plt.scatter(plot_df.mean(1), st.variation(plot_df, axis=1) ** 2, s=2)

#
plot_df = c_counts.copy().drop('sgPOLR2K_1')
plt.scatter(np.log10(plot_df[c_control]), np.log10(plot_df['CRISPR_C6596697.sample']), s=2)
