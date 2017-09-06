#!/usr/bin/env python
# Copyright (C) 2017 Emanuel Goncalves

import sys
import random
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime as dt
from sklearn.metrics import r2_score
from sklearn.model_selection import ShuffleSplit
from sklearn.gaussian_process.gpr import GaussianProcessRegressor
from limix_core.util.preprocess import covar_rescaling_factor_efficient
from sklearn.gaussian_process.kernels import WhiteKernel, ConstantKernel, RationalQuadratic, PairwiseKernel


# -- Imports
# Essential genes
essential = list(pd.read_csv('data/resources/curated_BAGEL_essential.csv', sep='\t')['gene'])

# sgRNA library
sgrna_lib = pd.read_csv('data/gdsc/crispr/KY_Library_v1.1_annotated.csv', index_col=0).dropna(subset=['STARTpos'])

# Chromosome cytobands
cytobands = pd.read_csv('data/resources/cytoBand.txt', sep='\t')
print('[%s] Misc files imported' % dt.now().strftime('%Y-%m-%d %H:%M:%S'))

# CRISPR
crispr = pd.read_csv('data/gdsc/crispr/deseq_gene_fold_changes.csv', index_col=0)

# CIRSPR segments
crispr_seg = pd.read_csv('data/gdsc/crispr/segments_cbs_deseq2.csv')
print('[%s] CRISPR data imported' % dt.now().strftime('%Y-%m-%d %H:%M:%S'))

# Copy-number absolute counts
cnv = pd.read_csv('data/gdsc/copynumber/Gene_level_CN.txt', sep='\t', index_col=0)
cnv = cnv.loc[{i.split('_')[0] for i in crispr.index}, crispr.columns].dropna(how='all').dropna(how='all', axis=1)
cnv_abs = cnv.applymap(lambda v: int(v.split(',')[0]))
cnv_loh = cnv.applymap(lambda v: v.split(',')[2])

# Copy-number segments
cnv_seg = pd.read_csv('data/gdsc/copynumber/Summary_segmentation_data_994_lines_picnic.txt', sep='\t')
cnv_seg['chr'] = cnv_seg['chr'].replace(23, 'X').replace(24, 'Y').astype(str)
print('[%s] Copy-number data imported' % dt.now().strftime('%Y-%m-%d %H:%M:%S'))


# - Read arguments: sample and sample_chr
sample, sample_chr = 'AU565', '17'

# sgRNA of the sample chromossome
sample_sgrnas = list(sgrna_lib[sgrna_lib['CHRM'] == sample_chr].index)

# Build sample data-frame
df = crispr[sample].dropna().rename('logfc')

# Concat sgRNAs information
df = pd.concat([sgrna_lib.loc[sample_sgrnas], df.loc[sample_sgrnas]], axis=1).dropna()

# Average per Gene
df = df.groupby('GENES').agg({
    'CODE': 'first',
    'CHRM': 'first',
    'STARTpos': np.mean,
    'ENDpos': np.mean,
    'logfc': np.mean
})

# Concat copy-number
df['cnv'] = [cnv_abs.loc[g, sample] if g in cnv_abs.index else np.nan for g in df.index]
df['loh'] = [cnv_loh.loc[g, sample] if g in cnv_loh.index else np.nan for g in df.index]

# Check if gene is essential
df['essential'] = [int(i in essential) for i in df.index]
# df = df[(df['STARTpos'] > 20000000) & (df['ENDpos'] < 42000000)]
# df = df[(df['STARTpos'] > 50000000) & (df['ENDpos'] < 130000000)]
print('[%s] Sample: %s; Chr: %s' % (dt.now().strftime('%Y-%m-%d %H:%M:%S'), sample, sample_chr))


# - Out-of-sample hyper-parameters estimation
parms = []
print('[%s] GP for variance decomposition started' % dt.now().strftime('%Y-%m-%d %H:%M:%S'))

# Define variables
y, X = df[['logfc']], df[['STARTpos']]

# Scale distances
X /= 1e9

# Define params
for alpha_lb in [1e-4, 1e-3, 1e-2, 1e-1, 1, 10]:

    for length_scale_lb in [1e-4, 1e-3, 1e-2, 1e-1]:

        # Cross-validation
        cv = ShuffleSplit(n_splits=10, test_size=.4)

        for idx_train, idx_test in cv.split(X):
            # Subsample
            y_train, X_train = y.iloc[idx_train], X.iloc[idx_train]
            y_test, X_test = y.iloc[idx_test], X.iloc[idx_test]

            # Gaussian preprocess model
            k = ConstantKernel() * RationalQuadratic(length_scale_bounds=(length_scale_lb, 10), alpha_bounds=(alpha_lb, 10)) + WhiteKernel()

            gp = GaussianProcessRegressor(k, n_restarts_optimizer=3, normalize_y=True)

            # Fit to data using Maximum Likelihood Estimation of the parameters
            gp.fit(X_train, y_train)

            # Evalute
            y_pred = pd.DataFrame(gp.predict(X_test, return_std=False), index=X_test.index, columns=['logfc'])
            r2 = r2_score(y_test, y_pred)

            # Hyperparameters + metrics
            kernels = [('RQ', gp.kernel_.k1), ('Noise', gp.kernel_.k2)]

            df_hp = {'%s_%s' % (n, p.name): k.get_params()[p.name] for n, k in kernels for p in k.hyperparameters}
            df_hp['r2'] = r2
            df_hp['ll'] = gp.log_marginal_likelihood_value_
            df_hp['alpha_lb'] = alpha_lb
            df_hp['length_scale_lb'] = length_scale_lb

            parms.append(df_hp)
            print('[%s] R2=%.2f: %s' % (dt.now().strftime('%Y-%m-%d %H:%M:%S'), r2, gp.kernel_))

parms = pd.DataFrame(parms).sort_values('r2')
parms.to_csv('data/sampling_params_custom_cv.csv', index=False)


# -
plot_df = pd.pivot_table(parms, index='alpha_lb', columns='length_scale_lb', values='r2', aggfunc=np.mean)

sns.set(style='ticks', context='paper', font_scale=.75, rc={'axes.linewidth': .3, 'xtick.major.width': .3, 'ytick.major.width': .3, 'xtick.major.size': 2.5, 'ytick.major.size': 2.5})
g = sns.heatmap(plot_df, cmap='viridis', annot=True, fmt='.2f')
plt.title('Out-of-sample prediction, R2')
plt.setp(g.get_yticklabels(), rotation=0)
plt.gcf().set_size_inches(3, 3)
plt.savefig('reports/gp_subsampling_heatmap.png', bbox_inches='tight', dpi=600)
plt.close('all')


# # -
# plot_df = pd.DataFrame({'y_true': y_test['logfc'], 'y_pred': y_pred['logfc']})
#
# sns.set(style='ticks', context='paper', font_scale=.75, rc={'axes.linewidth': .3, 'xtick.major.width': .3, 'ytick.major.width': .3, 'xtick.major.size': 2.5, 'ytick.major.size': 2.5})
# g = sns.jointplot(
#     'y_true', 'y_pred', data=plot_df, kind='reg', color='#34495e', space=0,
#     marginal_kws={'hist': True, 'rug': False, 'kde': False, 'bins': 20}, annot_kws={'template': 'R={val:.2g}, pvalue={p:.1e}', 'loc': 4},
#     joint_kws={'scatter_kws': {'s': 5, 'edgecolor': 'w', 'linewidth': .3, 'alpha': .5}, 'line_kws': {'linewidth': .5}, 'truncate': True}
# )
#
# g.ax_joint.axvline(0, ls='-', lw=0.3, c='black', alpha=.2)
# g.ax_joint.axhline(0, ls='-', lw=0.3, c='black', alpha=.2)
# g.set_axis_labels('y true', 'y pred')
# plt.gcf().set_size_inches(2, 2)
# plt.savefig('reports/gp_subsampling_pred_corrplot.png', bbox_inches='tight', dpi=600)
# plt.close('all')
