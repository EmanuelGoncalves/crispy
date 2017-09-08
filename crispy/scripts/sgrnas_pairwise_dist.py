#!/usr/bin/env python
# Copyright (C) 2017 Emanuel Goncalves

import sys
import numpy as np
import pandas as pd
from datetime import datetime as dt
from sklearn.externals import joblib
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
crispr = pd.read_csv('data/gdsc/crispr/crispy_gdsc_fold_change_sgrna.csv', index_col=0)

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
sample, sample_chr = 'AU565', '8'
# sample, sample_chr = sys.argv[1:]

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


# - Fill missing values with mean
df['cnv'] = df['cnv'].replace(np.nan, df['cnv'].median())
df['loh'] = df['loh'].replace(np.nan, 'H')
df = df.sort_values('STARTpos')


# - Variance decomposition
print('[%s] GP for variance decomposition started' % dt.now().strftime('%Y-%m-%d %H:%M:%S'))

# Define variables
y, X = df[['logfc']].values, df[['STARTpos']]

# # Scale distances
X /= 1e9

# Define parms bounds
length_scale_lb, alpha_lb = 1e-3, 1e-3

# Instanciate the covariance functions
K = ConstantKernel() * RationalQuadratic(length_scale_bounds=(length_scale_lb, 10), alpha_bounds=(alpha_lb, 10)) + WhiteKernel()
# K = ConstantKernel() * PairwiseKernel(metric='rbf') + WhiteKernel()

# Instanciate a Gaussian Process model
gp = GaussianProcessRegressor(K, n_restarts_optimizer=3, normalize_y=True)

# Fit to data using Maximum Likelihood Estimation of the parameters
gp.fit(X, y)
print(gp.kernel_)

# Normalise data
logfc_mean, logfc_se = gp.predict(X, return_std=True)

df = df.assign(logfc_se=logfc_se)
df = df.assign(logfc_mean=logfc_mean)
df = df.assign(logfc_norm=df['logfc'] - df['logfc_mean'])

# Export hyperparameters
kernels = [('RQ', gp.kernel_.k1), ('Noise', gp.kernel_.k2)]

# Variance explained
cov_var = pd.Series({n: (1 / covar_rescaling_factor_efficient(k.__call__(X))) for n, k in kernels}, name='Variance')
cov_var /= cov_var.sum()
print(cov_var.sort_values())

df_hp = {}
for n, k in kernels:
    df_hp.update({'%s_%s' % (n, p.name): k.get_params()[p.name] for p in k.hyperparameters})
pd.Series(df_hp).to_csv('reports/%s_%s_optimised_params.csv' % (sample, sample_chr))
print(pd.Series(df_hp))

# Export
joblib.dump(gp, 'reports/%s_%s_corrected.pkl' % (sample, sample_chr))
df.to_csv('reports/%s_%s_corrected.csv' % (sample, sample_chr))
print('[%s] GP for variance decomposition finshed' % dt.now().strftime('%Y-%m-%d %H:%M:%S'))
