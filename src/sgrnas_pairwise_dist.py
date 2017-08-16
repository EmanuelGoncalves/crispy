#!/usr/bin/env python
# Copyright (C) 2017 Emanuel Goncalves

import re
import sys
import numpy as np
import pandas as pd
from limix_core.gp import GP
from datetime import datetime as dt
from sklearn.externals import joblib
from limix_core.mean import MeanBase
from limix_core.covar import FixedCov, SQExpCov, SumCov
from limix_core.util.preprocess import covar_rescaling_factor_efficient
from sklearn.gaussian_process.gpr import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, Sum, ConstantKernel


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


# -- Read arguments: sample and sample_chr
# sample, sample_chr = 'HT-29', '17'
sample, sample_chr = sys.argv[1:]
print('[%s] Sample: %s; Chr: %s' % (dt.now().strftime('%Y-%m-%d %H:%M:%S'), sample, sample_chr))


# -- Define sample data-frame
# sgRNA of the sample chromossome
sample_sgrnas = list(sgrna_lib[sgrna_lib['CHRM'] == sample_chr].index)

# Build sample data-frame
df = crispr[sample].dropna().rename('logfc')

# Concat sgRNAs information
df = pd.concat([sgrna_lib.loc[sample_sgrnas], df.loc[sample_sgrnas]], axis=1).dropna()

# Concat copy-number
df['cnv'] = [cnv_abs.loc[g, sample] if g in cnv_abs.index else np.nan for g in df['GENES']]
df['loh'] = [cnv_loh.loc[g, sample] if g in cnv_loh.index else np.nan for g in df['GENES']]

# Check if gene is essential
df['essential'] = [int(i in essential) for i in df['GENES']]
# df = df[(df['STARTpos'] > 35000000) & (df['ENDpos'] < 42000000)]


# -- Variance decomposition
print('[%s] GP for variance decomposition started' % dt.now().strftime('%Y-%m-%d %H:%M:%S'))

# Define variables
y, X = df[['logfc']].values, df[['STARTpos']]

# Scale distances
X /= 1e8

# Instanciate the covariance functions
K = ConstantKernel(.5, (1e-2, 1.)) * RBF(X.mean(), (1e-2, 10)) + WhiteKernel()

# Instanciate a Gaussian Process model
gp = GaussianProcessRegressor(K, n_restarts_optimizer=3, normalize_y=False)

# Fit to data using Maximum Likelihood Estimation of the parameters
gp.fit(X, y)

# Normalise data
df = df.assign(logfc_mean=gp.predict(X))
df = df.assign(logfc_norm=df['logfc'] - df['logfc_mean'])

# Export
joblib.dump(gp, 'reports/%s_%s_corrected.pkl' % (sample, sample_chr))
df.to_csv('reports/%s_%s_corrected.csv' % (sample, sample_chr))
print('[%s] GP for variance decomposition finshed' % dt.now().strftime('%Y-%m-%d %H:%M:%S'))


# # Define variables
# Y, X = df[['logfc']].values, df[['STARTpos']]
#
# # Scale distances
# X /= 1e9
#
# # Copy-number COV
# # cov_cna = pd.get_dummies(df['cnv'])
# # cov_cna = cov_cna.dot(cov_cna.transpose())
# #
# # cov_cna = FixedCov(cov_cna, cov_cna)
#
# # # LOH
# # cov_loh = pd.get_dummies(df['loh'])
# # cov_loh = cov_loh.dot(cov_loh.transpose())
# #
# # cov_loh = FixedCov(cov_loh, cov_loh)
# # cov_loh.scale *= covar_rescaling_factor_efficient(cov_loh.K())
# # cov_loh.scale /= 10
#
# # Distance COV
# cov_dist = SQExpCov(X, X)
# cov_dist.scale *= covar_rescaling_factor_efficient(cov_dist.K())
# cov_dist.length = np.max(X.values) * .1
#
# # Noise COV
# cov_noise = FixedCov(np.eye(X.shape[0]))
#
# # Sum COVs
# # 'cna': cov_cna,
# covs = {'distance': cov_dist, 'noise': cov_noise}
# K = SumCov(*covs.values())
#
# # GP
# gp = GP(MeanBase(Y), K)
# gp.optimize()
#
# # Explained variance
# cov_var = pd.Series({n: (1 / covar_rescaling_factor_efficient(covs[n].K())) for n in covs})
# cov_var /= cov_var.sum()
# print(cov_var.sort_values())
#
# # Normalise data
# df = df.assign(logfc_mean=gp.predict())
# df = df.assign(logfc_norm=df['logfc'] - df['logfc_mean'])
