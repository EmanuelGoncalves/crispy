#!/usr/bin/env python
# Copyright (C) 2017 Emanuel Goncalves

import sys
import pickle
import crispy
import numpy as np
import pandas as pd
import seaborn as sns
import itertools as it
import matplotlib.pyplot as plt
from crispy import bipal_gray
from limix.qtl import qtl_test_lmm
from scipy.optimize import curve_fit
from matplotlib.gridspec import GridSpec
from sklearn.preprocessing import Imputer
from limix.stats import qvalues, linear_kinship
from limix_core.util.preprocess import gaussianize
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.metrics.regression import explained_variance_score


# - Import data-sets
# HGNC
hgnc = pd.read_csv('data/resources/hgnc/non_alt_loci_set.txt', sep='\t', index_col=1).dropna(subset=['location'])
hgnc = hgnc[hgnc['location'].apply(lambda x: not (x.startswith('X') or x.startswith('Y')))]
# hgnc = hgnc.query("locus_group != 'pseudogene'")

# Essential genes
essential = pd.read_csv('data/resources/curated_BAGEL_essential.csv', sep='\t')['gene'].rename('essential')

# Samplesheet
ss = pd.read_csv('data/gdsc/samplesheet.csv', index_col=0).dropna(subset=['Cancer Type'])

# sgRNA library
sglib = pd.read_csv('data/gdsc/crispr/KY_Library_v1.1_annotated.csv', index_col=0).dropna(subset=['GENES'])

# CRISPR
crispr = pd.read_csv('data/crispr_gdsc_crispy.csv', index_col=0)

# Copy-number absolute counts
cnv = pd.read_csv('data/gdsc/copynumber/Gene_level_CN.txt', sep='\t', index_col=0).dropna()

# Gene-expression
gexp = pd.read_csv('data/gdsc/gene_expression/merged_voom_preprocessed.csv', index_col=0)

# Mobems
mobems = pd.read_csv('data/gdsc/mobems/PANCAN_simple_MOBEM.rdata.annotated.csv', index_col=0)


# - Overlap
samples = list(set(crispr).intersection(cnv).intersection(gexp))
crispr, cnv = crispr[samples], cnv[samples]
print(crispr.shape, cnv.shape)


# - Filter
# CRISPR
crispr = crispr[(crispr.abs() >= 1).sum(1) >= 3]
crispr = crispr[(crispr < -1).sum(1) < (crispr.shape[1] * .9)]
crispr = crispr.drop(essential, axis=0, errors='ignore')

# CNV
genes = set(cnv.index).intersection(hgnc.index).intersection(gexp.index)

cnv_abs = cnv.loc[genes, samples].applymap(lambda v: int(v.split(',')[0]))

cnv_abs = cnv_abs.replace(-1, np.nan)
cnv_abs = pd.DataFrame(Imputer(strategy='most_frequent').fit_transform(cnv_abs), index=cnv_abs.index, columns=cnv_abs.columns)

cnv_abs = cnv_abs.loc[(cnv_abs < 1).sum(1) >= 3]
print(crispr.shape, cnv_abs.shape)


# -
# Covariates
covars = pd.get_dummies(ss.loc[samples, ['Cancer Type', 'Microsatellite']])

# Build matrices
y, x = crispr[samples].T, cnv_abs[samples].T

# Covars
k = pd.DataFrame(linear_kinship(gexp[samples].T.values), index=samples, columns=samples)

# Cap copy-number
x = x.clip(0, 3)

# Linear regression
lmm = qtl_test_lmm(pheno=y.values, snps=x.values, K=k.values)

# Extract info
c_gene, g_gene = zip(*list(it.product(y.columns, x.columns)))

# Build pandas data-frame
lmm_df = pd.DataFrame({
    'crispr': c_gene,
    'cnv': g_gene,
    'beta': lmm.getBetaSNP().ravel(),
    'pvalue': lmm.getPv().ravel(),
})
lmm_df = lmm_df[lmm_df['crispr'] != lmm_df['cnv']]
lmm_df = lmm_df.assign(qvalue=qvalues(lmm_df['pvalue'].values)).sort_values('qvalue')
print(lmm_df.head(30))


# - Plot
g_crispr, g_cnv = 'ARHGEF33', 'TUSC1'

plot_df = pd.concat([
    crispr.loc[g_crispr, samples].rename('crispr'),
    cnv_abs.loc[g_cnv, samples].rename('cnv'),
    mobems.loc[['loss:cnaPANCAN45']].T,
    ss.loc[samples, ['Cancer Type', 'SCREEN MEDIUM', 'Microsatellite']]
], axis=1).dropna().sort_values(['cnv', 'crispr'])

sns.boxplot('cnv', 'crispr', data=plot_df, palette='Reds_r', sym='')
sns.stripplot('cnv', 'crispr', data=plot_df, palette='Reds_r', jitter=.4, edgecolor='white', lw=.3)
plt.axhline(0, ls='-', lw=.3, c=bipal_gray[0])
plt.show()


# -
df = pd.concat([
    crispr.loc[g_crispr, samples].rename('crispr'),
    cnv_abs.loc[g_cnv, samples].rename('cnv'),
    mobems.T
], axis=1).dropna()

df_ench = pd.Series({i: roc_auc_score(df[i], df['cnv']) for i in df if (i not in ['crispr', 'cnv']) and (df[i].sum() >= 3)}).sort_values()
