#!/usr/bin/env python
# Copyright (C) 2017 Emanuel Goncalves

import numpy as np
import pandas as pd
import itertools as it
from limix.stats import qvalues
from limix.qtl import qtl_test_lmm
from sklearn.metrics import roc_auc_score


# - Cell lines samplesheet
samplesheet = pd.read_csv('data/gdsc/samplesheet.csv', index_col=0).dropna(subset=['GDSC1'])

# - Import CRISPR
crispr_mageck = pd.read_csv('data/gdsc/crispr/mageck_gene_fold_changes_mean.csv', index_col=0).dropna()
crispr_bagel = pd.read_csv('data/gdsc/crispr/crispr_bagel_gene_level_normalised.csv', index_col=0).dropna()


# - Import gene-sets
essential = list(pd.read_csv('data/resources/curated_BAGEL_essential.csv')['gene'])
non_essential = list(pd.read_csv('data/resources/curated_BAGEL_nonEssential.csv')['gene'])


# - AROC gene essential
mageck_essential = [roc_auc_score([int(i in essential) for i in crispr_mageck[c].index], -crispr_mageck[c]) for c in crispr_mageck]
bagel_essential = [roc_auc_score([int(i in essential) for i in crispr_bagel[c].index], crispr_bagel[c]) for c in crispr_bagel]
print('Mageck: %.2f; Bagel: %.2f' % (np.mean(mageck_essential), np.mean(bagel_essential)))


# - Linear model fit
# Import MOBEM
mobems = pd.read_csv('data/gdsc/mobems/PANCAN_simple_MOBEM.rdata.annotated.csv', index_col=0)

# Overlap
samples = list(set(crispr_mageck).intersection(mobems).intersection(samplesheet.index))
crispr_mageck, mobems = crispr_mageck[samples], mobems[samples]
print('Samples: %d' % len(samples))

# Filter
mobems = mobems[mobems.sum(1) >= 5]

# Tissue covariance matrix
c_tissue = samplesheet.loc[samples, 'GDSC1'].str.get_dummies()
c_tissue = c_tissue.dot(c_tissue.transpose()).astype(float).loc[samples]

# Linear model
lmm = qtl_test_lmm(pheno=crispr_mageck.T.values, snps=mobems.T.values, K=c_tissue.values)

# Extract solutions
gene, event = zip(*list(it.product(crispr_mageck.index, mobems.index)))
qvalue = qvalues(lmm.getPv())

lmm = pd.DataFrame({"gene": gene, "genomic": event, "beta": lmm.getBetaSNP().ravel(), "pvalue": lmm.getPv().ravel(), "qvalue": qvalue.ravel()})
print(lmm.sort_values('pvalue'))
