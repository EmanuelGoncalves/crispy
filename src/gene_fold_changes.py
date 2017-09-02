#!/usr/bin/env python
# Copyright (C) 2017 Emanuel Goncalves

import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from limix.qtl import qtl_test_lm


# -  Improt
# CRISPR
crispr = pd.read_csv('data/gdsc/crispr/crispy_fold_change.csv', index_col=0)

# sgRNA library
sgrna_lib = pd.read_csv('data/gdsc/crispr/KY_Library_v1.1_annotated.csv', index_col=0).dropna(subset=['GENES'])
sgrna_lib_genes = sgrna_lib.reset_index().groupby('GENES')['index'].agg(lambda x: set(x)).to_dict()

# -
m_betas, m_pvals = {}, {}
for g in sgrna_lib_genes:
    # g = 'A1BG'

    y = crispr.loc[sgrna_lib_genes[g]].unstack().dropna().to_frame()
    x = pd.get_dummies(y.index.get_level_values(0)).astype(int)

    c_lm = qtl_test_lm(pheno=y.values, snps=x.values)

    m_betas[g] = dict(zip(*(list(x), c_lm.getBetaSNP()[0])))
    m_pvals[g] = dict(zip(*(list(x), c_lm.getPv()[0])))
    print(g)

m_betas, m_pvals = pd.DataFrame(m_betas).T, pd.DataFrame(m_pvals).T

# - Export
m_betas.to_csv('data/gdsc/crispr/crispy_fold_change_glm_betas.csv')
m_pvals.to_csv('data/gdsc/crispr/crispy_fold_change_glm_pvals.csv')
