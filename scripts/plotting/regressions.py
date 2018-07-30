#!/usr/bin/env python
# Copyright (C) 2018 Emanuel Goncalves

import numpy as np
import crispy as cy
import pandas as pd
import scripts as mp
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from sklearn.model_selection import ShuffleSplit
from sklearn.linear_model import LinearRegression, Ridge


def pearson_fc(Y, X):
    genes = set(Y.index).intersection(X.index)

    gene_cvs = []
    for gene in genes:
        # Build data-frame
        y = Y.loc[gene, samples].dropna()
        x = X.loc[gene, y.index]

        # Pearson
        r, p = pearsonr(y, x)

        # Append
        gene_cvs.append(dict(gene=gene, pearson=r, pval=p))

    gene_cvs = pd.DataFrame(gene_cvs).groupby('gene').median()

    return gene_cvs


def pearson_fc_per_chr(Y, X):
    clines = list(set(X).intersection(Y))
    genes = set(Y.index).intersection(X.index)

    lib = cy.get_crispr_lib().groupby('gene')['chr'].first()[genes].reset_index()

    gene_cvs = []

    for s in clines:
        for c, df in lib.groupby('chr'):
            g = list(df['gene'])

            # Build data-frame
            y = Y.loc[g, s]
            x = X.loc[g, s]

            # Pearson
            r, p = pearsonr(y, x)

            # Append
            gene_cvs.append(dict(chrm=c, sample=s, pearson=r, pval=p))

    gene_cvs = pd.DataFrame(gene_cvs).dropna()

    return gene_cvs


def lm_fc(Y, X, n_splits=10, test_size=.5, ploidy=None, chrm=None):
    genes = set(Y.index).intersection(X.index)

    lib = cy.get_crispr_lib().groupby('gene')['chr'].first()[genes]

    gene_cvs = []
    for gene in genes:
        # Build data-frame
        y = Y.loc[gene, samples].dropna()
        x = X.loc[[gene], y.index].T

        if ploidy is not None:
            x = pd.concat([x, ploidy[samples]], axis=1)

        if chrm is not None:
            x = pd.concat([x, chrm.loc[lib[gene], samples]], axis=1)

        # Cross-validation
        cv = ShuffleSplit(n_splits=n_splits, test_size=test_size)

        for i, (train_idx, test_idx) in enumerate(cv.split(y)):
            # Train model
            lm = LinearRegression().fit(x.iloc[train_idx], y.iloc[train_idx])

            # Evalutate model
            r2 = lm.score(x.iloc[test_idx], y.iloc[test_idx])

            # Store
            gene_cvs.append(dict(r2=r2, gene=gene))

    gene_cvs = pd.DataFrame(gene_cvs).groupby('gene').median()

    return gene_cvs


if __name__ == '__main__':
    # - Import
    # Copy-number
    cnv = mp.get_copynumber().dropna()

    # Copy-number ratios
    cnv_ratios = mp.get_copynumber_ratios().dropna()

    # Non-expressed genes
    nexp = mp.get_non_exp()

    # CRISPR
    crispr = mp.get_crispr(mp.CRISPR_GENE_FC)

    # Ploidy
    ploidy = mp.get_ploidy()

    # Chromosome copies
    chrm = mp.get_chr_copies()

    # - Overlap samples
    samples = list(set(cnv).intersection(crispr).intersection(nexp))
    print('[INFO] Samples overlap: {}'.format(len(samples)))

    # - Predict gene CRISPR fold-change
    genes = set(crispr.index).intersection(cnv.index)
    # genes = cy.get_non_essential_genes().intersection(genes)

    pred_cnv = lm_fc(crispr.loc[genes, samples], cnv.loc[genes, samples])
    pred_cnv_covars = lm_fc(crispr.loc[genes, samples], cnv.loc[genes, samples], ploidy=ploidy, chrm=chrm)

    df = pd.concat([
        pred_cnv['r2'].rename('null'), pred_cnv_covars['r2'].rename('full')
    ], axis=1).dropna()

    sns.distplot(df.eval('full - null'))
    plt.show()

    g = 'C2orf71'
    plt.scatter(cnv.loc[g, samples], crispr.loc[g, samples]); plt.show()
    plt.scatter(cnv_ratios.loc[g, samples], crispr.loc[g, samples]); plt.show()
