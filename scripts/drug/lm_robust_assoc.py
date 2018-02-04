#!/usr/bin/env python
# Copyright (C) 2018 Emanuel Goncalves

import numpy as np
import pandas as pd
import seaborn as sns
import scripts.drug as dc
import matplotlib.pyplot as plt
from crispy import bipal_dbgd
from crispy.utils import qnorm
from crispy.regression.linear import lr
from statsmodels.stats.multitest import multipletests


def lm_drugcrispr_mobem(xs, ys_1, ys_2, ws, pairs):
    print('<Drug, CRISPR>: %d; Genomics: %d' % (len(pairs), xs.shape[1]))

    # Linear regressions of drug and crispr
    lm_res = []
    for gene, drug_id, drug_name, drug_screen in ppairs:
        # Linear regression CRISPR
        lm_res_crispr = lr(xs, ys_1[[gene]], ws)

        # Linear regression Drug
        lm_res_drug = lr(xs, ys_2[[(drug_id, drug_name, drug_screen)]], ws)

        # Melt result matrices
        lm_res_crispr_df = pd.concat([lm_res_crispr[i].unstack().rename(i) for i in lm_res_crispr], axis=1)
        lm_res_crispr_df = lm_res_crispr_df.reset_index().set_index('genomic').add_prefix('crispr_')

        lm_res_drug_df = pd.concat([lm_res_drug[i].unstack().rename(i) for i in lm_res_drug], axis=1)
        lm_res_drug_df = lm_res_drug_df.reset_index().set_index('genomic').add_prefix('drug_')

        # Concat data-frames
        lm_res.append(pd.concat([lm_res_crispr_df, lm_res_drug_df], axis=1).reset_index())

    # Merge results
    lm_res = pd.concat(lm_res)

    lm_res = lm_res.assign(drug_lr_fdr=multipletests(lm_res['drug_lr_pval'], method='fdr_bh')[1])
    lm_res = lm_res.assign(crispr_lr_fdr=multipletests(lm_res['crispr_lr_pval'], method='fdr_bh')[1])

    return lm_res


def plot_volcano(lm_res):
    for name in ['drug', 'crispr']:
        plot_df = lm_res.copy()

        plot_df['signif'] = ['*' if i < 0.05 else '-' for i in plot_df['{}_lr_fdr'.format(name)]]

        hue_order = ['*', '-']
        pal = dict(zip(*(hue_order, sns.light_palette(bipal_dbgd[0], len(hue_order) + 1, reverse=True).as_hex()[:-1])))

        # Scatter
        for i in ['-', '*']:
            plot_df_ = plot_df.query("(signif == '{}')".format(i))
            plt.scatter(
                plot_df_['{}_beta'.format(name)], -np.log10(plot_df_['{}_lr_pval'.format(name)]),
                edgecolor='white', lw=.1, s=3, alpha=1., c=pal[i]
            )

        # Add FDR threshold lines
        yy = plot_df.loc[(plot_df['{}_lr_fdr'.format(name)] - 0.05).abs().sort_values().index[0], '{}_lr_pval'.format(name)]
        plt.text(plt.xlim()[0] * .98, -np.log10(yy) * 1.01, 'FDR 5%', ha='left', color=pal['*'], alpha=0.65, fontsize=5)
        plt.axhline(-np.log10(yy), c=pal['*'], ls='--', lw=.3, alpha=.7)

        # Add axis lines
        plt.axvline(0, c=bipal_dbgd[0], lw=.1, ls='-', alpha=.3)

        # Add axis labels and title
        plt.title('Linear regression: {} ~ GENOMIC'.format(name.upper()), fontsize=8, fontname='sans-serif')
        plt.xlabel('Model coefficient (beta)', fontsize=8, fontname='sans-serif')
        plt.ylabel('Log-ratio test p-value (-log10)', fontsize=8, fontname='sans-serif')

        # Save plot
        plt.gcf().set_size_inches(2., 4.)
        plt.savefig('reports/drug/lm_{}_genomic_volcano.png'.format(name), bbox_inches='tight', dpi=300)
        plt.close('all')


if __name__ == '__main__':
    # - Imports
    # Crispr ~ Drug associations
    lm_df = pd.read_csv('data/drug/lm_drug_crispr.csv')

    # Samplesheet
    ss = pd.read_csv(dc.SAMPLESHEET_FILE, index_col=0).dropna(subset=['Cancer Type'])

    # Growth rate
    growth = pd.read_csv(dc.GROWTHRATE_FILE, index_col=0)

    # CRISPR gene-level corrected fold-changes
    crispr = pd.read_csv(dc.CRISPR_GENE_FC_CORRECTED, index_col=0, sep='\t').dropna()
    crispr_qnorm = pd.DataFrame({c: qnorm(crispr[c]) for c in crispr}, index=crispr.index)

    # Drug response
    d_response = pd.read_csv(dc.DRUG_RESPONSE_FILE, index_col=[0, 1, 2], header=[0, 1])
    d_response.columns = d_response.columns.droplevel(0)

    # MOBEMs
    mobems = pd.read_csv(dc.MOBEM_FILE, index_col=0)
    mobems.index.name = 'genomic'

    # - Overlap
    samples = list(set.intersection(set(d_response), set(crispr), set(ss.index), set(growth.index), set(mobems)))
    d_response, crispr, mobems = d_response[samples], crispr[samples], mobems[samples]
    print('Samples: %d' % len(samples))

    # - Filter MOBEM
    mobems = dc.filter_mobem(mobems)

    # - Covariates
    covariates = pd.concat([
        pd.get_dummies(ss[['Cancer Type']]),
        growth['growth_rate_median']
    ], axis=1).loc[samples]
    covariates = covariates.loc[:, covariates.sum() != 0]

    # - Rosbut pharmacogenomic associations
    a_thres = 0.05

    ppairs = lm_df.query('lr_fdr < {}'.format(a_thres))[['Gene', 'DRUG_ID', 'DRUG_NAME', 'VERSION']].values

    lm_res_df = lm_drugcrispr_mobem(
        mobems[samples].T, crispr_qnorm[samples].T, d_response[samples].T, covariates.loc[samples], ppairs
    )

    lm_res_df.sort_values('crispr_lr_fdr').to_csv('data/drug/lm_drug_crispr_genomic.csv', index=False)
    print(lm_res_df.query('crispr_lr_fdr < 0.05').sort_values('crispr_lr_fdr'))

    # - Plot
    # Volcanos
    plot_volcano(lm_res_df)
