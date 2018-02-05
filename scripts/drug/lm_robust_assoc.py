#!/usr/bin/env python
# Copyright (C) 2018 Emanuel Goncalves

import numpy as np
import pandas as pd
import seaborn as sns
import scripts.drug as dc
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from crispy import bipal_dbgd
from crispy.utils import qnorm
from scipy.stats import pearsonr
from crispy.regression.linear import lr
from statsmodels.stats.multitest import multipletests
from scripts.drug.lm_drug_crispr import plot_corrplot, plot_drug_associations_barplot


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


def marginal_boxplot(a, xs=None, ys=None, zs=None, vertical=False, **kws):
    ax = sns.boxplot(x=zs, y=ys, orient='v', **kws) if vertical else sns.boxplot(x=xs, y=zs, orient='h', **kws)
    ax.set_ylabel('')
    ax.set_xlabel('')


def plot_corr_discrete(x, y, z):
    scatter_kws = {'s': 20, 'edgecolor': 'w', 'linewidth': .5, 'alpha': .8}

    plot_df = pd.concat([x, y, z], axis=1)

    g = sns.JointGrid(x.name, y.name, plot_df, space=0, ratio=8)

    g.plot_marginals(
        marginal_boxplot, palette=bipal_dbgd, data=plot_df, linewidth=.3, fliersize=1, notch=False, saturation=1.0,
        xs=x.name, ys=y.name, zs=z.name
    )

    sns.regplot(
        x=x.name, y=y.name, data=plot_df, color=bipal_dbgd[0], truncate=True, fit_reg=True, scatter_kws=scatter_kws, line_kws={'linewidth': .5}, ax=g.ax_joint
    )
    sns.regplot(
        x=x.name, y=y.name, data=plot_df[plot_df[z.name] == 1], color=bipal_dbgd[1], truncate=True, fit_reg=False, scatter_kws=scatter_kws, ax=g.ax_joint
    )

    g.annotate(pearsonr, template='Pearson={val:.2g}, p={p:.1e}', loc=4)

    g.ax_joint.axhline(0, ls='-', lw=0.3, c='black', alpha=.2)
    g.ax_joint.axvline(0, ls='-', lw=0.3, c='black', alpha=.2)

    g.set_axis_labels('{} (log10 FC)'.format(x.name), '{} (ln IC50)'.format(y.name))

    handles = [mpatches.Circle([.0, .0], .25, facecolor=c, label='Yes' if t else 'No') for t, c in bipal_dbgd.items()]
    g.ax_marg_y.legend(handles=handles, title='', loc='center left', bbox_to_anchor=(1, 0.5))

    plt.suptitle(z.name, y=1.05, fontsize=8)


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
    # lm_res_df = pd.read_csv('data/drug/lm_drug_crispr_genomic.csv')

    # - Plot
    # Volcanos
    plot_volcano(lm_res_df)

    # Plot Drug barplot
    plot_drug_associations_barplot(lm_df[lm_df['lr_fdr'] < 0.15], ['Linsitinib'])
    plt.title('Top significant associations')
    plt.gcf().set_size_inches(2, 1.5)
    plt.savefig('reports/drug/drug_associations_barplot_single.png', bbox_inches='tight', dpi=600)
    plt.close('all')

    # Corr plot
    idx = 2332
    d_id, d_name, d_screen, gene = lm_df.loc[idx, ['DRUG_ID', 'DRUG_NAME', 'VERSION', 'Gene']].values
    d_id, d_name, d_screen, gene = 1510, 'Linsitinib', 'RS', 'FURIN'

    plot_corrplot(
        crispr.loc[gene].rename('{} CRISPR'.format(gene)), d_response.loc[(d_id, d_name, d_screen)].rename('{} {} Drug'.format(d_name, d_screen))
    )

    plt.gcf().set_size_inches(2., 2.)
    plt.savefig('reports/drug/crispr_drug_corrplot.png', bbox_inches='tight', dpi=600)
    plt.close('all')

    # Corr plot discrete
    idx = 0
    d_id, d_name, d_screen, gene, genomic = lm_res_df.loc[idx, ['drug_DRUG_ID', 'drug_DRUG_NAME', 'drug_VERSION', 'crispr_Gene', 'genomic']].values
    d_id, d_name, d_screen, gene, genomic = 1560, 'Alpelisib', 'RS', 'FOXA1', 'gain:cnaPANCAN301 (CDK12,ERBB2,MED24)'

    plot_corr_discrete(
        x=crispr.loc[gene].rename('{} CRISPR'.format(gene)),
        y=d_response.loc[(d_id, d_name, d_screen)].rename('{} {} Drug'.format(d_name, d_screen)),
        z=mobems.loc[genomic].rename(genomic)
    )
    plt.gcf().set_size_inches(2, 2)
    plt.savefig('reports/drug/crispr_drug_genomic_corrplot.png', bbox_inches='tight', dpi=600)
    plt.close('all')
