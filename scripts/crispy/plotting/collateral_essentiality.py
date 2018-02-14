#!/usr/bin/env python
# Copyright (C) 2018 Emanuel Goncalves

import pickle
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from crispy import bipal_dbgd
from natsort import natsorted
from crispy.utils import bin_cnv
from crispy.regression.linear import lr
from statsmodels.stats.multitest import multipletests


def recurrent_amplified_genes(cnv, cnv_ratios, ploidy, genelist, min_events=3):
    g_amp = pd.DataFrame({s: (cnv[s] > 8 if ploidy[s] > 2.7 else cnv[s] > 4).astype(int) for s in cnv})
    g_amp = g_amp.loc[g_amp.index.isin(cnv_ratios.index) & g_amp.index.isin(genelist)]
    g_amp = g_amp[g_amp.sum(1) >= min_events]
    return list(g_amp.index)


def recurrent_non_expressed_genes(nexp, min_events=3.):
    g_nexp = nexp[nexp.sum(1) >= min_events]
    return list(g_nexp.index)


def filter_low_var_essential(crispr_fc, abs_thres=2, min_events=3):
    genes_thres = (crispr_fc.abs() > abs_thres).sum(1)
    genes_thres = genes_thres[genes_thres >= min_events]
    return list(genes_thres.index)


def lr_ess(xs, ys, non_expressed):
    # Run fit
    lm = lr(xs, ys)

    # Export lm results
    lm_res = pd.concat([lm[i].unstack().rename(i) for i in lm], axis=1)
    lm_res = lm_res.reset_index().rename(columns={'gene': 'gene_crispr', 'level_1': 'gene_ratios'})

    # Discard same gene associations
    lm_res = lm_res[lm_res['gene_crispr'] != lm_res['gene_ratios']]

    # Count genes that display high copy-number ratios and are not expressed for associations
    ratio_nexp_ov = (xs > 1.5).T.dot(non_expressed.loc[ys.columns, xs.index].T)
    lm_res = lm_res.assign(overlap=[ratio_nexp_ov.loc[x, y] for x, y in lm_res[['gene_ratios', 'gene_crispr']].values])

    # FDR
    lm_res = lm_res.assign(f_fdr=multipletests(lm_res['f_pval'], method='fdr_bh')[1]).sort_values('f_fdr')

    return lm_res


def association_boxplot(idx, lm_res, c_gdsc_fc, cnv, cnv_ratios, nexp, cnv_bin_thres=10, ratio_bin_thres=4):
    y_feat, x_feat = lm_res.loc[idx, 'gene_crispr'], lm_res.loc[idx, 'gene_ratios']

    plot_df = pd.concat([
        c_gdsc_fc.loc[y_feat, samples].rename('crispy'),
        cnv.loc[x_feat, samples].rename('cnv'),
        cnv_ratios.loc[x_feat, samples].rename('ratio'),
        nexp.loc[y_feat].replace({1.0: 'No', 0.0: 'Yes'}).rename('Expressed')
    ], axis=1).dropna().sort_values('crispy')

    # Bin copy-number profiles
    plot_df = plot_df.assign(ratio_bin=plot_df['ratio'].apply(lambda v: bin_cnv(v, ratio_bin_thres)).values)
    plot_df = plot_df.assign(cnv_bin=plot_df['cnv'].apply(lambda v: bin_cnv(v, cnv_bin_thres)).values)

    # Palettes
    color = sns.light_palette(bipal_dbgd[0])[2]
    pal = {'No': bipal_dbgd[1], 'Yes': bipal_dbgd[0]}

    # Plot
    f, (ax1, ax2) = plt.subplots(1, 2, sharex=False, sharey=True)

    # Copy-number boxplot
    cnvorder = natsorted(set(plot_df['cnv_bin']))

    sns.boxplot('cnv_bin', 'crispy', data=plot_df, order=cnvorder, color=color, sym='', linewidth=.3, saturation=.1, ax=ax1)
    sns.stripplot('cnv_bin', 'crispy', 'Expressed', data=plot_df, order=cnvorder, palette=pal, edgecolor='white', linewidth=.1, size=2, jitter=.1, alpha=.8, ax=ax1)

    ax1.axhline(0, ls='-', lw=.1, c=color)

    ax1.set_ylabel('%s\nCRISPR/Cas9 fold-change (log2)' % y_feat)
    ax1.set_xlabel('%s\nCopy-number' % x_feat)

    # Ratio boxplot
    order = natsorted(set(plot_df['ratio_bin']))

    sns.boxplot('ratio_bin', 'crispy', data=plot_df, order=order, color=color, sym='', linewidth=.3, saturation=.1, ax=ax2)
    sns.stripplot('ratio_bin', 'crispy', 'Expressed', data=plot_df, order=order, palette=pal, edgecolor='white', linewidth=.1, size=2, jitter=.1, alpha=.8, ax=ax2)

    ax2.axhline(0, ls='-', lw=.1, c=color)

    ax2.set_xlabel('%s\nCopy-number ratio' % x_feat)
    ax2.set_ylabel('')

    # Labels
    plt.suptitle('Collateral essentiality {0} - {1}\nbeta={2:.1f}; FDR={3:1.2e}'.format(y_feat, x_feat, lm_res.loc[idx, 'beta'], lm_res.loc[idx, 'f_fdr']), y=1.02)

    return y_feat, x_feat


# def plot_count_associations():
#     plt.barh(plot_df['ypos'], plot_df['count'], .8, color=bipal_dbgd[0], align='center')
#
#     for x, y in plot_df[['ypos', 'count']].values:
#         plt.text(y - .5, x, str(y), color='white', ha='right' if y != 1 else 'center', va='center', fontsize=10)
#
#     plt.yticks(plot_df['ypos'])
#     plt.yticks(plot_df['ypos'], plot_df['type'])
#
#     plt.xlabel('# significant associations (FDR < 5% and B < -0.5)')


if __name__ == '__main__':
    # - Import
    # Samplesheet
    ss = pd.read_csv('data/gdsc/samplesheet.csv', index_col=0)

    # Non-expressed genes
    nexp = pickle.load(open('data/gdsc/nexp_pickle.pickle', 'rb'))
    nexp = pd.DataFrame({c: {g: 1 for g in nexp[c]} for c in nexp}).fillna(0)

    # Copy-number
    cnv = pd.read_csv('data/crispy_copy_number_gene_snp.csv', index_col=0)

    # Copy-number ratios
    cnv_ratios = pd.read_csv('data/crispy_copy_number_gene_ratio_snp.csv', index_col=0)

    # Ploidy
    ploidy = pd.read_csv('data/crispy_copy_number_ploidy_snp.csv', index_col=0, names=['sample', 'ploidy'])['ploidy']

    # CRISPR
    c_gdsc_fc = pd.read_csv('data/crispr_gdsc_logfc.csv', index_col=0)

    # Cancer gene census list
    cgenes = pd.read_csv('data/gene_sets/Census_allThu Dec 21 15_43_09 2017.tsv', sep='\t')

    # - Overlap samples
    samples = list(set(cnv).intersection(c_gdsc_fc).intersection(nexp).intersection(ploidy.index))
    print('[INFO] Samples overlap: {}'.format(len(samples)))

    # - Recurrent amplified cancer genes (following COSMIC definition)
    g_amp = recurrent_amplified_genes(cnv[samples], cnv_ratios[samples], ploidy[samples], genelist=cgenes['Gene Symbol'])

    # - Recurrent non-expressed genes (Genes not expressed in at least 50% of the samples)
    g_nexp = recurrent_non_expressed_genes(nexp.reindex(c_gdsc_fc.index)[samples].dropna(), min_events=(len(samples) * .5))

    # - Filter-out genes with low CRISPR var
    g_nexp_var = filter_low_var_essential(c_gdsc_fc.loc[g_nexp, samples])

    # - Linear regressions
    lm_res = lr_ess(cnv_ratios.loc[g_amp, samples].T, c_gdsc_fc.loc[g_nexp_var, samples].T, nexp)
    lm_res.query('f_fdr < 0.05').to_csv('data/crispy_df_collateral_essentialities.csv', index=False)
    print(lm_res.query('f_fdr < 0.05 & beta < -.5'))

    # - Plot
    crispr_gene, ratio_gene = association_boxplot(3314, lm_res, c_gdsc_fc, cnv, cnv_ratios, nexp)
    plt.gcf().set_size_inches(4, 3)
    plt.savefig('reports/crispy/collateral_essentiality_boxplot_{}_{}.png'.format(crispr_gene, ratio_gene), bbox_inches='tight', dpi=600)
    plt.close('all')

    # - Plot
    # plot_count_associations(lm_res.query('f_fdr < 0.05 & beta < -.5')['gene_ratios'].value_counts())
