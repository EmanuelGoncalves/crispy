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
from crispy.benchmark_plot import plot_chromosome, import_cytobands


def recurrent_amplified_genes(cnv, cnv_ratios, ploidy, min_events=3):
    g_amp = pd.DataFrame({s: (cnv[s] > 8 if ploidy[s] > 2.7 else cnv[s] > 4).astype(int) for s in cnv})
    g_amp = g_amp.loc[g_amp.index.isin(cnv_ratios.index)]
    g_amp = g_amp[g_amp.sum(1) >= min_events]
    return list(g_amp.index)


def recurrent_non_expressed_genes(nexp, min_events=3):
    g_nexp = nexp[nexp.sum(1) >= min_events]
    return list(g_nexp.index)


def filter_low_var_essential(crispr_fc, abs_thres=1, min_events=3):
    genes_thres = (crispr_fc.abs() > abs_thres).sum(1)
    genes_thres = genes_thres[genes_thres >= min_events]
    return list(genes_thres.index)


def lr_ess(xs, ys, non_expressed, ratio_thres=1.5):
    # Run fit
    lm = lr(xs, ys)

    # Export lm results
    lm_res = pd.concat([lm[i].unstack().rename(i) for i in lm], axis=1)
    lm_res = lm_res.reset_index().rename(columns={'gene': 'gene_crispr', 'level_1': 'gene_ratios'})

    # Discard same gene associations
    lm_res = lm_res[lm_res['gene_crispr'] != lm_res['gene_ratios']]

    # Calculate the percentages of cell lines in which X is amplified and Y is not expressed
    ratio_nexp_ov = (xs > ratio_thres).T.dot(non_expressed.loc[ys.columns, xs.index].T).T
    ratio_nexp_ov = (ratio_nexp_ov.divide((xs > ratio_thres).sum()) * 100).round(1)

    lm_res = lm_res.assign(
        overlap=[ratio_nexp_ov.loc[x, y] for x, y in lm_res[['gene_crispr', 'gene_ratios']].values]
    )

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


def plot_volcano(lm_res):
    # - Plot
    pal = dict(zip(*([0, 1], sns.light_palette(bipal_dbgd[0], 3).as_hex()[1:])))

    # Scatter dot sizes
    lm_res = lm_res.assign(
        bin_overlap=pd.cut(lm_res['overlap'], [0, 50, 80, 90, 100], labels=['0-50', '50-80', '80-90', '90-100'], include_lowest=True)
    )
    bin_order = natsorted(set(lm_res['bin_overlap']))
    bin_sizes = dict(zip(*(bin_order, np.arange(1, len(bin_order) * 4, 4))))

    # Scatter
    for i in pal:
        plt.scatter(
            x=lm_res.query('signif == {}'.format(i))['beta'], y=-np.log10(lm_res.query('signif == {}'.format(i))['f_fdr']),
            s=lm_res.query('signif == {}'.format(i))['bin_overlap'].apply(lambda v: bin_sizes[v]),
            c=pal[i], linewidths=.05, edgecolors='white', alpha=.5 if i == 1 else 0.3, label='Significant' if i == 1 else 'Non-significant'
        )

    # Add FDR threshold lines
    plt.axhline(-np.log10(0.05), c=pal[1], ls='--', lw=.1, alpha=.3)

    # Add beta lines
    plt.axvline(-0.5, c=pal[1], ls='--', lw=.1, alpha=.3)
    plt.axvline(0.5, c=pal[1], ls='--', lw=.1, alpha=.3)

    # Add axis lines
    plt.axvline(0, c=bipal_dbgd[0], lw=.05, ls='-', alpha=.3)

    # Add axis labels and title
    plt.title('Collateral essentiality')
    plt.xlabel('Model coefficient (beta)')
    plt.ylabel('F-test FDR (-log10)')

    # Legend
    [plt.scatter([], [], bin_sizes[k], label='Overlap: {}%'.format(k), c=pal[1], alpha=.5) for k in bin_sizes]
    plt.legend(loc='center left', bbox_to_anchor=(1.02, 0.5))

    # Axis range
    plt.xlim(xmax=0)


if __name__ == '__main__':
    # - Import
    # Samplesheet
    ss = pd.read_csv('data/gdsc/samplesheet.csv', index_col=0)

    # Non-expressed genes
    nexp = pickle.load(open('data/gdsc/nexp_pickle.pickle', 'rb'))
    nexp = pd.DataFrame({c: {g: 1 for g in nexp[c]} for c in nexp}).fillna(0)

    # Copy-number
    cnv = pd.read_csv('data/crispy_copy_number_gene_snp.csv', index_col=0).dropna()

    # Copy-number ratios
    cnv_ratios = pd.read_csv('data/crispy_copy_number_gene_ratio_snp.csv', index_col=0).dropna()

    # Ploidy
    ploidy = pd.read_csv('data/crispy_copy_number_ploidy_snp.csv', index_col=0, names=['sample', 'ploidy'])['ploidy']

    # CRISPR
    c_gdsc_fc = pd.read_csv('data/crispr_gdsc_logfc.csv', index_col=0)

    # CRISPR lib
    lib = pd.read_csv('data/crispr_libs/KY_Library_v1.1_updated.csv', index_col=0)
    lib = lib.assign(pos=lib[['start', 'end']].mean(1).values)

    # Cancer gene census list
    cgenes = set(pd.read_csv('data/gene_sets/Census_allThu Dec 21 15_43_09 2017.tsv', sep='\t')['Gene Symbol'])

    # - Overlap samples
    samples = list(set(cnv).intersection(c_gdsc_fc).intersection(nexp).intersection(ploidy.index))
    print('[INFO] Samples overlap: {}'.format(len(samples)))

    # - Recurrent amplified cancer genes (following COSMIC definition)
    g_amp = recurrent_amplified_genes(cnv[samples], cnv_ratios[samples], ploidy[samples])
    g_amp = list(set(g_amp).intersection(cgenes))

    # - Recurrent non-expressed genes (Genes not expressed in at least 50% of the samples)
    g_nexp = recurrent_non_expressed_genes(nexp.reindex(c_gdsc_fc.index)[samples].dropna(), min_events=int(len(samples) * .5))

    # - Filter-out genes with low CRISPR var
    g_nexp_var = filter_low_var_essential(c_gdsc_fc.loc[g_nexp, samples])
    print('[INFO] Amplified genes: {}; Non-expressed genes: {}'.format(len(g_amp), len(g_nexp_var)))

    # - Linear regressions
    lm_res = lr_ess(cnv_ratios.loc[g_amp, samples].T, c_gdsc_fc.loc[g_nexp_var, samples].T, nexp)

    # - Define significant associations
    lm_res = lm_res.assign(signif=[int(p < 0.05 and b < -.5) for p, b in lm_res[['f_fdr', 'beta']].values])

    lm_res.query('f_fdr < 0.05').to_csv('data/crispy_df_collateral_essentialities.csv', index=False)
    print(lm_res.query('f_fdr < 0.05 & beta < -.5').sort_values(['overlap', 'f_fdr'], ascending=[False, True]).head(60))

    # - Boxplots
    crispr_gene, ratio_gene = association_boxplot(23517, lm_res, c_gdsc_fc, cnv, cnv_ratios, nexp)
    plt.gcf().set_size_inches(4, 3)
    plt.savefig('reports/crispy/collateral_essentiality_boxplot_{}_{}.png'.format(crispr_gene, ratio_gene), bbox_inches='tight', dpi=600)
    plt.close('all')

    # - Volcano
    plot_volcano(lm_res)
    plt.gcf().set_size_inches(2, 4)
    plt.savefig('reports/crispy/collateral_essentiality_volcano.png', bbox_inches='tight', dpi=600)
    plt.close('all')

    # - Chromosome plot
    sample, chrm = 'UACC-893', 'chr17'

    plot_df = pd.read_csv('data/crispy/gdsc/crispy_crispr_{}.csv'.format(sample), index_col=0)
    plot_df = plot_df[plot_df['fit_by'] == chrm]
    plot_df = plot_df.assign(pos=lib.groupby('gene')['pos'].mean().loc[plot_df.index])

    cytobands = import_cytobands(chrm=chrm)

    seg = pd.read_csv('data/gdsc/copynumber/snp6_bed/{}.snp6.picnic.bed'.format(sample), sep='\t')

    ax = plot_chromosome(
        plot_df['pos'], plot_df['fc'].rename('CRISPR FC'), plot_df['k_mean'].rename('Crispy'), seg=seg[seg['#chr'] == chrm],
        highlight=['NEUROD2', 'ERBB2', 'MED24'], cytobands=cytobands, legend=True
    )

    ax.set_xlabel('Position on chromosome {} (Mb)'.format(chrm))
    ax.set_ylabel('Fold-change')
    ax.set_title('{}'.format(sample))

    plt.gcf().set_size_inches(3, 1.5)
    plt.savefig('reports/crispy/collateral_essentiality_chromosomeplot.png', bbox_inches='tight', dpi=600)
    plt.close('all')
