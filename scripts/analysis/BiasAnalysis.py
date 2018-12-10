#!/usr/bin/env python
# Copyright (C) 2018 Emanuel Goncalves

import numpy as np
import crispy as cy
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from natsort import natsorted
from scipy.stats import ttest_ind
from analysis.DataImporter import DataImporter
from statsmodels.stats.multitest import multipletests


def copy_number_recall_curves(cn_curves):
    pal = list(reversed(cy.QCplot.get_palette_continuous(len(cn_curves))))

    for i, c in enumerate(reversed(order_copy_number)):
        plt.plot(cn_curves[c][0], cn_curves[c][1], label=f'{c}: AURC={cn_curves[c][2]:.2f}', lw=1., c=pal[i], alpha=.8)

    plt.plot((0, 1), (0, 1), 'k--', lw=.3, alpha=.5)

    plt.xlim(0, 1)
    plt.ylim(0, 1)

    plt.xlabel('Ranked genes by fold-change')
    plt.ylabel(f'Recall')

    plt.title('Copy-number effect on CRISPR-Cas9\n(non-expressed genes)')

    plt.legend(frameon=False, title='Copy-number', prop={'size': 6}, fontsize=6)


def copy_number_aurcs_per_sample(cn_aucs_df):
    ax = cy.QCplot.bias_boxplot(
        cn_aucs_df[~cn_aucs_df['copy_number'].isin(['0', '1'])], x='copy_number', notch=True, add_n=True, n_text_y=.05
    )

    ax.set_ylim(0, 1)
    ax.axhline(.5, lw=.3, c=cy.QCplot.PAL_DBGD[0], ls=':', zorder=0)

    plt.xlabel('Copy-number')
    plt.ylabel('Copy-number AURC (per cell line)')

    plt.title('Copy-number effect on CRISPR-Cas9\n(non-expressed genes)')


def copy_number_aucs_per_sample_ploidy(cn_aucs_df):
    order = natsorted(set(cn_aucs_df['ploidy_bin']))
    pal = dict(zip(*(order, cy.QCplot.get_palette_continuous(len(order)))))

    ax = cy.QCplot.bias_boxplot(
        cn_aucs_df[~cn_aucs_df['copy_number'].isin(['0', '1'])], x='copy_number', hue='ploidy_bin', notch=False, add_n=False, n_text_y=.05, palette=pal
    )

    ax.set_ylim(0, 1)
    ax.axhline(.5, lw=.3, c=cy.QCplot.PAL_DBGD[0], ls=':', zorder=0)

    plt.xlabel('Copy-number')
    plt.ylabel('Copy-number AURC (per cell line)')

    plt.title('Copy-number effect on CRISPR-Cas9\n(non-expressed genes)')

    plt.legend(frameon=False, title='Ploidy', prop={'size': 6}, fontsize=6)


def ratio_recall_curves(ratio_curves):
    pal = list(reversed(cy.QCplot.get_palette_continuous(len(ratio_curves))))

    for i, c in enumerate(reversed(order_ratio)):
        plt.plot(ratio_curves[c][0], ratio_curves[c][1], label=f'{c}: AURC={ratio_curves[c][2]:.2f}', lw=1., c=pal[i], alpha=.8)

    plt.plot((0, 1), (0, 1), 'k--', lw=.3, alpha=.5)

    plt.xlim(0, 1)
    plt.ylim(0, 1)

    plt.xlabel('Ranked genes by fold-change')
    plt.ylabel(f'Recall')

    plt.title('Copy-number effect on CRISPR-Cas9\n(non-expressed genes)')

    plt.legend(frameon=False, title='Copy-number ratio', prop={'size': 7}).get_title().set_fontsize('7')


def ratios_heatmap_bias(data, min_evetns=10):
    x, y, z = 'copy_number_bin', 'chr_copy_bin', 'scaled'

    plot_df = data.copy()

    counts = {(g, c) for g, c, v in plot_df.groupby([x, y])[z].count().reset_index().values if v <= min_evetns}

    plot_df = plot_df[[(g, c) not in counts for g, c in plot_df[[x, y]].values]]
    plot_df = plot_df.groupby([x, y])[z].mean().reset_index()
    plot_df = pd.pivot_table(plot_df, index=x, columns=y, values=z)

    g = sns.heatmap(
        plot_df.loc[natsorted(plot_df.index, reverse=False)].T, cmap='RdGy_r', annot=True, fmt='.2f', square=True,
        linewidths=.3, cbar=False, annot_kws={'fontsize': 7}, center=0
    )
    plt.setp(g.get_yticklabels(), rotation=0)

    plt.ylabel('Chromosome copy-number')
    plt.xlabel('Gene absolute copy-number')

    plt.title('Mean CRISPR-Cas9 fold-change\n(non-expressed genes)')


def ratios_heatmap_bias_count(data):
    x, y, z = 'copy_number_bin', 'chr_copy_bin', 'scaled'

    plot_df = data.groupby([x, y])[z].count().reset_index()
    plot_df = pd.pivot_table(plot_df, index=x, columns=y, values=z, fill_value=0)

    g = sns.heatmap(
        plot_df.loc[natsorted(plot_df.index, reverse=False)].T, cmap='Greys', annot=True, fmt='.0f', square=True,
        linewidths=.3, cbar=False, annot_kws={'fontsize': 7}
    )
    plt.setp(g.get_yticklabels(), rotation=0)

    plt.ylabel('Chromosome copy-number')
    plt.xlabel('Gene absolute copy-number')

    plt.title('Mean CRISPR-Cas9 fold-change\n(non-expressed genes)')


def ratios_histogram(plot_df):
    f, axs = plt.subplots(2, 1, sharex='all')

    c = cy.QCplot.PAL_DBGD[0]

    sns.distplot(plot_df['ratio'], color=c, kde=False, axlabel='', ax=axs[0], hist_kws=dict(linewidth=0))
    sns.distplot(plot_df['ratio'], color=c, kde=False, ax=axs[1], hist_kws=dict(linewidth=0))

    axs[0].axvline(1, lw=.3, ls='--', c=c)
    axs[1].axvline(1, lw=.3, ls='--', c=c)

    axs[0].set_ylim(500000, 1000000)  # outliers only
    axs[1].set_ylim(0, 400000)  # most of the data

    axs[0].spines['bottom'].set_visible(False)
    axs[1].spines['top'].set_visible(False)
    axs[0].xaxis.tick_top()
    axs[0].tick_params(labeltop='off')
    axs[1].xaxis.tick_bottom()

    d = .01

    kwargs = dict(transform=axs[0].transAxes, color='gray', clip_on=False, lw=.3)
    axs[0].plot((-d, +d), (-d, +d), **kwargs)
    axs[0].plot((1 - d, 1 + d), (-d, +d), **kwargs)

    kwargs.update(transform=axs[1].transAxes)
    axs[1].plot((-d, +d), (1 - d, 1 + d), **kwargs)
    axs[1].plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)

    axs[1].set_xlabel('Copy-number ratio (Gene / Chromosome)')

    f.add_subplot(111, frameon=False)
    plt.tick_params(labelcolor='none', top='off', bottom='off', left='off', right='off')
    plt.grid(False)


def copy_number_aucs_per_sample_chrcopy(cn_aucs_chr_df):
    ax = cy.QCplot.bias_boxplot(
        cn_aucs_chr_df[~cn_aucs_chr_df['copy_number'].isin(['0', '1'])], x='copy_number', hue='chr_copy_bin', notch=False, add_n=True, n_text_y=.05
    )

    ax.set_ylim(0, 1)
    ax.axhline(.5, lw=.3, c=cy.QCplot.PAL_DBGD[0], ls=':', zorder=0)

    plt.xlabel('Copy-number')
    plt.ylabel('Copy-number AURC (per cell line)')

    plt.title('Copy-number effect on CRISPR-Cas9\n(non-expressed genes)')

    plt.legend(title='Chr copies', loc='center left', bbox_to_anchor=(1, 0.5), prop={'size': 6}, fontsize=6, frameon=False)


if __name__ == '__main__':
    # - Imports
    data = DataImporter()

    # Non-expressed genes
    nexp = data.get_rnaseq_nexp(filter_essential=True)

    # Crispy bed files
    beds = data.get_crispy_beds()

    # - Aggregate by gene
    agg_fun = dict(
        fold_change=np.mean, copy_number=np.mean, ploidy=np.mean, chr_copy=np.mean, ratio=np.mean, chr='first',
        corrected=np.mean
    )

    bed_nexp_gene = []
    for s in beds:
        df = beds[s].groupby('gene').agg(agg_fun)
        df = df.assign(scaled=cy.Crispy.scale_crispr(df['fold_change']))
        df = df.reindex(nexp[nexp[s] == 1].index).dropna()
        df = df.assign(sample=s)

        bed_nexp_gene.append(df)

    bed_nexp_gene = pd.concat(bed_nexp_gene).reset_index()

    bed_nexp_gene = bed_nexp_gene.assign(copy_number_bin=bed_nexp_gene['copy_number'].apply(cy.Utils.bin_cnv, args=(10,)))
    bed_nexp_gene = bed_nexp_gene.assign(ratio_bin=bed_nexp_gene['ratio'].apply(cy.Utils.bin_cnv, args=(4,)))
    bed_nexp_gene = bed_nexp_gene.assign(chr_copy_bin=bed_nexp_gene['chr_copy'].apply(cy.Utils.bin_cnv, args=(6,)))

    # Get ploidy and chromosome copies
    ploidy = pd.Series({s: beds[s]['ploidy'].mean() for s in beds})
    chr_copy = pd.DataFrame({s: beds[s].groupby('chr')['chr_copy'].mean() for s in beds}).unstack()

    # - Copy-number/Ratio bias
    # Copy-number recall curves
    order_copy_number = natsorted(set(bed_nexp_gene['copy_number_bin']))[2:]

    cn_curves = {
        c: cy.QCplot.recall_curve(bed_nexp_gene['scaled'], bed_nexp_gene.query('copy_number_bin == @c').index) for c in order_copy_number
    }

    # Copy-number AURCs per sample
    cn_aucs_df = pd.concat([cy.QCplot.recall_curve_discretise(
        bed_nexp_gene.query('sample == @s')['scaled'],
        bed_nexp_gene.query('sample == @s')['copy_number'],
        10
    ).assign(sample=s) for s in beds])

    cn_aucs_df = cn_aucs_df.assign(ploidy=ploidy[cn_aucs_df['sample']].values)
    cn_aucs_df = cn_aucs_df.assign(ploidy_bin=cn_aucs_df['ploidy'].apply(cy.Utils.bin_cnv, args=(5,)))

    # LOF comparison between cell lines groups
    lof_tests = []
    for c in natsorted(set(cn_aucs_df['copy_number'])):
        if c not in ['0', '1']:
            diploid = cn_aucs_df.query(f"(copy_number == '{c}') & (ploidy_bin == '2')")
            tetraploid = cn_aucs_df.query(f"(copy_number == '{c}') & ((ploidy_bin == '4') | (ploidy_bin == '5+'))")

            t, p = ttest_ind(diploid['auc'], tetraploid['auc'], equal_var=False)

            lof_tests.append(dict(t=t, p=p, c=c, n_diploid=diploid.shape[0], n_tetraploid=tetraploid.shape[0]))

            print(f'{c}, {t}, {p:.1e}, {diploid.shape}, {tetraploid.shape}')
    lof_tests = pd.DataFrame(lof_tests).dropna()
    lof_tests['fdr'] = multipletests(lof_tests['p'], method='fdr_bh')[1]
    lof_tests.to_csv('data/analysis/lof_comparison_ploidy_groups.csv', index=False)

    # Ratio recall curves
    order_ratio = natsorted(set(bed_nexp_gene['ratio_bin']))

    ratio_curves = {
        c: cy.QCplot.recall_curve(bed_nexp_gene['scaled'], bed_nexp_gene.query('ratio_bin == @c').index) for c in order_ratio
    }

    # Copy-number AURCs per sample
    chrms = natsorted(set(bed_nexp_gene['chr']))

    cn_aucs_chr_df = pd.concat([cy.QCplot.recall_curve_discretise(
        bed_nexp_gene.query('sample == @s').query('chr == @c')['scaled'],
        bed_nexp_gene.query('sample == @s').query('chr == @c')['copy_number'],
        10
    ).assign(sample=s).assign(chr=c) for s in beds for c in chrms])

    cn_aucs_chr_df = cn_aucs_chr_df.assign(chr_copy=[chr_copy.loc[(s, c)] for s, c in cn_aucs_chr_df[['sample', 'chr']].values])
    cn_aucs_chr_df = cn_aucs_chr_df.assign(chr_copy_bin=cn_aucs_chr_df['chr_copy'].apply(cy.Utils.bin_cnv, args=(6,)))

    # - Plot
    # Copy-number recall curves
    copy_number_recall_curves(cn_curves)
    plt.gcf().set_size_inches(3, 3)
    plt.savefig('reports/analysis/bias_copynumber_curves.pdf', bbox_inches='tight', transparent=True)
    plt.close('all')

    # Copy-number AURCs per sample
    copy_number_aurcs_per_sample(cn_aucs_df)
    plt.gcf().set_size_inches(3, 3)
    plt.savefig('reports/analysis/bias_copynumber_boxplots.pdf', bbox_inches='tight', transparent=True)
    plt.close('all')

    # Plot bias per cell line group by ploidy
    copy_number_aucs_per_sample_ploidy(cn_aucs_df)
    plt.gcf().set_size_inches(3, 3)
    plt.savefig('reports/analysis/bias_copynumber_boxplots_by_ploidy.pdf', bbox_inches='tight', transparent=True)
    plt.close('all')

    # Plot bias per cell line group by chromosome copy
    copy_number_aucs_per_sample_chrcopy(cn_aucs_chr_df)
    plt.gcf().set_size_inches(3, 3)
    plt.savefig('reports/analysis/bias_copynumber_boxplots_by_chrm_copy.pdf', bbox_inches='tight', transparent=True)
    plt.close('all')

    # Ratios histogram
    ratios_histogram(bed_nexp_gene)
    plt.title('Copy-number ratio histogram (non-expressed genes)')
    plt.gcf().set_size_inches(4, 3)
    plt.savefig('reports/analysis/bias_ratio_histogram.pdf', bbox_inches='tight', transparent=True)
    plt.close('all')

    # Ratio recall curves
    ratio_recall_curves(ratio_curves)
    plt.gcf().set_size_inches(3, 3)
    plt.savefig('reports/analysis/bias_ratio_curves.pdf', bbox_inches='tight', transparent=True)
    plt.close('all')

    # Ratio heatmap
    ratios_heatmap_bias(bed_nexp_gene)
    plt.gcf().set_size_inches(5.5, 5.5)
    plt.savefig('reports/analysis/bias_ratio_heatmap.pdf', bbox_inches='tight', transparent=True)
    plt.close('all')

    # Ratio heatmap count
    ratios_heatmap_bias_count(bed_nexp_gene)
    plt.gcf().set_size_inches(5.5, 5.5)
    plt.savefig('reports/analysis/bias_ratio_heatmap_count.pdf', bbox_inches='tight', transparent=True)
    plt.close('all')

    # Ratio fold-change boxplots
    ax = cy.QCplot.bias_boxplot(bed_nexp_gene, x='ratio_bin', y='scaled', tick_base=.5, add_n=True)

    ax.axhline(0, ls='-', lw=.3, c=cy.QCplot.PAL_DBGD[2], zorder=0)
    ax.axhline(-1, ls=':', lw=.3, c=cy.QCplot.PAL_DBGD[2], zorder=0)

    ax.set_xlabel('Copy-number ratio')
    ax.set_ylabel('CRISPR-Cas9 fold-change (scaled)')
    ax.set_title('Copy-number ratio effect\n(non-expressed genes)')

    plt.gcf().set_size_inches(2, 3)
    plt.savefig('reports/analysis/bias_ratio_boxplot.png', bbox_inches='tight', transparent=True, dpi=600)
    plt.close('all')

    # Representative example
    gene = 'NEUROD2'

    plot_df = bed_nexp_gene[bed_nexp_gene['index'] == gene].sort_values('copy_number')
    plot_df['Cancer Type'] = data.samplesheet.loc[plot_df['sample'], 'Cancer Type'].values

    f, (ax1, ax2) = plt.subplots(1, 2, sharex='none', sharey='row')

    cy.QCplot.bias_boxplot(plot_df, 'copy_number_bin', 'scaled', ax=ax1, tick_base=.5, notch=False, add_n=True)
    ax1.axhline(0, ls='-', color='black', lw=.1, zorder=0, alpha=.5)
    ax1.set_xlabel('Copy-number')
    ax1.set_ylabel('CRISPR-Cas9 fold-change')

    cy.QCplot.bias_boxplot(plot_df, 'ratio_bin', 'scaled', ax=ax2, tick_base=.5, notch=False, add_n=True)
    ax2.axhline(0, ls='-', color='black', lw=.1, zorder=0, alpha=.5)
    ax2.set_xlabel('Copy-number ratio')
    ax2.set_ylabel('')

    plt.suptitle(gene, fontsize=10, y=1.02)

    plt.gcf().set_size_inches(4, 2)
    plt.savefig(f'reports/analysis/bias_ratio_example.pdf', bbox_inches='tight', transparent=True)
    plt.close('all')
