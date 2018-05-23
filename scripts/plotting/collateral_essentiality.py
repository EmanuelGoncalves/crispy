#!/usr/bin/env python
# Copyright (C) 2018 Emanuel Goncalves

import numpy as np
import pandas as pd
import scripts as mp
import seaborn as sns
import matplotlib.pyplot as plt
from crispy import bipal_dbgd
from natsort import natsorted
from crispy.utils import bin_cnv
from crispy.regression.linear import lr
from statsmodels.stats.multitest import multipletests
from crispy.benchmark_plot import plot_chromosome, import_cytobands


def recurrent_high_ratio(cnv_ratios, thres=2, min_events=3):
    g_amp = (cnv_ratios > thres).astype(int)

    g_amp = g_amp[g_amp.sum(1) >= min_events]

    return g_amp


def recurrent_nexp_essential(nexp, c_gdsc_fc, essential, min_events=3, max_events=100, essential_thres=0.75):
    g_ess = (c_gdsc_fc < (c_gdsc_fc.reindex(essential).mean() * essential_thres)).astype(int)

    g_nexp_ess = nexp * g_ess

    g_nexp_ess = g_nexp_ess[g_nexp_ess.sum(1) >= min_events]

    g_nexp_ess = g_nexp_ess[g_nexp_ess.sum(1) <= max_events]

    return g_nexp_ess


def lr_ess(xs, ys):
    # Run fit
    lm = lr(xs, ys)

    # Export lm results
    lm_res = pd.concat([lm[i].unstack().rename(i) for i in lm], axis=1)
    lm_res = lm_res.reset_index().rename(columns={'level_0': 'gene_crispr', 'level_1': 'gene_ratios'})

    return lm_res


def set_keep_order(gene_dup_list):
    gene_unique_list = []

    for g in gene_dup_list:
        if g not in gene_unique_list:
            gene_unique_list.append(g)

    return gene_unique_list


def association_boxplot(plot_df, gr, gc, cnv_bin_thres=10, ratio_bin_thres=4):
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
    sns.stripplot('cnv_bin', 'crispy', 'Expressed', data=plot_df, order=cnvorder, palette=pal, edgecolor='white', linewidth=.3, size=3.5, jitter=.1, alpha=.9, ax=ax1)

    ax1.axhline(0, ls='-', lw=.1, c=color)

    ax1.set_ylabel('{}\nCRISPR-Cas9 fold-change (log2)'.format(gc))
    ax1.set_xlabel('{}\nCopy-number'.format(gr))

    # Ratio boxplot
    order = natsorted(set(plot_df['ratio_bin']))

    sns.boxplot('ratio_bin', 'crispy', data=plot_df, order=order, color=color, sym='', linewidth=.3, saturation=.1, ax=ax2)
    sns.stripplot('ratio_bin', 'crispy', 'Expressed', data=plot_df, order=order, palette=pal, edgecolor='white', linewidth=.3, size=3.5, jitter=.1, alpha=.8, ax=ax2)

    ax2.axhline(0, ls='-', lw=.1, c=color)

    ax2.set_xlabel('{}\nCopy-number ratio'.format(gr))
    ax2.set_ylabel('')

    # Labels
    plt.suptitle('Collateral essentiality {} - {}'.format(gc, gr), y=1)

    return plot_df


def chromosome_plot_sample(sample, chrm, xlim=None):
    plot_df = pd.read_csv('data/crispy/gdsc/crispy_crispr_{}.csv'.format(sample), index_col=0)
    plot_df = plot_df[plot_df['fit_by'] == chrm]
    plot_df = plot_df.assign(pos=lib.groupby('gene')['pos'].mean().loc[plot_df.index])

    cytobands = import_cytobands(chrm=chrm)

    seg = pd.read_csv('data/gdsc/copynumber/snp6_bed/{}.snp6.picnic.bed'.format(sample), sep='\t')

    ax = plot_chromosome(
        plot_df['pos'], plot_df['fc'].rename('CRISPR-Cas9'), plot_df['k_mean'].rename('Fitted mean'), seg=seg[seg['#chr'] == chrm],
        highlight=[gc, gr], cytobands=cytobands, legend=True, legend_size=7
    )

    if xlim is not None:
        ax.set_xlim(xlim)

    ax.set_xlabel('Position on chromosome {} (Mb)'.format(chrm))
    ax.set_ylabel('Fold-change')
    ax.set_title('{}'.format(sample))


def plot_collateral_essentiality(gc, gr, xlim=None, c_size=(1, 2.5)):
    plot_df = pd.concat([
        c_gdsc_fc.loc[gc, samples].rename('crispy'),
        cnv.loc[gr, samples].rename('cnv'),
        cnv_ratios.loc[gr, samples].rename('ratio'),
        nexp.loc[gc].replace({1.0: 'No', 0.0: 'Yes'}).rename('Expressed')
    ], axis=1).dropna().sort_values('crispy')

    # Boxplot
    plot_df = association_boxplot(plot_df, gr, gc)
    plt.gcf().set_size_inches(4, 3)
    plt.savefig('reports/crispy/collateral_essentiality_boxplot_{}_{}.png'.format(gr, gc), bbox_inches='tight', dpi=600)
    plt.close('all')

    # Chromosome plot
    sample, chrm = plot_df.sort_values(['ratio_bin', 'crispy'], ascending=[False, True]).index[0], gene_chrm.loc[gc]

    chromosome_plot_sample(sample, chrm, xlim=xlim)
    plt.gcf().set_size_inches(c_size[0], c_size[1])
    plt.savefig('reports/crispy/collateral_essentiality_chromosomeplot_{}_{}.png'.format(sample, chrm), bbox_inches='tight', dpi=600)
    plt.close('all')


def plot_overall_collateral_essentialities():
    plot_df = pd.concat([
        ss.loc[samples]['Cancer Type'].value_counts().rename('Total'),
        ss.loc[set(coless['Cell_line'])]['Cancer Type'].value_counts().rename('Count')
    ], axis=1).replace(np.nan, 0).astype(int).sort_values('Total')

    plot_df = plot_df.reset_index().assign(ypos=np.arange(plot_df.shape[0]))

    plt.barh(plot_df['ypos'], plot_df['Total'], .8, color=bipal_dbgd[0], align='center')
    plt.barh(plot_df['ypos'], plot_df['Count'], .8, color=bipal_dbgd[1], align='center', label='Cell lines with\ncollateral essentiality')

    for xx, yy in plot_df[['ypos', 'Count']].values:
        if yy > 0:
         plt.text(yy + .08, xx - 0.05, '{}'.format(yy), color=bipal_dbgd[1], ha='left', va='center', fontsize=7)

    plt.yticks(plot_df['ypos'])
    plt.yticks(plot_df['ypos'], plot_df['index'])

    # plt.xticks(np.arange(0, 41, 10))
    plt.grid(True, color=bipal_dbgd[0], linestyle='-', linewidth=.1, axis='x')

    plt.xlabel('Number of cell lines')
    plt.title('Histogram of collateral vulnerabilities')

    plt.legend(loc='center left', bbox_to_anchor=(1.02, 0.0), prop={'size': 8})

    plt.gcf().set_size_inches(2, 3)
    plt.savefig('reports/crispy/collateral_essentiality_type_count.png', bbox_inches='tight', dpi=600)
    plt.close('all')

    # -
    labels = '', ''
    sizes = [len(samples) - len(set(coless['Cell_line'])), len(set(coless['Cell_line']))]
    explode = (0, 0.1)

    fig1, ax1 = plt.subplots()

    ax1.pie(
        sizes, explode=explode, labels=labels, autopct='%1.1f%%', shadow=False, startangle=90, colors=[bipal_dbgd[0], bipal_dbgd[1]],
        textprops={'color': 'white', 'fontsize': 14}
    )
    ax1.axis('equal')

    plt.gcf().set_size_inches(2, 2)
    plt.savefig('reports/crispy/collateral_essentiality_count_piechart.png', bbox_inches='tight', dpi=600)
    plt.close('all')


if __name__ == '__main__':
    # - Import
    # Samplesheet
    ss = pd.read_csv(mp.SAMPLESHEET, index_col=0)

    # Essential
    essential = pd.read_csv(mp.HART_ESSENTIAL, sep='\t')['gene'].rename('essential')

    # Non-expressed genes
    nexp = pd.read_csv(mp.NON_EXP, index_col=0)

    # Copy-number
    cnv = pd.read_csv(mp.CN_GENE.format('snp'), index_col=0).dropna()

    # Copy-number ratios
    cnv_ratios = pd.read_csv(mp.CN_GENE_RATIO.format('snp'), index_col=0).dropna()

    # Ploidy
    ploidy = pd.read_csv(mp.CN_PLOIDY.format('snp'), index_col=0, names=['sample', 'ploidy'])['ploidy']

    # CRISPR
    c_gdsc_fc = pd.read_csv(mp.CRISPR_GENE_FC, index_col=0)

    # COSMIC cancer genes
    cgenes = pd.read_csv('data/cancer_gene_census.csv')

    # - CRISPR lib
    lib = pd.read_csv(mp.LIBRARY, index_col=0)
    lib = lib.assign(pos=lib[['start', 'end']].mean(1).values)

    gene_chrm = lib.groupby('gene')['chr'].first()
    gene_coord = lib.groupby('gene')['start', 'end'].first().mean(1)

    # - Overlap samples
    samples = list(set(cnv).intersection(c_gdsc_fc).intersection(nexp).intersection(ploidy.index))
    genes = list(set(cnv.index).intersection(c_gdsc_fc.index).intersection(nexp.index))
    print('[INFO] Samples overlap: {}, Gene: {}'.format(len(samples), len(genes)))

    # - Define gene-sets
    # Recurrent amplified cancer genes
    g_amp = recurrent_high_ratio(cnv_ratios.loc[genes, samples], thres=2, min_events=1)

    # Recurrent non-expressed essential genes
    g_nexp_ess = recurrent_nexp_essential(
        nexp.loc[genes, samples], c_gdsc_fc.loc[genes, samples], essential, essential_thres=.75, min_events=1, max_events=int(.15*len(samples))
    )
    print('[INFO] Amplified genes: {}; Non-expressed essential genes: {}'.format(len(g_amp), len(g_nexp_ess)))

    # - Colateral essentialities pairs
    gce = g_amp[samples].dot(g_nexp_ess[samples].T).unstack().sort_values(ascending=False)

    # Min events
    gce = gce[gce >= 1]

    # Discard associations with same gene
    gce = gce[[gc != gr for gc, gr in gce.index]]

    # Discard gene associations that are not in the same chromosome and genomically close
    gce = gce[[gene_chrm.loc[gc] == gene_chrm.loc[gr] for gc, gr in gce.index]]
    gce = gce[[abs(gene_coord.loc[gc] - gene_coord.loc[gr]) < 1e6 for gc, gr in gce.index]]

    gce = gce.reset_index().rename(columns={'level_0': 'gc', 'level_1': 'gr', 0: 'count'})

    # - List of collateral essentialities per cell line
    coless = []
    for gc in set_keep_order(gce['gc']):
        grs = gce[gce['gc'] == gc]['gr']

        coless_samples = g_nexp_ess.loc[gc, samples] * g_amp.loc[grs, samples].max()

        for s in coless_samples[coless_samples == 1].index:
            s_tissue = ss.loc[s, 'Tissue']

            is_tissue_expressed = nexp.loc[gc].reindex(ss[ss['Tissue'] == s_tissue].index, axis=1).dropna()
            perc_tissue_expressed = round((is_tissue_expressed == 1).sum() / len(is_tissue_expressed) * 100, 2)

            if perc_tissue_expressed > 50.:
                coless.append({
                    'Gene_CRISPR': gc,
                    'Cell_line': s,
                    'Cancer Type': ss.loc[s, 'Cancer Type'],
                    'Tissue': ss.loc[s, 'Tissue'],
                    'Tissue non-expression (%)': perc_tissue_expressed,
                    'Gene_Amplified': ';'.join(grs),
                    'Chr': gene_chrm.loc[grs].iloc[0]
                })

    coless = pd.DataFrame(coless)
    coless = coless.assign(Cancer_Gene=[len(set(i.split(';')).intersection(cgenes['Gene Symbol'])) > 0 for i in coless['Gene_Amplified']])
    coless = coless[['Gene_CRISPR', 'Cell_line', 'Cancer Type', 'Tissue', 'Tissue non-expression (%)', 'Chr', 'Cancer_Gene', 'Gene_Amplified']]
    coless.to_csv('data/crispy_df_collateral_essentialities.csv', index=False)

    print(
        'Collateral essentialies:\n\tUnique genes: {};\n\tUnique cell lines: {}\n\tUnique cancer type: {}'
            .format(len(set(coless['Gene_CRISPR'])), len(set(coless['Cell_line'])), len(set(coless['Cancer Type'])))
    )

    # - Plot collateral essentiality
    # gc, gr, xlim = 'NEUROD2', 'ERBB2', (32, 45)
    gc, gr, xlim = 'MRGPRD', 'CCND1', (60, 80)
    plot_collateral_essentiality(gc, gr, xlim=xlim, c_size=(2, 3))

    # - Plot number of collateral essentialities per Cancer Type
    plot_overall_collateral_essentialities()
