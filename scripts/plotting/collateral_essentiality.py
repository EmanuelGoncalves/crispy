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


def recurrent_amplified_genes(cnv, cnv_ratios, ploidy, min_events=3):
    g_amp = pd.DataFrame({s: (cnv[s] > 8 if ploidy[s] > 2.7 else cnv[s] > 4).astype(int) for s in cnv})
    g_amp = g_amp.loc[g_amp.index.isin(cnv_ratios.index)]
    g_amp = g_amp[g_amp.sum(1) >= min_events]
    return list(g_amp.index)


def recurrent_high_copy_number_ratio(cnv_ratios, thres=2, min_events=3):
    g_amp = (cnv_ratios > thres).sum(1)
    g_amp = g_amp[g_amp > min_events]
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

    return y_feat, x_feat, plot_df


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

    # CRISPR lib
    lib = pd.read_csv(mp.LIBRARY, index_col=0)
    lib = lib.assign(pos=lib[['start', 'end']].mean(1).values)
    gene_chrm = lib.groupby('gene')['chr'].first()

    # GTex TPM per tissue
    gtex = pd.read_csv('data/GTEx_Analysis_2016-01-15_v7_RNASeQCv1.1.8_gene_median_tpm.gct', sep='\t', index_col=1)

    # - Overlap samples
    samples = list(set(cnv).intersection(c_gdsc_fc).intersection(nexp).intersection(ploidy.index))
    print('[INFO] Samples overlap: {}'.format(len(samples)))

    # - Define gene-sets
    # Recurrent amplified cancer genes
    g_amp = recurrent_high_copy_number_ratio(cnv_ratios[samples], min_events=3, thres=2)

    # Recurrent non-expressed genes (Genes not expressed in at least 50% of the samples)
    g_nexp = recurrent_non_expressed_genes(nexp.reindex(c_gdsc_fc.index)[samples].dropna(), min_events=int(len(samples) * .25))

    # Filter-out genes with low CRISPR var
    g_nexp_var = filter_low_var_essential(c_gdsc_fc.loc[g_nexp, samples])
    print('[INFO] Amplified genes: {}; Non-expressed genes: {}'.format(len(g_amp), len(g_nexp_var)))

    # - Linear regressions
    lm = lr_ess(cnv_ratios.loc[g_amp, samples].T, c_gdsc_fc.loc[g_nexp_var, samples].T, nexp)

    # - Define significant associations
    lm_res = lm.assign(signif=[int(p < 0.05 and b < -.5) for p, b in lm[['f_fdr', 'beta']].values])
    lm_res\
        .query('f_fdr < 0.05 & beta < -.5')\
        .sort_values(['overlap', 'f_fdr'], ascending=[False, True])\
        .to_csv('data/crispy_df_collateral_essentialities_interactions.csv', index=False)
    # lm_res = pd.read_csv('data/crispy_df_collateral_essentialities_interactions.csv')
    print(lm_res.query('f_fdr < 0.05 & beta < -.5').sort_values(['overlap', 'f_fdr'], ascending=[False, True]))

    # - Volcano
    plot_volcano(lm_res)
    plt.gcf().set_size_inches(2, 4)
    plt.savefig('reports/crispy/collateral_essentiality_volcano.png', bbox_inches='tight', dpi=600)
    plt.close('all')

    # - Boxplots
    crispr_gene, ratio_gene, plot_df = association_boxplot(149, lm_res, c_gdsc_fc, cnv, cnv_ratios, nexp)
    plt.gcf().set_size_inches(4, 3)
    plt.savefig('reports/crispy/collateral_essentiality_boxplot_{}_{}.png'.format(crispr_gene, ratio_gene), bbox_inches='tight', dpi=600)
    plt.close('all')

    # - Chromosome plot
    sample, chrm = plot_df.sort_values(['ratio_bin', 'crispy'], ascending=[False, True]).index[0], gene_chrm.loc[crispr_gene]

    plot_df = pd.read_csv('data/crispy/gdsc/crispy_crispr_{}.csv'.format(sample), index_col=0)
    plot_df = plot_df[plot_df['fit_by'] == chrm]
    plot_df = plot_df.assign(pos=lib.groupby('gene')['pos'].mean().loc[plot_df.index])

    cytobands = import_cytobands(chrm=chrm)

    seg = pd.read_csv('data/gdsc/copynumber/snp6_bed/{}.snp6.picnic.bed'.format(sample), sep='\t')

    ax = plot_chromosome(
        plot_df['pos'], plot_df['fc'].rename('CRISPR FC'), plot_df['k_mean'].rename('Crispy'), seg=seg[seg['#chr'] == chrm],
        highlight=[crispr_gene, ratio_gene, 'MED24'], cytobands=cytobands, legend=True
    )

    ax.set_xlabel('Position on chromosome {} (Mb)'.format(chrm))
    ax.set_ylabel('Fold-change')
    ax.set_title('{}'.format(sample))

    # plt.xlim(30, 45)

    plt.gcf().set_size_inches(3, 1.5)
    plt.savefig('reports/crispy/collateral_essentiality_chromosomeplot_{}_{}.png'.format(sample, chrm), bbox_inches='tight', dpi=600)
    plt.close('all')

    # - List of collateral essentialities per cell line
    def set_keep_order(gene_dup_list):
        gene_unique_list = []
        for g in gene_dup_list:
            if g not in gene_unique_list:
                gene_unique_list.append(g)
        return gene_unique_list

    lm_res_filtered = lm_res.query('f_fdr < 0.05 & beta < -.5')

    coless = []
    for g_crispr in set_keep_order(lm_res_filtered['gene_crispr']):
        ess_samples = (c_gdsc_fc.loc[g_crispr, samples] < c_gdsc_fc.reindex(essential)[samples].mean()).astype(int)

        for s in ess_samples[ess_samples == 1].index:
            s_tissue = ss.loc[s, 'Tissue']
            is_tissue_expressed = nexp.loc[g_crispr].reindex(ss[ss['Tissue'] == s_tissue].index, axis=1).dropna()
            perc_tissue_expressed = round((is_tissue_expressed == 1).sum() / len(is_tissue_expressed) * 100, 2)

            if s in nexp.loc[g_crispr] and perc_tissue_expressed > 50.:
                coless.append({
                    'Gene_CRISPR': g_crispr,
                    'Cell_line': s,
                    'Cancer Type': ss.loc[s, 'Cancer Type'],
                    'Tissue': ss.loc[s, 'Tissue'],
                    'nexp': perc_tissue_expressed,
                    'Gene_Amplified': ';'.join(lm_res_filtered[lm_res_filtered['gene_crispr'] == g_crispr]['gene_ratios'])
                })
    coless = pd.DataFrame(coless)[['Gene_CRISPR', 'Cell_line', 'Cancer Type', 'Tissue', 'nexp', 'Gene_Amplified']]
    coless.to_csv('data/crispy_df_collateral_essentialities.csv', index=False)
    # coless = pd.read_csv('data/crispy_df_collateral_essentialities.csv')

    print(
        'Collateral essentialies:\n\tUnique genes: {};\n\tUnique cell lines: {}\n\tUnique cancer type: {}'
            .format(len(set(coless['Gene_CRISPR'])), len(set(coless['Cell_line'])), len(set(coless['Cancer Type'])))
    )

    # - Plot number of collateral essentialities per Cancer Type
    plot_df = pd.concat([
        ss.loc[samples]['Cancer Type'].value_counts().rename('Total'),
        ss.loc[set(coless['Cell_line'])]['Cancer Type'].value_counts().rename('Count')
    ], axis=1).replace(np.nan, 0).astype(int).sort_values('Total')
    plot_df = plot_df.reset_index().assign(ypos=np.arange(plot_df.shape[0]))

    plt.barh(plot_df['ypos'], plot_df['Total'], .8, color=bipal_dbgd[0], align='center', label='All')
    plt.barh(plot_df['ypos'], plot_df['Count'], .8, color=bipal_dbgd[1], align='center', label='w/ Collateral Essentiality')

    for xx, yy in plot_df[['ypos', 'Count']].values:
        if yy > 0:
         plt.text(yy - .05, xx, '{}'.format(yy), color='white', ha='right', va='center', fontsize=6)

    plt.yticks(plot_df['ypos'])
    plt.yticks(plot_df['ypos'], plot_df['index'])

    # plt.xticks(np.arange(0, 41, 10))
    plt.grid(True, color=bipal_dbgd[0], linestyle='-', linewidth=.1, axis='x')

    plt.xlabel('Number of collateral vulnerable genes')
    plt.title('Histogram of collateral vulnerabilities')

    plt.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), prop={'size': 6})

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
