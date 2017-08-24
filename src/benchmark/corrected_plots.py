#!/usr/bin/env python
# Copyright (C) 2017 Emanuel Goncalves

import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime as dt
from scipy.stats.stats import rankdata
from sklearn.metrics.ranking import auc
from matplotlib.gridspec import GridSpec


# - Imports
# Essential genes
essential = list(pd.read_csv('data/resources/curated_BAGEL_essential.csv', sep='\t')['gene'])

# Chromosome cytobands
cytobands = pd.read_csv('data/resources/cytoBand.txt', sep='\t')
print('[%s] Misc files imported' % dt.now().strftime('%Y-%m-%d %H:%M:%S'))

# CRISPR
crispr = pd.read_csv('data/gdsc/crispr/deseq_gene_fold_changes.csv', index_col=0)

# CIRSPR segments
crispr_seg = pd.read_csv('data/gdsc/crispr/segments_cbs_deseq2.csv')
print('[%s] CRISPR data imported' % dt.now().strftime('%Y-%m-%d %H:%M:%S'))

# Copy-number absolute counts
cnv = pd.read_csv('data/gdsc/copynumber/Gene_level_CN.txt', sep='\t', index_col=0)
cnv = cnv.loc[{i.split('_')[0] for i in crispr.index}, crispr.columns].dropna(how='all').dropna(how='all', axis=1)
cnv_abs = cnv.applymap(lambda v: int(v.split(',')[0]))
cnv_loh = cnv.applymap(lambda v: v.split(',')[2])

# Copy-number segments
cnv_seg = pd.read_csv('data/gdsc/copynumber/Summary_segmentation_data_994_lines_picnic.txt', sep='\t')
cnv_seg['chr'] = cnv_seg['chr'].replace(23, 'X').replace(24, 'Y').astype(str)
print('[%s] Copy-number data imported' % dt.now().strftime('%Y-%m-%d %H:%M:%S'))

# CrisprCleanR corrected
cc_crispr = pd.read_csv('data/gdsc/crispr/all_GLCFC.csv', index_col=0)


# - Plot
sample_files = [os.path.join('data/crispy/', f) for f in os.listdir('data/crispy/')]

# sample = 'AU565'
# sample = 'HT-29'
for sample in ['AU565', 'HT-29', 'SW48']:
    # - Import sample normalisation output
    # Optimisation params
    df_params = pd.DataFrame({'%s_chr%s' % (sample, f.split('_')[1]): pd.Series.from_csv(f) for f in sample_files if f.startswith('data/crispy/%s_' % sample) and f.endswith('_optimised_params.csv')}).T
    print(df_params)

    # sgRNA level fc
    df = pd.concat([pd.read_csv(f, index_col=0) for f in sample_files if f.startswith('data/crispy/%s_' % sample) and f.endswith('_corrected.csv')]).sort_values('STARTpos')
    print('sgRNAs: ', df.shape)

    # Gene level fc
    f = {'logfc': np.mean, 'cnv': 'first', 'loh': 'first', 'essential': 'first', 'logfc_mean': np.mean, 'logfc_norm': np.mean}
    df_genes = df[['GENES', 'logfc', 'cnv', 'loh', 'essential', 'logfc_mean', 'logfc_norm']].groupby('GENES').agg(f)
    df_genes['ccleanr'] = cc_crispr.loc[df_genes.index, sample]
    print('Genes: ', df_genes.shape)

    # - Evaluate
    plot_df = df_genes.dropna(subset=['logfc', 'logfc_norm', 'ccleanr'])

    sns.set(style='ticks', context='paper', font_scale=0.75, palette='PuBu_r', rc={'axes.linewidth': .3, 'xtick.major.width': .3, 'ytick.major.width': .3, 'xtick.major.size': 2.5, 'ytick.major.size': 2.5, 'xtick.direction': 'in', 'ytick.direction': 'in'})
    for c, f in zip(sns.color_palette('viridis', 3).as_hex(), ['logfc', 'ccleanr', 'logfc_norm']):
        # Build data-frame
        plot_df_f = plot_df.sort_values(f).copy()

        # Rank fold-changes
        x = rankdata(plot_df_f[f]) / plot_df_f.shape[0]

        # Observed cumsum
        y = plot_df_f['essential'].cumsum() / plot_df_f['essential'].sum()

        # Plot
        plt.plot(x, y, label='%s: %.2f' % (f, auc(x, y)), lw=1., c=c)

        # Random
        plt.plot((0, 1), (0, 1), 'k--', lw=.3, alpha=.5)

    plt.xlim(0, 1)
    plt.ylim(0, 1)

    plt.title('%s - Essential genes' % sample)
    plt.xlabel('Ranked genes')
    plt.ylabel('Cumulative sum essential')

    plt.legend(loc=4)

    plt.gcf().set_size_inches(1.5, 1.5)
    plt.savefig('reports/%s_eval_plot.png' % sample, bbox_inches='tight', dpi=600)
    plt.close('all')
    print('[%s] AROCs done.' % dt.now().strftime('%Y-%m-%d %H:%M:%S'))

    # - Boxplot
    plot_df = df_genes.dropna(subset=['cnv', 'logfc', 'logfc_norm', 'ccleanr'])
    plot_df = plot_df[plot_df['cnv'] != -1]

    plot_df['logfc_ranked'] = rankdata(plot_df['logfc']) / plot_df.shape[0]
    plot_df['logfc_norm_ranked'] = rankdata(plot_df['logfc_norm']) / plot_df.shape[0]
    plot_df['ccleanr_ranked'] = rankdata(plot_df['ccleanr']) / plot_df.shape[0]

    plot_df['cnv'] = plot_df['cnv'].astype(int)

    sns.set(style='ticks', context='paper', font_scale=0.5, rc={'axes.linewidth': .3, 'xtick.major.width': .3, 'ytick.major.width': .3, 'xtick.major.size': 2.5, 'ytick.major.size': 2.5, 'xtick.direction': 'out', 'ytick.direction': 'out'})
    (f, axs), pos = plt.subplots(3), 0
    for i in ['logfc_ranked', 'logfc_norm_ranked', 'ccleanr_ranked']:
        sns.boxplot('cnv', i, data=plot_df, linewidth=.3, notch=True, fliersize=1, orient='v', palette='viridis', ax=axs[pos], sym='')
        sns.stripplot('cnv', i, data=plot_df, orient='v', palette='viridis', ax=axs[pos], size=0.75, linewidth=.1, edgecolor='white', jitter=.15, alpha=.75)

        axs[pos].axhline(.5, lw=.1, c='black', alpha=.5)

        axs[pos].set_xlabel('')
        axs[pos].set_ylabel(i)
        pos += 1

    plt.setp([a.get_xticklabels() for a in f.axes[:-1]], visible=False)

    plt.suptitle(sample, y=.95)
    plt.xlabel('Copy-number (absolute)')

    plt.gcf().set_size_inches(2, 3)
    plt.savefig('reports/%s_crispy_norm_boxplots.png' % sample, bbox_inches='tight', dpi=600)
    plt.close('all')
    print('[%s] Boxplot done.' % dt.now().strftime('%Y-%m-%d %H:%M:%S'))

    # - Scatter plot
    plot_df = df_genes.dropna(subset=['cnv'])

    cmap = plt.cm.get_cmap('viridis')

    sns.set(style='ticks', context='paper', font_scale=0.5, rc={'axes.linewidth': .3, 'xtick.major.width': .3, 'ytick.major.width': .3, 'xtick.major.size': 2.5, 'ytick.major.size': 2.5, 'xtick.direction': 'in', 'ytick.direction': 'in'})

    plt.scatter(plot_df['logfc'], plot_df['logfc_norm'], s=3, alpha=.2, edgecolor='w', lw=0.05, c=plot_df['cnv'], cmap=cmap)

    xlim, ylim = plt.xlim(), plt.ylim()
    xlim, ylim = np.min([xlim[0], ylim[0]]), np.max([xlim[1], ylim[1]])
    plt.plot((xlim, ylim), (xlim, ylim), 'k--', lw=.3, alpha=.5)
    plt.xlim((xlim, ylim))
    plt.ylim((xlim, ylim))

    plt.axhline(0, lw=.1, c='black', alpha=.5)
    plt.axvline(0, lw=.1, c='black', alpha=.5)

    plt.xlabel('Original FCs')
    plt.ylabel('Corrected FCs')

    plt.colorbar()

    plt.title(sample)

    plt.gcf().set_size_inches(2, 1.8)
    plt.savefig('reports/%s_crispy_norm_scatter.png' % sample, bbox_inches='tight', dpi=600)
    plt.close('all')
    print('[%s] Scatter plot done.' % dt.now().strftime('%Y-%m-%d %H:%M:%S'))

    # - Chromossome plot
    sns.set(style='ticks', context='paper', font_scale=0.75, rc={'axes.linewidth': .3, 'xtick.major.width': .3, 'ytick.major.width': .3, 'xtick.major.size': 2.5, 'ytick.major.size': 2.5, 'xtick.direction': 'in', 'ytick.direction': 'in'})
    gs, pos = GridSpec(len(set(df['CHRM'])), 1, hspace=.4, wspace=.1), 0

    for sample_chr in set(df['CHRM']):
        # Define plot data-frame
        ax = plt.subplot(gs[pos])
        plot_df = df[df['CHRM'] == sample_chr]

        # Cytobads
        for i in cytobands[cytobands['chr'] == 'chr%s' % sample_chr].index:
            s, e, t = cytobands.loc[i, ['start', 'end', 'band']].values

            if t == 'acen':
                ax.axvline(s, lw=.2, ls='-', color='#b1b1b1', alpha=.3)
                ax.axvline(e, lw=.2, ls='-', color='#b1b1b1', alpha=.3)

            elif not i % 2:
                ax.axvspan(s, e, alpha=0.2, facecolor='#b1b1b1')

        # Plot guides
        for s in set(plot_df['essential']):
            ax.scatter(plot_df.loc[plot_df['essential'] == s, 'STARTpos'], plot_df.loc[plot_df['essential'] == s, 'logfc'], s=2, marker='X' if s else '.', lw=0, c='#b1b1b1', alpha=.5)

        # Plot CRISPR segments
        for s, e, fc in crispr_seg.loc[(crispr_seg['sample'] == sample) & (crispr_seg['chrom'] == str(sample_chr)), ['loc.start', 'loc.end', 'seg.mean']].values:
            ax.plot([s, e], [fc, fc], lw=.3, c='#ff511d', alpha=.9)

        # Plot CNV segments
        for s, e, c in cnv_seg.loc[(cnv_seg['cellLine'] == sample) & (cnv_seg['chr'] == str(sample_chr)), ['startpos', 'endpos', 'totalCN']].values:
            ax.plot([s, e], [c, c], lw=.3, c='#3498db', alpha=.9)

        # Plot sgRNAs mean
        ax.scatter(plot_df['STARTpos'], plot_df['logfc_mean'], s=4, marker='.', lw=0, alpha=.9, c=plot_df['cnv'], cmap='viridis', vmin=0, vmax=14)
        # ax.fill_between(plot_df['STARTpos'], plot_df['logfc_mean'] - plot_df['logfc_se'], plot_df['logfc_mean'] + plot_df['logfc_se'], alpha=0.2, cmap='viridis')

        # Misc
        ax.axhline(0, lw=.3, ls='-', color='black')

        # Labels and dim
        ax.set_ylabel('chr%s' % sample_chr)
        ax.set_xlim(0, plot_df['STARTpos'].max())

        if pos == 0:
            ax.set_title(sample)

        pos += 1

    plt.gcf().set_size_inches(3, 1.5 * len(set(df['CHRM'])))
    plt.savefig('reports/%s_chromosome_plot.png' % sample, bbox_inches='tight', dpi=600)
    plt.close('all')
    print('[%s] Chromossome plot done.' % dt.now().strftime('%Y-%m-%d %H:%M:%S'))
