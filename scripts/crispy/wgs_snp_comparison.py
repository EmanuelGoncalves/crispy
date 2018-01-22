#!/usr/bin/env python
# Copyright (C) 2018 Emanuel Goncalves

import crispy
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from crispy import bipal_dbgd
from scipy.stats import pearsonr
from crispy.ratio import plot_gene


def corrfunc(x, y, **kws):
    r, p = pearsonr(x, y)
    ax = plt.gca()
    ax.annotate(
        'R={:.2f}\n{:s}'.format(r, 'p=%.1e' % p),
        xy=(.5, .5), xycoords=ax.transAxes, horizontalalignment='center', verticalalignment='center', fontsize=7
    )


# - Compare estimated ploidy
picnic_ploidy = pd.read_csv('data/gdsc/ploidy.csv', index_col=0)['PLOIDY'].rename('picnic')
ascat_ploidy = pd.read_csv('data/gdsc/wgs/wgs_samplestatistics.csv', index_col='sample')['Ploidy'].rename('ascat')

crispy_picnic_ploidy = pd.read_csv('data/crispy_ploidy_snp.csv', names=['sample', 'ploidy'], index_col=0)['ploidy'].rename('crispy_picnic')
crispy_ascat_ploidy = pd.read_csv('data/crispy_ploidy_wgs.csv', names=['sample', 'ploidy'], index_col=0)['ploidy'].rename('crispy_ascat')

# Plot
plot_df = pd.concat([picnic_ploidy, ascat_ploidy, crispy_ascat_ploidy, crispy_picnic_ploidy], axis=1).dropna()

g = sns.PairGrid(plot_df, despine=False)

g.map_upper(sns.regplot, color='#34495e', truncate=True, scatter_kws={'s': 5, 'edgecolor': 'w', 'linewidth': .3, 'alpha': .8}, line_kws={'linewidth': .5})
g.map_diag(sns.distplot, kde=False, bins=10, hist_kws={'color': '#34495e'})
g.map_lower(corrfunc)

plt.suptitle('Ploidy WGS vs SNP', y=1.05)

plt.gcf().set_size_inches(3.5, 3.5)
plt.savefig('reports/crispy/wgs_snp_comparison_ploidy.png', bbox_inches='tight', dpi=600)
plt.close('all')


# - Compare gene copy-number
cnv_gene = pd.read_csv('data/gdsc/copynumber/Gene_level_CN.txt', sep='\t', index_col=0)

cnv_gene_max = cnv_gene.drop(['chr', 'start', 'stop'], axis=1, errors='ignore').applymap(lambda v: int(v.split(',')[0])).replace(-1, np.nan)
cnv_gene_min = cnv_gene.drop(['chr', 'start', 'stop'], axis=1, errors='ignore').applymap(lambda v: int(v.split(',')[1])).replace(-1, np.nan)

cnv_gene_snp = pd.read_csv('data/crispy_gene_copy_number_snp.csv', index_col=0)
cnv_gene_wgs = pd.read_csv('data/crispy_gene_copy_number_wgs.csv', index_col=0)

# Overlap
samples = set(cnv_gene_max).intersection(cnv_gene_min).intersection(cnv_gene_snp).intersection(cnv_gene_wgs)
genes = set(cnv_gene_max.index).intersection(cnv_gene_min.index).intersection(cnv_gene_snp.index).intersection(cnv_gene_wgs.index)
print('Samples: %d; Genes: %d' % (len(samples), len(genes)))

#
snp_vs_max = cnv_gene_snp.loc[genes, samples].unstack().subtract(cnv_gene_max.loc[genes, samples].unstack()).dropna()
snp_vs_min = cnv_gene_snp.loc[genes, samples].unstack().subtract(cnv_gene_min.loc[genes, samples].unstack()).dropna()

#
sample, gene = 'HCC2218', 'CDK12'

ax = plot_gene(gene, 'data/gdsc/copynumber/snp6_bed/%s.snp6.picnic.bed' % sample)

ax.axhline(cnv_gene_min.loc[gene, sample], lw=.3, ls='--', label='min', c=bipal_dbgd[0])
ax.axhline(cnv_gene_snp.loc[gene, sample], lw=.3, ls='--', label='wmean', c=bipal_dbgd[1])

ax.set_title(gene)
ax.set_xlabel('Genomic coordinates')
ax.set_ylabel('Total copy-number')

ax.legend()

plt.gcf().set_size_inches(4, 2)
plt.savefig('reports/crispy/wgs_snp_comparison_%s_%s.png' % (sample, gene), bbox_inches='tight', dpi=600)
plt.close('all')

# Differences barplot
for n, df in [('max', snp_vs_max), ('min', snp_vs_min)]:
    f, axs = plt.subplots(2, 1, sharex=True)

    sns.distplot(df.astype(int), color=bipal_dbgd[0], kde=False, axlabel='', ax=axs[0])
    sns.distplot(df.astype(int), color=bipal_dbgd[0], kde=False, ax=axs[1])

    ymax = df.astype(int).value_counts()
    axs[0].set_ylim(ymax.iloc[0] - 100, ymax.iloc[0] + 10)  # outliers only
    axs[1].set_ylim(0, ymax.iloc[1] + 20)  # most of the data

    axs[0].spines['bottom'].set_visible(False)
    axs[1].spines['top'].set_visible(False)
    axs[0].xaxis.tick_top()
    axs[0].tick_params(labeltop='off')
    axs[1].xaxis.tick_bottom()

    axs[0].get_yaxis().get_major_formatter().set_useOffset(False)
    axs[1].get_yaxis().get_major_formatter().set_useOffset(False)

    d = .01

    kwargs = dict(transform=axs[0].transAxes, color='gray', clip_on=False, lw=.3)
    axs[0].plot((-d, +d), (-d, +d), **kwargs)
    axs[0].plot((1 - d, 1 + d), (-d, +d), **kwargs)

    kwargs.update(transform=axs[1].transAxes)
    axs[1].plot((-d, +d), (1 - d, 1 + d), **kwargs)
    axs[1].plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)

    axs[1].set_xlabel('Copy-number difference')

    f.add_subplot(111, frameon=False)
    plt.tick_params(labelcolor='none', top='off', bottom='off', left='off', right='off')
    plt.grid(False)

    plt.title('Gene copy-number weighted vs %s' % n)
    plt.gcf().set_size_inches(3, 2)
    plt.savefig('reports/crispy/wgs_snp_comparison_%s_barplot.png' % n, bbox_inches='tight', dpi=600)
    plt.close('all')
