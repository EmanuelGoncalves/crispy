#!/usr/bin/env python
# Copyright (C) 2017 Emanuel Goncalves

import os
import crispy
import pickle
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from crispy import bipal_dbgd
from natsort import natsorted
from sklearn.metrics import roc_curve, auc, roc_auc_score


# - Imports
# Gene HGNC info
ginfo = pd.read_csv('data/crispy_hgnc_info.csv', index_col=0)

# Samplesheet
ss = pd.read_csv('data/gdsc/samplesheet.csv', index_col=0)

# MOBEMs
mobems = pd.read_csv('data/gdsc/mobems/PANCAN_simple_MOBEM.rdata.annotated.csv', index_col=0)

# Non-expressed genes
nexp = pickle.load(open('data/gdsc/nexp_pickle.pickle', 'rb'))

# GDSC
c_gdsc_fc = pd.read_csv('data/crispr_gdsc_crispy.csv', index_col=0)

c_gdsc = pd.DataFrame({
    os.path.splitext(f)[0].replace('crispr_gdsc_crispy_', ''):
        pd.read_csv('data/crispy/' + f, index_col=0)['k_mean']
    for f in os.listdir('data/crispy/') if f.startswith('crispr_gdsc_crispy_')
}).dropna()

# Copy-number absolute counts
cnv_abs = pd.read_csv('data/crispy_gene_copy_number.csv', index_col=0)

# Chromosome copies
chr_copies = pd.read_csv('data/crispy_chrm_copy_number.csv', index_col=(0, 1))

# Gene copy-number ratio
cnv_ratios = pd.read_csv('data/crispy_copy_number_ratio.csv', index_col=0).replace(-1, np.nan)


# - Overlap
genes, samples = set(c_gdsc.index).intersection(cnv_abs.index), set(c_gdsc).intersection(cnv_abs).intersection(nexp)
print(len(genes), len(samples))


# - Estimate gene copies ratio
df = pd.DataFrame({c: c_gdsc.loc[nexp[c], c] for c in c_gdsc if c in nexp})

df = pd.concat([
    df.loc[genes, samples].unstack().rename('crispy'),
    cnv_abs.loc[genes, samples].unstack().rename('cnv'),
    cnv_ratios.loc[genes, samples].unstack().rename('ratio')
], axis=1).dropna()

df = df.reset_index().rename(columns={'level_0': 'sample', 'level_1': 'gene'})

df = df.assign(chr=ginfo.loc[df['gene'], 'chr'].values)

df = df.assign(chr_cnv=chr_copies.loc[[(s, int(c)) for s, c in df[['sample', 'chr']].values]].round(0).astype(int).values)

df = df.assign(ratio_bin=[str(i) if i < 4 else '4+' for i in df['ratio'].round(0).astype(int)])


# - Ratios histogram
f, axs = plt.subplots(2, 1, sharex=True)

sns.distplot(df['ratio'], color=bipal_dbgd[0], kde=False, axlabel='', ax=axs[0])
sns.distplot(df['ratio'], color=bipal_dbgd[0], kde=False, ax=axs[1])

axs[0].axvline(1, lw=.3, ls='--', c=bipal_dbgd[1])
axs[1].axvline(1, lw=.3, ls='--', c=bipal_dbgd[1])

axs[0].set_ylim(470000, 550000)  # outliers only
axs[1].set_ylim(0, 120000)  # most of the data

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

plt.title('Non-expressed genes copy-number ratio')
plt.gcf().set_size_inches(3, 2)
plt.savefig('reports/ratio_histogram.png', bbox_inches='tight', dpi=600)
plt.close('all')


# - Copy-number ratio association with CRISPR/Cas9 bias
order = natsorted(set(df['ratio_bin']))
pal = sns.light_palette(bipal_dbgd[0], n_colors=len(order)).as_hex()

sns.boxplot('crispy', 'ratio_bin', data=df, orient='h', linewidth=.3, fliersize=1, order=order, palette=pal, notch=True)

# plt.axhline(1, lw=.3, ls='--', c=bipal_dbgd[1])
plt.axvline(0, lw=.1, ls='-', c=bipal_dbgd[0])

plt.title('Gene copy-number ratio\neffect on CRISPR/Cas9 bias')
plt.xlabel('CRISPR/Cas9 fold-change bias (log2)')
plt.ylabel('Copy-number ratio')
plt.gcf().set_size_inches(2, 2)
plt.savefig('reports/ratio_boxplot.png', bbox_inches='tight', dpi=600)
plt.close('all')


# - Ratio enrichment
ax = plt.gca()

for c, thres in zip(pal[2:], order[2:]):
    fpr, tpr, _ = roc_curve((df['ratio_bin'] == thres).astype(int), -df['crispy'])
    ax.plot(fpr, tpr, label='%s: AUC=%.2f' % (thres, auc(fpr, tpr)), lw=1., c=c)

ax.plot((0, 1), (0, 1), 'k--', lw=.3, alpha=.5)
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.set_xlabel('False positive rate')
ax.set_ylabel('True positive rate')
ax.set_title('CRISPR/Cas9 copy-number ratio bias\nnon-expressed genes')
legend = ax.legend(loc=4, title='Copy-number ratio', prop={'size': 8})
legend.get_title().set_fontsize('8')

plt.gcf().set_size_inches(2, 2)
plt.savefig('reports/ratio_aucs.png', bbox_inches='tight', dpi=600)
plt.close('all')


# - Ratios heatmap
plot_df = df.query("ratio_bin == '1'").groupby(['cnv', 'chr_cnv'])['ratio'].count().reset_index()
plot_df = pd.pivot_table(plot_df, index='cnv', columns='chr_cnv', values='ratio')

sns.set_style('whitegrid')
g = sns.heatmap(
    plot_df.loc[natsorted(plot_df.index, reverse=True)], center=0, cmap=sns.light_palette(bipal_dbgd[0], as_cmap=True), annot=True, fmt='g', square=True,
    linewidths=.3, cbar=False, annot_kws={'fontsize': 7}
)
plt.setp(g.get_yticklabels(), rotation=0)
plt.xlabel('# chromosome copies')
plt.ylabel('# gene copies')
plt.title('Non-expressed genes with copy-number ratio ~1')
plt.gcf().set_size_inches(5.5, 5.5)
plt.savefig('reports/ratio_counts_heatmap.png', bbox_inches='tight', dpi=600)
plt.close('all')
