#!/usr/bin/env python
# Copyright (C) 2017 Emanuel Goncalves

import sys

if sys.platform == 'linux':
    import matplotlib
    matplotlib.use('Agg')
    print('Matplotlib Agg background set!')

import re
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from limix_core.gp import GP
from datetime import datetime as dt
from limix_core.mean import MeanBase
from scipy.stats.stats import rankdata
from sklearn.metrics.ranking import auc
from matplotlib.gridspec import GridSpec
from limix_core.covar import FixedCov, SQExpCov, SumCov
from limix_core.util.preprocess import covar_rescaling_factor_efficient
from sklearn.gaussian_process.gpr import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, Sum, ConstantKernel


# -- Ploting feature
class MidpointNormalize(colors.Normalize):
    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        colors.Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y), np.isnan(value))


# -- Imports
# Essential genes
essential = list(pd.read_csv('data/resources/curated_BAGEL_essential.csv', sep='\t')['gene'])

# sgRNA library
sgrna_lib = pd.read_csv('data/gdsc/crispr/KY_Library_v1.0.csv')

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


# -- Read arguments: sample and sample_chr
# sample, sample_chr = 'AU565', '17'
sample, sample_chr = sys.argv[1:]
print('[%s] Sample: %s; Chr: %s' % (dt.now().strftime('%Y-%m-%d %H:%M:%S'), sample, sample_chr))


# -- Define sample data-frame
# sgRNA of the sample chromossome
sample_sgrnas = list(sgrna_lib[sgrna_lib['CHRM'] == sample_chr].index)

# Build sample data-frame
df = crispr[sample].dropna().rename('logfc')

# Concat sgRNAs information
df = pd.concat([sgrna_lib.loc[sample_sgrnas], df.loc[sample_sgrnas]], axis=1).dropna()

# Concat copy-number
df['cnv'] = [cnv_abs.loc[g, sample] if g in cnv_abs.index else np.nan for g in df['GENES']]
df['loh'] = [cnv_loh.loc[g, sample] if g in cnv_loh.index else np.nan for g in df['GENES']]

# Check if gene is essential
df['essential'] = [int(i in essential) for i in df['GENES']]
# df = df[(df['STARTpos'] > 30000000) & (df['ENDpos'] < 45000000)]


# -- Variance decomposition
print('[%s] GP for variance decomposition started' % dt.now().strftime('%Y-%m-%d %H:%M:%S'))

# Define variables
y, X = df[['logfc']].values, df[['STARTpos']]

# Scale distances
X /= 1e8

# Instanciate the covariance functions
K = ConstantKernel(.5, (1e-2, 1.)) * RBF(X.mean(), (1e-2, 10)) + WhiteKernel()

# Instanciate a Gaussian Process model
gp = GaussianProcessRegressor(K, n_restarts_optimizer=3, normalize_y=False)

# Fit to data using Maximum Likelihood Estimation of the parameters
gp.fit(X, y)

# Normalise data
df = df.assign(logfc_mean=gp.predict(X))
df = df.assign(logfc_norm=df['logfc'] - df['logfc_mean'])

# Export
df.to_csv('reports/%s_%s_corrected.csv' % (sample, sample_chr))
print('[%s] GP for variance decomposition finshed' % dt.now().strftime('%Y-%m-%d %H:%M:%S'))


# # Define variables
# Y, X = df[['logfc']].values, df[['STARTpos']]
#
# # Scale distances
# X /= 1e9
#
# # Copy-number COV
# # cov_cna = pd.get_dummies(df['cnv'])
# # cov_cna = cov_cna.dot(cov_cna.transpose())
# #
# # cov_cna = FixedCov(cov_cna, cov_cna)
#
# # # LOH
# # cov_loh = pd.get_dummies(df['loh'])
# # cov_loh = cov_loh.dot(cov_loh.transpose())
# #
# # cov_loh = FixedCov(cov_loh, cov_loh)
# # cov_loh.scale *= covar_rescaling_factor_efficient(cov_loh.K())
# # cov_loh.scale /= 10
#
# # Distance COV
# cov_dist = SQExpCov(X, X)
# cov_dist.scale *= covar_rescaling_factor_efficient(cov_dist.K())
# cov_dist.length = np.max(X.values) * .1
#
# # Noise COV
# cov_noise = FixedCov(np.eye(X.shape[0]))
#
# # Sum COVs
# # 'cna': cov_cna,
# covs = {'distance': cov_dist, 'noise': cov_noise}
# K = SumCov(*covs.values())
#
# # GP
# gp = GP(MeanBase(Y), K)
# gp.optimize()
#
# # Explained variance
# cov_var = pd.Series({n: (1 / covar_rescaling_factor_efficient(covs[n].K())) for n in covs})
# cov_var /= cov_var.sum()
# print(cov_var.sort_values())
#
# # Normalise data
# df = df.assign(logfc_mean=gp.predict())
# df = df.assign(logfc_norm=df['logfc'] - df['logfc_mean'])


# - Evaluate
sns.set(style='ticks', context='paper', font_scale=0.75, palette='PuBu_r', rc={'axes.linewidth': .3, 'xtick.major.width': .3, 'ytick.major.width': .3, 'xtick.major.size': 2.5, 'ytick.major.size': 2.5, 'xtick.direction': 'in', 'ytick.direction': 'in'})
gs, pos = GridSpec(1, 2, hspace=.3, wspace=.3), 0
for t in ['essential', 'cnv']:
    ax = plt.subplot(gs[pos])

    pass_flag = 0

    for f in ['logfc', 'logfc_norm']:
        # Build data-frame
        plot_df = df.sort_values(f).copy()

        if t == 'cnv':
            plot_df[t] = plot_df[t].apply(lambda x: 1 if x >= 4 else 0)

        # Check if enough events are present
        if plot_df[t].sum() <= 1:
            pass_flag += 1
            continue

        # Rank fold-changes
        x = rankdata(plot_df[f]) / plot_df.shape[0]

        # Observed cumsum
        y = plot_df[t].cumsum() / plot_df[t].sum()

        # Plot
        ax.plot(x, y, label='%s: %.2f' % (f, auc(x, y)), lw=1.)

    # Check if this is any plot
    if pass_flag != 2:
        # Random
        r = np.array([plot_df[t].sample(frac=1.).cumsum() for i in range(101)]) / plot_df[t].sum()
        r_auc = [auc(x, i) for i in r]

        y_r_min, y_r_max, y_r_median = r[np.argmin(r_auc)], r[np.argmax(r_auc)], r[np.where(r_auc == np.median(r_auc))[0][0]]

        ax.plot(x, y_r_min, lw=.3, c='#969696')
        ax.plot(x, y_r_median, lw=.5, c='#969696', ls='-', label='Random')
        ax.plot(x, y_r_max, lw=.3, c='#969696')
        ax.fill_between(x, y_r_min, y_r_max, facecolor='#f7f7f7', alpha=0.5)

        # Misc
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

        ax.set_title('%s: %s' % (t.capitalize(), f.replace('_', ' ')))
        ax.set_xlabel('Ranked %s' % f)
        ax.set_ylabel('Cumulative sum %s' % t)

        ax.legend(loc=4)

    pos += 1

plt.gcf().set_size_inches(4.5, 2)
plt.savefig('reports/%s_%s_eval_plot.png' % (sample, sample_chr), bbox_inches='tight', dpi=600)
plt.close('all')


# - Chromossome plot
sns.set(style='ticks', context='paper', font_scale=0.75, rc={'axes.linewidth': .3, 'xtick.major.width': .3, 'ytick.major.width': .3, 'xtick.major.size': 2.5, 'ytick.major.size': 2.5, 'xtick.direction': 'in', 'ytick.direction': 'in'})
gs, pos = GridSpec(2, 1, hspace=.6, wspace=.6), 0

for feature in ['logfc', 'logfc_norm']:
    ax = plt.subplot(gs[pos])

    # Plot guides
    ax.scatter(df['STARTpos'], df[feature], s=2, marker='.', lw=0, c='#b1b1b1', alpha=.5)

    # Plot sgRNAs mean
    ax.scatter(df['STARTpos'], df['logfc_mean'], s=4, marker='.', lw=0, c='#2ecc71', alpha=.5)

    # # Plot sgRNAs of highlighted genes
    # for g, c in zip(*(['ERBB2'], sns.color_palette('OrRd_r', 2))):
    #     xs, ys = zip(*df.loc[df['GENES'] == g, ['STARTpos', feature]].values)
    #     ax.scatter(xs, ys, s=2, marker='x', lw=0.5, c=c, alpha=.9, label=g)

    # Plot CRISPR segments
    if pos == 0:
        for s, e, fc in crispr_seg.loc[(crispr_seg['sample'] == sample) & (crispr_seg['chrom'] == sample_chr), ['loc.start', 'loc.end', 'seg.mean']].values:
            ax.plot([s, e], [fc, fc], lw=.3, c='#ff511d', alpha=.5)

    # Plot CNV segments
    for s, e, c in cnv_seg.loc[(cnv_seg['cellLine'] == sample) & (cnv_seg['chr'] == sample_chr), ['startpos', 'endpos', 'totalCN']].values:
        ax.plot([s, e], [c, c], lw=.3, c='#3498db', alpha=.5)

    # Cytobads
    for i in cytobands[cytobands['chr'] == 'chr%s' % sample_chr].index:
        s, e, t = cytobands.loc[i, ['start', 'end', 'band']].values

        if t == 'acen':
            ax.axvline(s, lw=.2, ls='-', color='#b1b1b1', alpha=.3)
            ax.axvline(e, lw=.2, ls='-', color='#b1b1b1', alpha=.3)

        elif not i % 2:
            ax.axvspan(s, e, alpha=0.2, facecolor='#b1b1b1')

    # Misc
    ax.axhline(0, lw=.3, ls='-', color='black')

    ax.set_xlim(df['STARTpos'].min(), df['STARTpos'].max())

    ax.set_xlabel('Chromosome position (chr %s)' % sample_chr)
    ax.set_ylabel('sgRNA (%s)' % feature)

    plt.suptitle('%s' % gp.kernel_)
    # plt.suptitle('Variance explained: %s\nPath length:%.4E; Scale:%.4E' % (re.sub(' +', '=', cov_var.to_string(float_format='%.2f')).replace('\n', '; '), cov_dist.length, cov_dist.scale))

    pos += 1

plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
# plt.legend(loc=3)

plt.gcf().set_size_inches(4, 2.5)
plt.savefig('reports/%s_%s_chromosome_plot.png' % (sample, sample_chr), bbox_inches='tight', dpi=600)
plt.close('all')
print('[%s] Chromossome plot done.' % dt.now().strftime('%Y-%m-%d %H:%M:%S'))


# - Scatter plot
cmap = plt.cm.get_cmap('jet_r')

sns.set(style='ticks', context='paper', font_scale=0.5, rc={'axes.linewidth': .3, 'xtick.major.width': .3, 'ytick.major.width': .3, 'xtick.major.size': 2.5, 'ytick.major.size': 2.5, 'xtick.direction': 'in', 'ytick.direction': 'in'})

plt.scatter(df['logfc'], df['logfc_norm'], s=4, alpha=.3, edgecolor='w', lw=.1, c=df['cnv'], cmap=cmap, norm=MidpointNormalize(vmin=0, midpoint=2))

xlim, ylim = plt.xlim(), plt.ylim()
xlim, ylim = np.min([xlim[0], ylim[0]]), np.max([xlim[1], ylim[1]])
plt.plot((xlim, ylim), (xlim, ylim), 'k--', lw=.3, alpha=.5)
plt.xlim((xlim, ylim))
plt.ylim((xlim, ylim))

plt.axhline(0, lw=.3, c='black', alpha=.5)
plt.axvline(0, lw=.3, c='black', alpha=.5)

plt.xlabel('Original FCs')
plt.ylabel('Corrected FCs')

plt.colorbar()

plt.title('%s chr%s' % (sample, sample_chr))

plt.gcf().set_size_inches(2, 1.8)
plt.savefig('reports/%s_%s_crispy_norm_scatter.png' % (sample, sample_chr), bbox_inches='tight', dpi=600)
plt.close('all')
print('[%s] Scatter plot done.' % dt.now().strftime('%Y-%m-%d %H:%M:%S'))
