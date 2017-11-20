#!/usr/bin/env python
# Copyright (C) 2017 Emanuel Goncalves

import os
import pickle
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from natsort import natsorted
from matplotlib.gridspec import GridSpec
from sklearn.metrics import roc_curve, auc
from crispy.benchmark_plot import plot_cumsum_auc, plot_chromosome, plot_cnv_rank


# - Imports
# sgRNA library
sgrna_lib = pd.read_csv('data/gdsc/crispr/KY_Library_v1.1_annotated.csv', index_col=0).dropna(subset=['STARTpos'])

# Samplesheet
ss = pd.read_csv('data/gdsc/samplesheet.csv', index_col=0)

# Essential genes
essential = pd.read_csv('data/resources/curated_BAGEL_essential.csv', sep='\t')['gene'].rename('essential genes')

# Non-expressed genes
nexp = pickle.load(open('data/gdsc/gene_expression/non_expressed_genes.pickle', 'rb'))

# Chromosome cytobands
cytobands = pd.read_csv('data/resources/cytoBand.txt', sep='\t')

# CRISPR fold-changes
fc_sgrna = pd.read_csv('data/gdsc/crispr/crispy_gdsc_fold_change_sgrna.csv', index_col=0)

# Copy-number segments
cnv_seg = pd.read_csv('data/gdsc/copynumber/Summary_segmentation_data_994_lines_picnic.txt', sep='\t')
cnv_seg['chr'] = cnv_seg['chr'].replace(23, 'X').replace(24, 'Y').astype(str)

# Copy-number absolute counts
cnv = pd.read_csv('data/gdsc/copynumber/Gene_level_CN.txt', sep='\t', index_col=0)
cnv = cnv.loc[:, cnv.columns.isin(list(fc_sgrna))]
cnv_abs = cnv.applymap(lambda v: int(v.split(',')[0]))

# Gene-expression
gexp = pd.read_csv('data/gdsc/gene_expression/merged_voom_preprocessed.csv', index_col=0)

# CIRSPR segments
crispr_seg = pd.read_csv('data/gdsc/crispr/segments_cbs_deseq2.csv')

# ccleanr
ccleanr = pd.read_csv('data/gdsc/crispr/all_GLCFC.csv', index_col=0)

# Ploidy
ploidy = pd.read_csv('data/gdsc/ploidy.csv', index_col=0).drop(['ACF'], axis=1)

# - Imports CCLE
cmap = pd.read_csv('data/ccle/CCLE_GDSC_cellLineMappoing.csv', index_col=2)

# CCLE: ceres
ceres = pd.read_csv('data/ccle/crispr_ceres/ceresgeneeffects.csv', index_col=0)
ceres = ceres.rename(columns=cmap['GDSC1000 name'])

# Project DRIVE
d_drive_rsa = pd.read_csv('data/ccle/DRIVE_RSA_data.txt', sep='\t')
d_drive_rsa.columns = [c.upper() for c in d_drive_rsa]
d_drive_rsa = d_drive_rsa.rename(columns=cmap['GDSC1000 name'])
print('Project DRIVE: ', d_drive_rsa.shape)


# - Crispy corrected fold-changes
crispy_files = [f for f in os.listdir('data/crispy/') if f.startswith('crispy_gdsc_')]

fc_crispy = {f.split('_')[2]: pd.read_csv('data/crispy/%s' % f, index_col=0) for f in crispy_files if f.endswith('_gene.csv')}

crispy_original = pd.DataFrame({i: fc_crispy[i]['logfc_beta'] for i in fc_crispy})
crispy_corrected = pd.DataFrame({i: fc_crispy[i]['regressed_out_beta'] for i in fc_crispy})

# sgrna level
fc_crispy_sgrna = {f.split('_')[2]: pd.read_csv('data/crispy/%s' % f, index_col=0) for f in crispy_files if f.endswith('_sgrna.csv')}


# - Copy-number effect variation
plot_df = pd.concat([fc_crispy_sgrna[i].groupby('GENES')[['k_mean', 'cnv']].first().assign(sample=i).reset_index() for i in fc_crispy_sgrna]).query('cnv != -1')
plot_df = plot_df.assign(Diploidy=list(plot_df.groupby('sample')['cnv'].median().loc[plot_df['sample']]))

order = natsorted(set(plot_df['cnv']))
hue_order = natsorted(set(plot_df['Diploidy']))

sns.pointplot(
    x='k_mean', y='cnv', hue='Diploidy', data=plot_df, orient='h',
    ci='sd', order=order, hue_order=hue_order, palette='Reds_r',
    errwidth=1., capsize=.1, scale=.5
)
plt.xlabel('CRISPR/Cas9 fold-change bias (mean)')
plt.ylabel('Copy-number variation')
plt.gcf().set_size_inches(2, 3)
plt.savefig('reports/cnv_effects.png', bbox_inches='tight', dpi=600)
plt.close('all')


# - Copy-number effect variation
genes_ov = set(crispy_original.index).intersection(d_drive_rsa.index).intersection(cnv_abs.index)
samples_ov = set(crispy_original).intersection(d_drive_rsa).intersection(cnv_abs)
print(len(genes_ov), len(samples_ov))

plot_df = pd.concat([
    crispy_original.loc[genes_ov, samples_ov].unstack().rename('crispr'),
    d_drive_rsa.loc[genes_ov, samples_ov].unstack().rename('shrna'),
    cnv_abs.loc[genes_ov, samples_ov].unstack().rename('cnv'),
    crispy_corrected.loc[genes_ov, samples_ov].unstack().rename('crispy'),
], axis=1).query('cnv != -1').reset_index().rename(columns={'level_0': 'sample', 'level_1': 'gene'})
plot_df = plot_df.assign(ploidy=cnv_abs.median().astype(int).loc[plot_df['sample']].values)

order = natsorted(set(plot_df['cnv']))
hue_order = natsorted(set(plot_df['ploidy']))

sns.pointplot(
    x='crispr', y='cnv', hue='ploidy', data=plot_df, orient='h',
    ci='sd', order=order, hue_order=hue_order, palette='Reds_r',
    errwidth=1., capsize=.1, scale=.5, alpha=.7, aspect=.5, size=2
)
plt.xlabel('CRISPR/Cas9 fold-change (mean)')
plt.ylabel('Copy-number variation')
plt.gcf().set_size_inches(2, 3)
plt.savefig('reports/cnv_effects_crispr.png', bbox_inches='tight', dpi=600)
plt.close('all')

sns.pointplot(
    x='shrna', y='cnv', hue='ploidy', data=plot_df, orient='h',
    ci='sd', order=order, hue_order=hue_order, palette='Reds_r',
    errwidth=1., capsize=.1, scale=.5, alpha=.7
)
plt.xlabel('shRNA RSA (mean)')
plt.ylabel('Copy-number variation')
plt.gcf().set_size_inches(2, 3)
plt.savefig('reports/cnv_effects_shrna.png', bbox_inches='tight', dpi=600)
plt.close('all')

sns.pointplot(
    x='crispy', y='cnv', hue='ploidy', data=plot_df, orient='h',
    ci='sd', order=order, hue_order=hue_order, palette='Reds_r',
    errwidth=1., capsize=.1, scale=.5, alpha=.7
)
plt.xlabel('Crispy fold-changes (mean)')
plt.ylabel('Copy-number variation')
plt.gcf().set_size_inches(2, 3)
plt.savefig('reports/cnv_effects_crispy.png', bbox_inches='tight', dpi=600)
plt.close('all')


# - Essential genes polar
plot_df = {}
for i in fc_crispy:
    plot_df[i] = round((fc_crispy[i].drop(essential, errors='ignore').query('regressed_out_qvalue < 0.05').count()['sample'] / fc_crispy[i].shape[0]) * 100, 1)
plot_df = pd.Series(plot_df, name='Count')
plot_df = pd.concat([plot_df, ss.loc[plot_df.index]], axis=1).dropna(subset=['TCGA']).sort_values(['TCGA', 'Count'])

pal = pd.Series(dict(zip(*(set(plot_df['TCGA']), sns.color_palette('Set1', len(set(plot_df['TCGA']))).as_hex()))))

ax = plt.subplot(111, projection='polar')

angles = np.deg2rad(np.arange(90, 90 + 360, 360.0 / plot_df.shape[0]))
ax.scatter(angles, plot_df['Count'], c=pal.loc[plot_df['TCGA']])

ax.set_ylim(0, 7)
ax.xaxis.set_visible(False)
ax.set_rgrids(range(1, 8), labels=['%d%%' % i for i in range(1, 8)], angle=90)

handles = [mpatches.Circle([.0, .0], .25, facecolor=c, label=t) for t, c in pal.items()]
ax.legend(handles=handles, title='Tissue', loc='center left', bbox_to_anchor=(1.05, 0.5))

plt.title('Percentage of essential genes')
plt.savefig('reports/essential_genes_radial.png', bbox_inches='tight', dpi=600)
plt.close('all')


# - Essential genes AROCs
genes = set(crispy_original.index).intersection(ccleanr.index).intersection(crispy_corrected.index)
samples = set(crispy_original).intersection(ccleanr).intersection(crispy_corrected)
print(len(genes), len(samples))

res_auc = {}
for s in samples:
    res_auc[s] = {}
    for n, d in [('crispy', crispy_corrected), ('ccleanr', ccleanr), ('original', crispy_original)]:
        ax, ax_stats = plot_cumsum_auc(d[[s]], essential, legend=False)
        plt.close('all')
        res_auc[s][n] = ax_stats['auc'][s]
        print(s, n, ax_stats)

res_auc = pd.DataFrame(res_auc).T
print(res_auc.sort_values('ccleanr'))

# Plot
x1, x2, y = res_auc['original'], res_auc['ccleanr'], res_auc['crispy']

gs = GridSpec(1, 2, hspace=.6, wspace=.6)
for i, (x, y, xn, yn) in enumerate([(x1, y, 'Original FC', 'Crispy FC'), (x2, y, 'CCleanR FC', 'Crispy FC')]):
    ax = plt.subplot(gs[i])

    ax.scatter(x, y, c='#37454B', s=3, marker='.', lw=0.5)
    sns.rugplot(x, height=.02, axis='x', c='#37454B', lw=.3, ax=ax)
    sns.rugplot(y, height=.02, axis='y', c='#37454B', lw=.3, ax=ax)

    xlim, ylim = 0.82, 0.9
    ax.plot([xlim, ylim], [xlim, ylim], ls='-', lw=.2, alpha=.8, c='#37454B')

    ax.set_xlim((xlim, ylim))
    ax.set_ylim((xlim, ylim))

    ax.set_xlabel(xn)
    ax.set_ylabel(yn)

plt.suptitle('Essential genes AROC', y=1.05, fontsize=8)
plt.gcf().set_size_inches(4, 1.5)
plt.savefig('reports/gdsc_essential_arocs.png', bbox_inches='tight', dpi=600)
plt.close('all')


# - Copy-number effect in non-expressed genes
plot_df = pd.concat([
    pd.concat([crispy_original.loc[nexp[i], i] for i in crispy_original if i in nexp], axis=1).unstack().rename('original'),
    pd.concat([crispy_corrected.loc[nexp[i], i] for i in crispy_corrected if i in nexp], axis=1).unstack().rename('crispy'),
    pd.concat([ccleanr.loc[nexp[i], i] for i in ccleanr if i in nexp], axis=1).unstack().rename('ccleanr'),
    cnv_abs.unstack().rename('cnv')
], axis=1).dropna().query('cnv != -1')

plot_df = plot_df.assign(cnv_d=['%d' % i if i < 10 else '10+' for i in plot_df['cnv']])

# Plot boxplots
order = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10+']

gs = GridSpec(3, 1, hspace=.3, wspace=.1)
for i, f in enumerate(['original', 'crispy', 'ccleanr']):
    ax = plt.subplot(gs[i])

    plot_cnv_rank(plot_df['cnv_d'], plot_df[f], stripplot=False, order=order, ax=ax)
    ax.set_ylabel(f.capitalize())

    if i == 0:
        ax.set_title('CNV effect non-expressed genes')

plt.gcf().set_size_inches(2.5, 3.5)
plt.savefig('reports/gdsc_cnv_effect_nexp.png', bbox_inches='tight', dpi=600)
plt.close('all')

# CCleanR scatter
g = sns.jointplot(
    plot_df['crispy'].rename('Crispy FC'), plot_df['ccleanr'].rename('CCleanR FC'), kind='reg', color='#37454B', space=0,
    marginal_kws={'hist': True, 'rug': False, 'kde': False, 'bins': 20}, annot_kws={'template': 'R={val:.2g}, p={p:.1e}', 'loc': 4},
    joint_kws={'scatter_kws': {'s': 2, 'edgecolor': 'w', 'linewidth': .3, 'alpha': .5}, 'line_kws': {'linewidth': .5}, 'truncate': True}
)
g.ax_joint.axhline(0, ls='-', lw=.1, c='#37454B')
g.ax_joint.axvline(0, ls='-', lw=.1, c='#37454B')
plt.gcf().set_size_inches(2, 2)
plt.savefig('reports/gdsc_ccleanr_scatter.png', bbox_inches='tight', dpi=600)
plt.close('all')


# - Chromossome plot
s, c = 'AU565', '8'

ghighlight = None

plot_df = fc_crispy[s].query("chrm == '%s'" % c)
plot_df = plot_df.assign(STARTpos=sgrna_lib.groupby('GENES')['STARTpos'].mean())

plot_chromosome(
    plot_df['STARTpos'], plot_df['original'].rename('Original FC'), plot_df['mean'].rename('Crispy FC'),
    seg=cnv_seg.query("(cellLine == '%s') & (chr == '%s')" % (s, c)),
    cytobands=cytobands.query("chr == 'chr%s'" % c),
    highlight=ghighlight, legend=True
)

plt.ylabel('')
plt.title('Sample: %s; Chr: %s; Var: %.2f' % (s, c, plot_df['var_rq'].mean()))
plt.gcf().set_size_inches(3, 1.5)
plt.savefig('reports/gdsc_chromossome_plot_%s_%s.png' % (s, c), bbox_inches='tight', dpi=600)
plt.close('all')


# - Gene essential DRIVE AROC
samples = set([idx for idx, val in cmap['GDSC1000 name'].iteritems() if val in ccleanr.columns]).intersection(d_drive_rsa)
genes = set(crispy_corrected.index).intersection(d_drive_rsa.index)
print('Overlap: ', len(genes), len(samples))

plot_df = pd.DataFrame({
    'CRISPR': crispy_corrected.loc[genes, cmap.loc[samples, 'GDSC1000 name']].unstack(),
    'shRNA': d_drive_rsa.loc[genes, samples].rename(columns=cmap.loc[samples, 'GDSC1000 name'].to_dict()).unstack()
}).dropna()
plot_df['shRNA_int'] = [round(i, 0) for i in plot_df['shRNA']]
plot_df['shRNA_int'] = plot_df['shRNA_int'].clip(-6, 0).astype(int)

# Plot
f, axs = plt.subplots(1, 2)

# AROCs
ax = axs[0]

thresholds = np.arange(-6, 0)
for c, thres in zip(sns.color_palette('Reds_r', len(thresholds)), thresholds):
    fpr, tpr, _ = roc_curve((plot_df['shRNA'] < thres).astype(int), -plot_df['CRISPR'])
    ax.plot(fpr, tpr, label='<%d = %.2f (AUC)' % (thres, auc(fpr, tpr)), lw=1., c=c)

ax.plot((0, 1), (0, 1), 'k--', lw=.3, alpha=.5)
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.set_xlabel('False positive rate')
ax.set_ylabel('True positive rate')
ax.set_title('Classify shRNA with CRISPR')
ax.legend(loc=4, title='DRIVE')

# Boxplots
ax = axs[1]

sns.boxplot('shRNA_int', 'CRISPR', data=plot_df, fliersize=1, palette='Reds_r', notch=True, linewidth=.5)
ax.axhline(0, ls='--', lw=.3, alpha=.5, c='black')
ax.set_xlabel('Rounded and capped shRNA RSA')
ax.set_ylabel('Crispy log2FC')
ax.set_title('CRISPR fold-change distributions')

plt.gcf().set_size_inches(7, 3)
plt.savefig('reports/gdsc_shrna_validation.png', bbox_inches='tight', dpi=600)
plt.close('all')


# -
plot_df = pd.DataFrame({i: fc_crispy[i].groupby('chrm')['var_rq'].first() for i in fc_crispy}).round(2).unstack().rename('var').to_frame()
plot_df = plot_df.assign(tcga=list(ss.loc[plot_df.index.get_level_values(0), 'TCGA'])).dropna().reset_index().rename(columns={'level_0': 'sample'})

s_counts = plot_df.groupby(['tcga'])['sample'].agg(lambda x: len(set(x)))

# t_type = set(s_counts[s_counts > 3].index)
t_type = {'BRCA'}
plot_df = plot_df[plot_df['tcga'].isin(t_type)]

g = sns.pointplot(
    'chrm', 'var', 'tcga', plot_df,
    palette='Set3', ci=None, estimator=np.mean, order=natsorted(set(plot_df['chrm'])), hue_order=set(plot_df['tcga'])
)

plt.legend(loc='center left', bbox_to_anchor=(1., 0.5))
plt.savefig('reports/gdsc_chr_var.png', bbox_inches='tight', dpi=600)
plt.close('all')
