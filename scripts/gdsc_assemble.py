#!/usr/bin/env python
# Copyright (C) 2017 Emanuel Goncalves

import os
import pickle
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from crispy.benchmark_plot import plot_cumsum_auc, plot_chromosome, plot_cnv_rank


# - Imports
# Samplesheet
ss = pd.read_csv('data/gdsc/samplesheet.csv', index_col=0)

# Essential genes
essential = pd.read_csv('data/resources/curated_BAGEL_essential.csv', sep='\t')['gene'].rename('essential genes')

# Non-expressed genes
nexp = pickle.load(open('data/gdsc/gene_expression/non_expressed_genes.pickle', 'rb'))

# CRISPR fold-changes
fc_sgrna = pd.read_csv('data/gdsc/crispr/crispy_gdsc_fold_change_sgrna.csv', index_col=0)

# Copy-number segments
cnv_seg = pd.read_csv('data/gdsc/copynumber/Summary_segmentation_data_994_lines_picnic.txt', sep='\t')
cnv_seg['chr'] = cnv_seg['chr'].replace(23, 'X').replace(24, 'Y').astype(str)

# Copy-number absolute counts
cnv = pd.read_csv('data/gdsc/copynumber/Gene_level_CN.txt', sep='\t', index_col=0)
cnv = cnv.loc[:, cnv.columns.isin(list(fc_sgrna))]
cnv_abs = cnv.applymap(lambda v: int(v.split(',')[0]))


# - CRISPR
# Assemble corrected fold-changes
fc_crispy = {
    os.path.splitext(f)[0].split('_')[7]: pd.read_csv('data/crispy/%s' % f, index_col=0) for f in os.listdir('data/crispy/') if f.startswith('crispy_gdsc_fold_change_gene_unsupervised_1e-4')
}
fcc = pd.DataFrame({i: fc_crispy[i]['original'] for i in fc_crispy})
fcc_gene = pd.DataFrame({i: fc_crispy[i]['corrected'] for i in fc_crispy})
fcc_var = pd.DataFrame({i: fc_crispy[i].groupby('chrm')['var_rq'].first() for i in fc_crispy})

# Assemble corrected fold-changes
fc_rbf = {
    os.path.splitext(f)[0].split('_')[7]: pd.read_csv('data/crispy/%s' % f, index_col=0) for f in os.listdir('data/crispy/') if f.startswith('crispy_gdsc_fold_change_gene_cnv_rbf')
}
fcc_rbf_gene = pd.DataFrame({i: fc_rbf[i]['corrected'] for i in fc_rbf})

# Assemble corrected fold-changes
fc_linear = {
    os.path.splitext(f)[0].split('_')[7]: pd.read_csv('data/crispy/%s' % f, index_col=0) for f in os.listdir('data/crispy/') if f.startswith('crispy_gdsc_fold_change_gene_cnv_linear')
}
fcc_linear_gene = pd.DataFrame({i: fc_linear[i]['corrected'] for i in fc_linear})

# ccleanr
ccleanr = pd.read_csv('data/gdsc/crispr/all_GLCFC.csv', index_col=0)


# - Overlap with CCleanR
genes, samples = set(fcc_gene.index).intersection(ccleanr.index).intersection(fcc.index), set(fcc_gene).intersection(ccleanr).intersection(fcc)
print(len(genes), len(samples))

res_auc = {}
for s in samples:
    res_auc[s] = {}
    for n, d in [('crispy', fcc_gene), ('ccleanr', ccleanr), ('original', fcc)]:
        ax, ax_stats = plot_cumsum_auc(d[[s]], essential, legend=False)
        plt.close('all')
        res_auc[s][n] = ax_stats['auc'][s]
        print(s, n, ax_stats)

res_auc = pd.DataFrame(res_auc).T
print(res_auc.sort_values('ccleanr'))

# Plot
x1, x2, y = res_auc['original'], res_auc['ccleanr'], res_auc['crispy']

gs, pos = GridSpec(1, 2, hspace=.6, wspace=.6), 0
for x, y, xn, yn in [(x1, y, 'Original FC', 'Crispy FC'), (x2, y, 'CCleanR FC', 'Crispy FC')]:
    ax = plt.subplot(gs[pos])
    ax.scatter(x, y, c='#F2C500', s=3, marker='x', lw=0.5)
    sns.rugplot(x, height=.02, axis='x', c='#F2C500', lw=.3, ax=ax)
    sns.rugplot(y, height=.02, axis='y', c='#F2C500', lw=.3, ax=ax)

    x_lim, y_lim = ax.get_xlim(), ax.get_ylim()
    ax.plot(x_lim, y_lim, 'k--', lw=.3)

    ax.set_xlim(x_lim)
    ax.set_ylim(y_lim)

    ax.set_xlabel(xn)
    ax.set_ylabel(yn)

    pos += 1

plt.suptitle('Essential genes AROC')
plt.gcf().set_size_inches(4, 1.5)
plt.savefig('reports/comparison_essential_scatter.png', bbox_inches='tight', dpi=600)
plt.close('all')


# - Overlap with linear/rbf
genes, samples = set(fcc_gene.index).intersection(fcc_rbf_gene.index).intersection(fcc_linear_gene.index), set(fcc_gene).intersection(fcc_rbf_gene).intersection(fcc_linear_gene)
print(len(genes), len(samples))

res_auc = {}
for s in samples:
    res_auc[s] = {}
    for n, d in [('crispy', fcc_gene), ('rbf', fcc_rbf_gene), ('linear', fcc_linear_gene)]:
        ax, ax_stats = plot_cumsum_auc(d[[s]], essential, legend=False)
        plt.close('all')
        res_auc[s][n] = ax_stats['auc'][s]
        print(s, n, ax_stats)

res_auc = pd.DataFrame(res_auc).T
print(res_auc.sort_values('crispy'))

# Plot
x1, x2, y = res_auc['rbf'], res_auc['linear'], res_auc['crispy']

gs, pos = GridSpec(1, 2, hspace=.6, wspace=.6), 0
for x, y, xn, yn in [(x1, y, 'RBF FC', 'Crispy FC'), (x2, y, 'Linear FC', 'Crispy FC')]:
    ax = plt.subplot(gs[pos])
    ax.scatter(x, y, c='#F2C500', s=3, marker='x', lw=0.5)
    sns.rugplot(x, height=.02, axis='x', c='#F2C500', lw=.3, ax=ax)
    sns.rugplot(y, height=.02, axis='y', c='#F2C500', lw=.3, ax=ax)

    x_lim, y_lim = ax.get_xlim(), ax.get_ylim()
    ax.plot(x_lim, y_lim, 'k--', lw=.3)

    ax.set_xlim(x_lim)
    ax.set_ylim(y_lim)

    ax.set_xlabel(xn)
    ax.set_ylabel(yn)

    pos += 1

plt.suptitle('Essential genes AROC')
plt.gcf().set_size_inches(4, 1.5)
plt.savefig('reports/comparison_essential_scatter_cnv.png', bbox_inches='tight', dpi=600)
plt.close('all')


# -
plot_df = pd.concat([
    pd.concat([fcc.loc[nexp[i], i] for i in fcc if i in nexp], axis=1).unstack().rename('original'),
    pd.concat([fcc_gene.loc[nexp[i], i] for i in fcc_gene if i in nexp], axis=1).unstack().rename('crispy'),
    pd.concat([ccleanr.loc[nexp[i], i] for i in ccleanr if i in nexp], axis=1).unstack().rename('ccleanr'),
    cnv_abs.unstack().rename('cnv')
], axis=1).dropna()
plot_df = plot_df.query('cnv != -1')
plot_df = plot_df.assign(cnv_d=['%d' % i if i < 10 else '10+' for i in plot_df['cnv']])

order = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10+']

for f in ['original', 'crispy', 'ccleanr']:
    plot_cnv_rank(plot_df['cnv_d'], plot_df[f], stripplot=False, order=order)
    plt.title('Overall CNV effect on non-expressed genes')
    plt.gcf().set_size_inches(3, 1.5)
    plt.savefig('reports/comparison_cnv_%s.png' % f, bbox_inches='tight', dpi=600)
    plt.close('all')


# - Chromossome plot
s, c = 'EBC-1', '7'

plot_df = fc_crispy[s].query("chrm == '%s'" % c)
plot_chromosome(plot_df['STARTpos'], plot_df['original'], plot_df['mean'], seg=cnv_seg.query("(cellLine == '%s') & (chr == '%s')" % (s, c)))
plt.ylabel('Chrm %s' % c)
plt.title('Sample: %s; Var: %.2f' % (s, plot_df['var_rq'].mean()))
plt.gcf().set_size_inches(3, 1.5)
plt.savefig('reports/chromosome_plot.png', bbox_inches='tight', dpi=600)
plt.close('all')


# - Copy-number plot
s = 'EBC-1'

# Crispy
for df, n in [(fc_crispy, 'crispy'), (fc_rbf, 'RBF'), (fc_linear, 'Linear')]:
    if n != 'crispy':
        plot_df = df[s].loc[nexp[s]].query('cnv != -1').dropna()
    else:
        plot_df = pd.concat([df[s], cnv_abs[s].rename('cnv')], axis=1).loc[nexp[s]].query('cnv != -1').dropna()

    gs = GridSpec(1, 2, hspace=.6, wspace=.3)

    ax = plt.subplot(gs[0])
    plot_cnv_rank(plot_df['cnv'].astype(int), plot_df['original'].rename('Original FC'))

    ax = plt.subplot(gs[1])
    plot_cnv_rank(plot_df['cnv'].astype(int), plot_df['corrected'].rename('%s FC' % n))

    plt.suptitle('Non-expressed genes, %s, diplody: %d' % (s, round(cnv_abs[s].mean())), y=1.05)
    plt.gcf().set_size_inches(6, 1.5)
    plt.savefig('reports/cnv_boxplot_%s.png' % n, bbox_inches='tight', dpi=600)
    plt.close('all')


# -
for df, n in [(fc_crispy, 'Original'), (fc_crispy, 'Crispy'), (fc_rbf, 'RBF'), (fc_linear, 'Linear'), (ccleanr, 'ccleanr')]:
    if n == 'Original':
        plot_df = pd.concat([
            pd.concat([df[s], cnv_abs[s].rename('cnv')], axis=1).loc[nexp[s]].query('cnv != -1').dropna().groupby('cnv')['original'].mean().reset_index().assign(name=s).rename(columns={'original': 'corrected'})
            for s in set(df).intersection(cnv_abs).intersection(nexp)
        ]).reset_index()

    elif n == 'Crispy':
        plot_df = pd.concat([
            pd.concat([df[s], cnv_abs[s].rename('cnv')], axis=1).loc[nexp[s]].query('cnv != -1').dropna().groupby('cnv')['corrected'].mean().reset_index().assign(name=s)
            for s in set(df).intersection(cnv_abs).intersection(nexp)
        ]).reset_index()
    elif n in {'RBF', 'Linear'}:
        plot_df = pd.concat([
            df[s].loc[nexp[s]].query('cnv != -1').dropna().groupby('cnv')['corrected'].mean().reset_index().assign(name=s)
            for s in set(df).intersection(cnv_abs).intersection(nexp)
        ]).reset_index()
    else:
        plot_df = pd.concat([
            pd.concat([df[s].rename('corrected'), cnv_abs[s].rename('cnv')], axis=1).loc[nexp[s]].query('cnv != -1').dropna().groupby('cnv')['corrected'].mean().reset_index().assign(name=s)
            for s in set(df).intersection(cnv_abs).intersection(nexp)
        ]).reset_index()

    plot_cnv_rank(plot_df['cnv'].astype(int), plot_df['corrected'])

    plt.ylabel('FC non-expressed genes')
    plt.title('%s mean FC per cell lines' % n)

    plt.gcf().set_size_inches(3, 1.5)
    plt.savefig('reports/cnv_boxplot_average_%s.png' % n, bbox_inches='tight', dpi=600)
    plt.close('all')
