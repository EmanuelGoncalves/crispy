#!/usr/bin/env python
# Copyright (C) 2017 Emanuel Goncalves

import os
import pickle
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from crispy import bipal_dbgd
from crispy.benchmark_plot import plot_chromosome
from sklearn.linear_model import LinearRegression, Ridge


# - Imports
# sgRNA library
sgrna_lib = pd.read_csv('data/gdsc/crispr/KY_Library_v1.1_annotated.csv', index_col=0).dropna(subset=['STARTpos', 'GENES'])

# Samplesheet
ss = pd.read_csv('data/gdsc/samplesheet.csv', index_col=0)

# Non-expressed genes
nexp = pickle.load(open('data/gdsc/nexp_pickle.pickle', 'rb'))
nexp_df = pd.DataFrame({c: {g: 1 for g in nexp[c]} for c in nexp}).fillna(0)

# Original CRISPR fold-changes
gdsc_fc = pd.read_csv('data/crispr_gdsc_logfc.csv', index_col=0)

# CRISPR bias
gdsc_kmean = pd.DataFrame({
    os.path.splitext(f)[0].replace('crispr_gdsc_crispy_', ''):
        pd.read_csv('data/crispy/' + f, index_col=0)['k_mean']
    for f in os.listdir('data/crispy/') if f.startswith('crispr_gdsc_crispy_')
}).dropna()

# MOBEMs
mobems = pd.read_csv('data/gdsc/mobems/PANCAN_simple_MOBEM.rdata.annotated.csv', index_col=0)
mobems = mobems.loc[[i for i in mobems.index if i.startswith('gain')]]

# Copy-number ratios
cnr = pd.read_csv('data/copy_number_ratio.csv')
cnr_m = pd.pivot_table(cnr, index='gene', columns='sample', values='ratio')
cnv_m = pd.pivot_table(cnr, index='gene', columns='sample', values='cnv')

# Copy-number ratios non-expressed genes
df = pd.read_csv('data/crispr_gdsc_cnratio_nexp.csv')

# Copy-number segments
cnv_seg = pd.read_csv('data/gdsc/copynumber/Summary_segmentation_data_994_lines_picnic.txt', sep='\t')
cnv_seg['chr'] = cnv_seg['chr'].replace(23, 'X').replace(24, 'Y').astype(str)


# - Overlap and Filter
samples = set(gdsc_kmean).intersection(mobems).intersection(nexp_df)

nexp_df = nexp_df[nexp_df.loc[:, samples].sum(1) >= 3]
mobems = mobems[mobems.loc[:, samples].sum(1) >= 3]

genes = set(gdsc_kmean.index).intersection(nexp_df.index)
print(len(samples), len(genes))


# -
y = (gdsc_kmean.loc[genes, samples] * nexp_df.loc[genes, samples]).T
x = mobems[y.index].T

lms = Ridge(normalize=True).fit(x, y)

lm_betas = pd.DataFrame(lms.coef_, index=y.columns, columns=x.columns)

lm_df = lm_betas.unstack().sort_values(ascending=True)
print(lm_df.head(10))


# -
y_feat, x_feat, x_gene = 'NEUROD2', 'gain:cnaPANCAN301 (CDK12,ERBB2,MED24)', 'ERBB2'

plot_df = pd.concat([
    nexp_df.loc[y_feat, samples].rename('expression'),
    gdsc_kmean.loc[y_feat, samples].rename('crispr'),
    mobems.loc[x_feat, samples].rename('genomic'),
    ss.loc[samples, 'Cancer Type'].rename('type'),
    cnr_m.loc[x_gene, samples].rename('x_ratio'),
    (cnr_m.loc[x_gene, samples].rename('x_ratio') > 2).astype(int).rename('x_ratio_bin'),
    cnv_m.loc[x_gene, samples].rename('x_cnv')
], axis=1).dropna().sort_values('crispr')

#
sns.boxplot('genomic', 'crispr', data=plot_df, color=bipal_dbgd[0], sym='')
sns.stripplot('genomic', 'crispr', data=plot_df, color=bipal_dbgd[0], edgecolor='white', linewidth=.1, size=3, jitter=.4)

plt.axhline(0, ls='-', lw=.1, c=bipal_dbgd[0])

plt.ylabel('%s CRISPR/Cas9 bias' % y_feat)
plt.xlabel(x_feat)

plt.gcf().set_size_inches(2, 3)
plt.savefig('reports/collateral_essentiality_boxplot.png', bbox_inches='tight', dpi=600)
plt.close('all')

#
hue_order = [0, 1]
hue_color = sns.light_palette(bipal_dbgd[0], n_colors=len(hue_order)).as_hex()

sns.boxplot('genomic', 'crispr', 'x_ratio_bin', data=plot_df, sym='', hue_order=hue_order, palette=hue_color)
sns.stripplot('genomic', 'crispr', 'x_ratio_bin', data=plot_df, color=bipal_dbgd[0], edgecolor='white', linewidth=.1, size=3, jitter=.2, split=True, hue_order=hue_order, palette=hue_color)

plt.axhline(0, ls='-', lw=.1, c=bipal_dbgd[0])

handles = [mpatches.Circle([.0, .0], .25, facecolor=c, label=l) for c, l in zip(*(hue_color, hue_order))]
legend = plt.legend(handles=handles, title='Copy-number ratio > 2', prop={'size': 6}).get_title().set_fontsize('6')

plt.ylabel('%s CRISPR/Cas9 bias' % y_feat)
plt.xlabel(x_feat)

plt.gcf().set_size_inches(2, 3)
plt.savefig('reports/collateral_essentiality_boxplot_ratio.png', bbox_inches='tight', dpi=600)
plt.close('all')

#
g = sns.jointplot(
    'x_ratio', 'crispr', data=plot_df, kind='reg', color='#34495e', space=0,
    marginal_kws={'hist': True, 'rug': False, 'kde': False, 'bins': 20}, annot_kws={'template': 'R={val:.2g}, pvalue={p:.1e}', 'loc': 4},
    joint_kws={'scatter_kws': {'s': 8, 'edgecolor': 'w', 'linewidth': .3, 'alpha': 1.}, 'line_kws': {'linewidth': .5}, 'truncate': True}
)

g.ax_joint.axvline(0, ls='-', lw=0.1, c='black', alpha=.2)
g.ax_joint.axhline(0, ls='-', lw=0.1, c='black', alpha=.2)
g.set_axis_labels('%s - copy-number ratio' % x_gene, '%s - CRISPR/Cas9 bias' % y_feat)
plt.gcf().set_size_inches(3, 3)
plt.savefig('reports/collateral_essentiality_jointplot_ratio.png', bbox_inches='tight', dpi=600)
plt.close('all')

#
g = sns.jointplot(
    'x_cnv', 'crispr', data=plot_df, kind='reg', color='#34495e', space=0,
    marginal_kws={'hist': True, 'rug': False, 'kde': False, 'bins': 20}, annot_kws={'template': 'R={val:.2g}, pvalue={p:.1e}', 'loc': 4},
    joint_kws={'scatter_kws': {'s': 8, 'edgecolor': 'w', 'linewidth': .3, 'alpha': 1.}, 'line_kws': {'linewidth': .5}, 'truncate': True}
)

g.ax_joint.axvline(0, ls='-', lw=0.1, c='black', alpha=.2)
g.ax_joint.axhline(0, ls='-', lw=0.1, c='black', alpha=.2)
g.set_axis_labels('%s - copy-number ratio' % x_gene, '%s - CRISPR/Cas9 bias' % y_feat)
plt.gcf().set_size_inches(3, 3)
plt.savefig('reports/collateral_essentiality_jointplot_cnv.png', bbox_inches='tight', dpi=600)
plt.close('all')

# -
c = 'UACC-893'

plot_df = pd.concat([
    gdsc_fc[c].rename('fc'),
    gdsc_kmean[c].rename('bias'),
    cnv_m[c].rename('cnv'),
    sgrna_lib.groupby('GENES')['CHRM'].first().rename('chr'),
    sgrna_lib.groupby('GENES')['STARTpos'].min().rename('pos'),
], axis=1).dropna().query('cnv != -1').sort_values(['chr', 'pos'])
plot_df = plot_df.query("chr == '%s'" % plot_df.loc[y_feat, 'chr'])

plot_df_seg = cnv_seg.query("cellLine == '%s'" % c).query("chr == '%s'" % plot_df.loc[y_feat, 'chr'])

# plot_df = plot_df.query('(pos > 3.5e7) & (pos < 4e7)')

ax = plot_chromosome(plot_df['pos'], plot_df['fc'].rename('Original fold-change'), plot_df['bias'].rename('Copy-number bias'), seg=plot_df_seg, highlight=[y_feat, 'ERBB2', 'MDM2', 'MED24'])
plt.title('Chromosome %s in %s' % (plot_df.loc[y_feat, 'chr'], c))
plt.ylabel('')
# plt.xlabel('')
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.gcf().set_size_inches(4, 2)
plt.savefig('reports/collateral_essentiality_chromosome_plot.png', bbox_inches='tight', dpi=600)
plt.close('all')
