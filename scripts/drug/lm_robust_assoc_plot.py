#!/usr/bin/env python
# Copyright (C) 2018 Emanuel Goncalves

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from crispy import bipal_dbgd
from scipy.stats import pearsonr

# - Imports
# <Drug, CRISPR> ~ Genomic association
lm_res = pd.read_csv('data/drug/lm_drug_crispr_genomic.csv')

# Samplesheet
ss = pd.read_csv('data/gdsc/samplesheet.csv', index_col=0).dropna(subset=['Cancer Type'])

# Growth rate
growth = pd.read_csv('data/gdsc/growth_rate.csv', index_col=0)

# CRISPR
crispr = pd.read_csv('data/gdsc/crispr/crisprcleanr_gene.csv', index_col=0).dropna()
crispr = crispr.subtract(crispr.mean()).divide(crispr.std())

# Drug response
d_response = pd.read_csv('data/gdsc/drug_single/drug_ic50_merged_matrix.csv', index_col=[0, 1, 2], header=[0, 1])
d_response.columns = d_response.columns.droplevel(0)

# MOBEMs
mobems = pd.read_csv('data/gdsc/mobems/PANCAN_simple_MOBEM.rdata.annotated.csv', index_col=0)
mobems.index.name = 'genomic'


# - Overlap
samples = list(set(d_response).intersection(crispr).intersection(ss.index).intersection(growth.index).intersection(mobems))
d_response, crispr, mobems = d_response[samples], crispr[samples], mobems[samples]
print('Samples: %d' % len(samples))


# - Robust pharmacogenomics associations
lm_robust = lm_res[(lm_res['crispr_lr_fdr'] < 0.1) & (lm_res['drug_lr_fdr'] < 0.1)].sort_values('crispr_lr_fdr')
print(lm_robust)

lm_robust_either = lm_res[(lm_res['crispr_lr_fdr'] < 0.1) | (lm_res['drug_lr_fdr'] < 0.1)].sort_values('crispr_lr_fdr')
print(lm_robust_either)


# - Volcano
for name in ['drug', 'crispr']:
    plot_df = lm_res.copy()
    plot_df['signif'] = ['*' if i < 0.05 else '-' for i in plot_df['%s_lr_fdr' % name]]
    plot_df['log10_qvalue'] = -np.log10(plot_df['%s_lr_pval' % name])

    pal = dict(zip(*(['*', '-'], sns.light_palette(bipal_dbgd[0], 3, reverse=True).as_hex()[:-1])))

    # Scatter
    for i in ['*', '-']:
        plot_df_ = plot_df.query("(signif == '%s')" % i)
        plt.scatter(plot_df_['%s_beta' % name], plot_df_['log10_qvalue'], edgecolor='white', lw=.1, s=3, alpha=.5, c=pal[i])

    # Add FDR threshold lines
    yy = plot_df.loc[(plot_df['%s_lr_fdr' % name] - 0.05).abs().sort_values().index[0], '%s_lr_pval' % name]
    plt.text(plt.xlim()[0]*.98, -np.log10(yy) * 1.01, 'FDR 5%', ha='left', color=pal['*'], alpha=0.65, fontsize=5)
    plt.axhline(-np.log10(yy), c=pal['*'], ls='--', lw=.3, alpha=.7)

    # Add axis lines
    plt.axvline(0, c=bipal_dbgd[0], lw=.1, ls='-', alpha=.3)

    # Add axis labels and title
    plt.title('Linear regression: %s ~ GENOMIC' % name.upper(), fontsize=8, fontname='sans-serif')
    plt.xlabel('Model coefficient (beta)', fontsize=8, fontname='sans-serif')
    plt.ylabel('Log-ratio test p-value (-log10)', fontsize=8, fontname='sans-serif')

    # Save plot
    plt.gcf().set_size_inches(2., 4.)
    plt.savefig('reports/drug/lm_%s_genomic_volcano.png' % name, bbox_inches='tight', dpi=300)
    plt.close('all')


# - Plot
i = 23

drug_id, drug_name, drug_screen, crispr_gene, genomic = lm_robust.loc[i, ['drug_DRUG_ID', 'drug_DRUG_NAME', 'drug_VERSION', 'crispr_GENES', 'genomic']].values

plot_df = pd.concat([
    d_response.loc[(drug_id, drug_name, drug_screen), samples].rename('drug'),
    mobems.loc[genomic, samples].rename('genomic'),
    crispr.loc[crispr_gene, samples].rename('crispr'),
    ss.loc[samples, ['Cancer Type', 'Microsatellite']],
    (ss['Microsatellite'] == 'MSI-H').astype(int).rename('msi')
], axis=1).dropna()


# Corrplot
def marginal_boxplot(a, vertical=False, **kws):
    ax = sns.boxplot(x=z, y=y, orient='v', **kws) if vertical else sns.boxplot(x=x, y=z, orient='h', **kws)
    ax.set_ylabel('')
    ax.set_xlabel('')


x, y, z = 'crispr', 'drug', 'genomic'

g = sns.JointGrid(x, y, plot_df, space=0, ratio=8)
g.plot_marginals(marginal_boxplot, palette=bipal_dbgd, data=plot_df, linewidth=.3, fliersize=1, notch=False, saturation=1.0)

scatter_kws = {'s': 20, 'edgecolor': 'w', 'linewidth': .5, 'alpha': .8}
sns.regplot(x=x, y=y, data=plot_df, color=bipal_dbgd[0], truncate=True, fit_reg=True, scatter_kws=scatter_kws, line_kws={'linewidth': .5}, ax=g.ax_joint)
sns.regplot(x=x, y=y, data=plot_df.query('%s == 1' % z), color=bipal_dbgd[1], truncate=True, fit_reg=False, scatter_kws=scatter_kws, ax=g.ax_joint)

g.annotate(pearsonr, template='Pearson={val:.2g}, p={p:.1e}', loc=4)

g.ax_joint.axhline(0, ls='-', lw=0.3, c='black', alpha=.2)
g.ax_joint.axvline(0, ls='-', lw=0.3, c='black', alpha=.2)

g.set_axis_labels('%s (fold-change)' % crispr_gene, '%s (screen=%s, ln IC50)' % (drug_name, drug_screen))

handles = [mpatches.Circle([.0, .0], .25, facecolor=c, label='Yes' if t else 'No') for t, c in bipal_dbgd.items()]
g.ax_marg_y.legend(handles=handles, title='', loc='center left', bbox_to_anchor=(1, 0.5))

plt.suptitle(genomic if z != 'msi' else 'MSI-H', y=1.05, fontsize=8)

plt.gcf().set_size_inches(2, 2)
plt.savefig('reports/drug/lmm_crispr_drug_genomic.png', bbox_inches='tight', dpi=600)
plt.close('all')
