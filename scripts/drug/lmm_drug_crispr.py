#!/usr/bin/env python
# Copyright (C) 2018 Emanuel Goncalves

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from crispy import bipal_dbgd
from limix.stats import qvalues, linear_kinship
from limix.qtl import qtl_test_lmm
from scipy.stats.stats import rankdata
from limix_core.util.preprocess import gaussianize


# - Imports
# CRISPR
crispr = pd.read_csv('data/gdsc/crispr/crisprcleanr_gene.csv', index_col=0).dropna()

# Drug response
d_response = pd.read_csv('data/gdsc/drug_single/drug_ic50_merged_matrix.csv', index_col=[0, 1, 2], header=[0, 1])
d_response.columns = d_response.columns.droplevel(0)

# Samplesheet
ss = pd.read_csv('data/gdsc/samplesheet.csv', index_col=0).dropna(subset=['Cancer Type'])

# Essential genes
essential = list(pd.read_csv('data/resources/curated_BAGEL_essential.csv', sep='\t')['gene'])


# - Kinship
kinship = pd.DataFrame(linear_kinship(crispr.T.values), index=crispr.columns, columns=crispr.columns)


# - Overlap
samples = list(set(d_response).intersection(crispr).intersection(ss.index))
d_response, crispr = d_response[samples], crispr[samples]
print('Samples: %d' % len(samples))


# - Filter
# Drug response
d_response = d_response[d_response.count(1) > d_response.shape[1] * .85]
d_response = d_response.loc[(d_response < d_response.mean().mean()).sum(1) >= 3]

# CRISPR
crispr = crispr[(crispr < crispr.loc[essential].mean().mean()).sum(1) < (crispr.shape[1] * .5)]
crispr = crispr[(crispr.abs() >= 1.5).sum(1) >= 3]
crispr = crispr.drop(essential, axis=0, errors='ignore')


# - lmm: drug ~ crispr + tissue
print('CRISPR genes: %d, Drug: %d' % (len(set(crispr.index)), len(set(d_response.index))))

# Linear regressions
lmm_cd = pd.DataFrame()

# d = (1047, 'Nutlin-3a (-)', 'RS')
for d in d_response.index:
    # Discard samples with NaN for drug measurements
    y = d_response.loc[[d], samples].T.dropna()
    x = crispr[y.index].T

    # Random effects
    covs = pd.get_dummies(ss.loc[samples, ['Cancer Type']]).loc[y.index].astype(float)

    # Normalise
    x = pd.DataFrame(gaussianize(x.values), index=x.index, columns=x.columns)

    # Linear regression
    lmm = qtl_test_lmm(pheno=y.values, snps=x.loc[y.index].values, K=kinship.loc[y.index, y.index].values)

    # Extract solutions
    res = pd.DataFrame({
        'drug_id': d[0], 'drug_name': d[1], 'drug_screen': d[2],
        'crispr': x.columns,
        'beta': lmm.getBetaSNP().ravel(), 'pvalue': lmm.getPv().ravel(),
        'rank_beta': rankdata(lmm.getBetaSNP().ravel()), 'rank_pval': rankdata(lmm.getPv().ravel()),
        'len': len(y.index)
    })
    lmm_cd = lmm_cd.append(res)

lmm_cd['qvalue'] = qvalues(lmm_cd['pvalue'].values)
lmm_cd.sort_values('qvalue').to_csv('data/drug/lmm_drug_crispr.csv', index=False)
# lmm_cd = pd.read_csv('data/lmm_crispr_drug_associations.csv')
print(lmm_cd[lmm_cd['qvalue'] < 0.05].sort_values('qvalue'))


# - Volcano
plot_df = lmm_cd.copy()
plot_df['signif'] = ['*' if i < 0.05 else '-' for i in plot_df['qvalue']]
plot_df['log10_qvalue'] = -np.log10(plot_df['qvalue'])

pal = dict(zip(*(['*', '-'], sns.light_palette(bipal_dbgd[0], 3, reverse=True).as_hex()[:-1])))

g = sns.lmplot(
    x='beta', y='log10_qvalue', data=plot_df, hue='signif', fit_reg=False, palette=pal, legend=False,
    scatter_kws={'edgecolor': 'w', 'lw': .1, 's': 10, 'alpha': 0.4}
)
g.axes[0, 0].set_ylim(0,)
g.despine(right=False, top=False)

# Add FDR threshold lines
plt.text(plt.xlim()[0]*.98, -np.log10(.05) * 1.01, 'FDR 5%', ha='left', color=pal['*'], alpha=0.65, fontsize=5)
plt.axhline(-np.log10(.05), c=pal['*'], ls='--', lw=.3, alpha=.7)

# Add axis lines
plt.axvline(0, c=bipal_dbgd[0], lw=.1, ls='-', alpha=.3)

# Add axis labels and title
plt.title('Drug ~ CRISPR', fontsize=8, fontname='sans-serif')
plt.xlabel('Beta', fontsize=8, fontname='sans-serif')
plt.ylabel('q-value (-log10)', fontsize=8, fontname='sans-serif')

# Save plot
plt.gcf().set_size_inches(2., 4.)
plt.savefig('reports/drug/lmm_crispr_volcano.png', bbox_inches='tight', dpi=300)
plt.close('all')
