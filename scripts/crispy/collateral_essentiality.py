#!/usr/bin/env python
# Copyright (C) 2017 Emanuel Goncalves

import os
import re
import pickle
import numpy as np
import pandas as pd
import seaborn as sns
import itertools as it
import matplotlib.pyplot as plt
from natsort import natsorted
from crispy import bipal_dbgd
from limix.stats import qvalues
from limix.qtl import qtl_test_lm


# - Imports
# Samplesheet
ss = pd.read_csv('data/gdsc/samplesheet.csv', index_col=0)

# Gene HGNC info
ginfo = pd.read_csv('data/crispy_hgnc_info.csv', index_col=0)

# Copy-number
cnv = pd.read_csv('data/crispy_gene_copy_number.csv', index_col=0)

# Ploidy
ploidy = pd.read_csv('data/crispy_sample_ploidy.csv', index_col=0)['ploidy']

# Copy-number ratios
cnv_ratios = pd.read_csv('data/crispy_copy_number_ratio.csv', index_col=0).replace(-1, np.nan)

# Non-expressed genes
nexp = pickle.load(open('data/gdsc/nexp_pickle.pickle', 'rb'))
nexp = pd.DataFrame({c: {g: 1 for g in nexp[c]} for c in nexp}).fillna(0)

# CRISPR bias
gdsc_kmean = pd.DataFrame({
    os.path.splitext(f)[0].replace('crispr_gdsc_crispy_', ''):
        pd.read_csv('data/crispy/' + f, index_col=0)['k_mean']
    for f in os.listdir('data/crispy/') if f.startswith('crispr_gdsc_crispy_')
}).dropna()

# Cancer gene census list
cgenes = pd.read_csv('data/resources/Census_allThu Dec 21 15_43_09 2017.tsv', sep='\t')


# - Overlap
samples = set(gdsc_kmean).intersection(nexp).intersection(cnv_ratios).intersection(ploidy.index)
print(len(samples))


# - Recurrent amplified genes
g_amp = pd.DataFrame({s: (cnv[s] > 8 if ploidy[s] > 2.7 else cnv[s] > 4).astype(int) for s in samples})
g_amp = g_amp.loc[g_amp.index.isin(cnv_ratios.index) & g_amp.index.isin(cgenes['Gene Symbol'])]
g_amp = g_amp[g_amp.sum(1) >= 3]


# - Recurrent non-expressed genes
g_nexp = nexp.loc[gdsc_kmean.index, samples].dropna()
g_nexp = g_nexp[g_nexp.sum(1) >= (g_nexp.shape[1] * .5)]


# - Linear regressions
# Build matrices
xs = cnv_ratios.loc[g_amp.index, samples].T
xs = xs.loc[:, (xs > 2).sum() >= 3]
xs = xs.replace(np.nan, 1)

ys = gdsc_kmean.loc[g_nexp.index, xs.index].T

# Run fit
lm = qtl_test_lm(xs.values, ys.values)

# Extract info
c_gene, g_gene = zip(*list(it.product(ys.columns, xs.columns)))

# Build pandas data-frame
lmm_df = pd.DataFrame({
    'gene_crispr': c_gene,
    'gene_cnvabs': g_gene,
    'lm_beta': lm.getBetaSNP().ravel(),
    'lm_pvalue': lm.getPv().ravel(),
})

lmm_df = lmm_df[lmm_df['gene_crispr'] != lmm_df['gene_cnvabs']]

lmm_df = lmm_df.assign(chr_crispr=ginfo.loc[lmm_df['gene_crispr'], 'chr'].values)
lmm_df = lmm_df.assign(chr_cnvabs=ginfo.loc[lmm_df['gene_cnvabs'], 'chr'].values)

lmm_df = lmm_df.assign(lm_qvalue=qvalues(lmm_df['lm_pvalue'].values)).sort_values('lm_qvalue')
print(lmm_df)


# - Plot
idx = 72246
y_feat, x_feat = lmm_df.loc[idx, 'gene_crispr'], lmm_df.loc[idx, 'gene_cnvabs']

plot_df = pd.concat([
    gdsc_kmean.loc[y_feat, samples].rename('crispy'),
    cnv.loc[x_feat, samples].rename('cnv'),
    cnv_ratios.loc[x_feat, samples].rename('ratio'),
    g_amp.loc[x_feat].rename('amp'),
    nexp.loc[y_feat].replace({1.0: 'No', 0.0: 'Yes'}).rename('expressed'),
    ss['Cancer Type'],
], axis=1).dropna().sort_values('crispy')
plot_df = plot_df.assign(ratio_bin=[str(i) if i < 4 else '4+' for i in plot_df['ratio'].round(0).astype(int)])

order = natsorted(set(plot_df['ratio_bin']))
pal = {'No': bipal_dbgd[1], 'Yes': bipal_dbgd[0]}

# Boxplot
sns.boxplot('ratio_bin', 'crispy', data=plot_df, order=order, color=bipal_dbgd[0], sym='', linewidth=.3)
sns.stripplot('ratio_bin', 'crispy', 'expressed', data=plot_df, order=order, palette=pal, edgecolor='white', linewidth=.1, size=2, jitter=.1, alpha=.8)

plt.axhline(0, ls='-', lw=.1, c=bipal_dbgd[0])

plt.ylabel('%s\nCRISPR/Cas9 cop-number bias' % y_feat)
plt.xlabel('%s\nCopy-number ratio' % x_feat)
plt.title('Collateral essentiality\nbeta=%.1f; q-value=%.1e' % (lmm_df.loc[idx, 'lm_beta'], lmm_df.loc[idx, 'lm_qvalue']))

legend = plt.legend(title='Expressed', prop={'size': 8})
legend.get_title().set_fontsize('8')

plt.gcf().set_size_inches(2, 3)
plt.savefig('reports/collateral_essentiality_boxplot.png', bbox_inches='tight', dpi=600)
plt.close('all')
