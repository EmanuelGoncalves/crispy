#!/usr/bin/env python
# Copyright (C) 2017 Emanuel Goncalves

import os
import pickle
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from crispy import bipal_dbgd
from crispy.benchmark_plot import plot_chromosome
from sklearn.linear_model import LinearRegression, Ridge

# - Imports
# sgRNA library
sgrna_lib = pd.read_csv('data/gdsc/crispr/KY_Library_v1.1_annotated.csv', index_col=0).dropna(subset=['STARTpos', 'GENES'])

# Samplesheet
ss = pd.read_csv('data/gdsc/samplesheet.csv', index_col=0)

# CRISPR bias
ccrispy = pd.read_csv('data/crispr_gdsc_crispy.csv', index_col=0)
c_gdsc = pd.DataFrame({
    os.path.splitext(f)[0].replace('crispr_gdsc_crispy_', ''):
        pd.read_csv('data/crispy/' + f, index_col=0)['k_mean']
    for f in os.listdir('data/crispy/') if f.startswith('crispr_gdsc_crispy_')
}).dropna()

# Non-expressed genes
nexp = pickle.load(open('data/gdsc/nexp_pickle.pickle', 'rb'))


# - Overlap
samples = set(c_gdsc).intersection(nexp)
print(len(samples))

cnexp = pd.DataFrame({k: c_gdsc.loc[nexp[k], k] for k in samples})


# - Copy-number
# Copy-number absolute counts
cnv = pd.read_csv('data/gdsc/copynumber/Gene_level_CN.txt', sep='\t', index_col=0)
cnv = cnv.loc[:, samples.intersection(cnv)]
cnv_abs = cnv.applymap(lambda v: int(v.split(',')[0]))

# Segments
cnv_seg = pd.read_csv('data/gdsc/copynumber/Summary_segmentation_data_994_lines_picnic.txt', sep='\t')
cnv_seg = cnv_seg[cnv_seg['cellLine'].isin(samples)]
cnv_seg['chr'] = cnv_seg['chr'].replace(23, 'X').replace(24, 'Y').astype(str)


# -
plot_df = pd.Series({k: len(nexp[k]) for k in samples}, name='count').sort_values().reset_index().rename(columns={'index': 'sample'})

sns.barplot('count', 'sample', data=plot_df, orient='h', color=bipal_dbgd[0])
plt.show()


# -
plot_df = cnexp.corr()

ctype = set(ss.loc[plot_df.index, 'Cancer Type'])
ctype_cmap = pd.Series(dict(zip(*(ctype, sns.color_palette('tab20c', n_colors=len(ctype)).as_hex()))))

sns.clustermap(plot_df, col_colors=ctype_cmap[ss.loc[plot_df.columns, 'Cancer Type']].values)
plt.show()


# -
x = pd.get_dummies(ss.loc[samples, 'Cancer Type'])

ctype_lm = {}
# g = 'ZSWIM2'
for g in cnexp.index:
    ys = cnexp.loc[g, samples].dropna()

    xs = x.loc[ys.index]
    xs = xs.loc[:, xs.sum() >= 3]

    if len(ys) >= 3:
        ctype_lm[g] = {c: LinearRegression().fit(xs[[c]], ys).coef_[0] for c in xs}

ctype_lm = pd.DataFrame(ctype_lm).T
print(ctype_lm.min())


# -
g = ctype_lm['Breast Carcinoma'].argmin()
plot_df = pd.concat([cnexp.loc[g].rename('crispr'), ss.loc[samples, 'Cancer Type']], axis=1).dropna()

# order = list(plot_df.groupby('Cancer Type')['crispr'].mean().sort_values().index)

sns.boxplot('crispr', 'Cancer Type', data=plot_df, color=bipal_dbgd[0], orient='h', sym='', palette=ctype_cmap)
sns.stripplot('crispr', 'Cancer Type', data=plot_df, color=bipal_dbgd[0], orient='h', edgecolor='k', linewidth=.1, size=5, jitter=.4, palette=ctype_cmap)
plt.show()


# -
t = 'Breast Carcinoma'
c = cnexp.loc[ctype_lm[t].argmin()].sort_values().head().index[0]

df = pd.concat([
    ccrispy[c].rename('fc'),
    c_gdsc[c].rename('bias'),
    cnv_abs[c].rename('cnv'),
    sgrna_lib.groupby('GENES')['CHRM'].first().rename('chr'),
    sgrna_lib.groupby('GENES')['STARTpos'].min().rename('pos'),
], axis=1).dropna().query('cnv != -1').sort_values(['chr', 'pos'])
df = df.query("chr == '%s'" % df.loc[g, 'chr'])

df_seg = cnv_seg.query("cellLine == '%s'" % c).query("chr == '%s'" % df.loc[g, 'chr'])

ax = plot_chromosome(df['pos'], df['fc'], df['bias'], seg=df_seg, highlight=[g, 'BRCA1', 'ERBB2'])
# plt.show()
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.gcf().set_size_inches(3, 2)
plt.savefig('reports/synthetic_essentiality_chromosome_plot.png', bbox_inches='tight', dpi=600)
plt.close('all')
