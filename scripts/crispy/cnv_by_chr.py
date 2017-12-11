#!/usr/bin/env python
# Copyright (C) 2017 Emanuel Goncalves

import pickle
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from crispy import bipal_dbgd
from natsort import natsorted
from sklearn.metrics import roc_auc_score


# - Imports
# Ploidy
ploidy = pd.read_csv('data/gdsc/cell_lines_project_ploidy.csv', index_col=0)['Average Ploidy']

# sgRNA library
sgrna_lib = pd.read_csv('data/gdsc/crispr/KY_Library_v1.1_annotated.csv', index_col=0).dropna(subset=['STARTpos', 'GENES'])

# HGNC
ginfo = pd.read_csv('data/resources/hgnc/non_alt_loci_set.txt', sep='\t', index_col=1).dropna(subset=['location'])
ginfo = ginfo.assign(chr=ginfo['location'].apply(lambda x: x.replace('q', 'p').split('p')[0]))
ginfo = ginfo[ginfo['chr'].isin(sgrna_lib['CHRM'])]

# Non-expressed genes
nexp = pickle.load(open('data/gdsc/nexp_pickle.pickle', 'rb'))

# Original CRISPR fold-changes
gdsc_fc = pd.read_csv('data/crispr_gdsc_logfc.csv', index_col=0)

# Copy-number absolute counts
cnv = pd.read_csv('data/gdsc/copynumber/Gene_level_CN.txt', sep='\t', index_col=0)
cnv = cnv.loc[:, cnv.columns.isin(list(gdsc_fc))]
cnv_abs = cnv.drop(['chr', 'start', 'stop'], axis=1, errors='ignore').applymap(lambda v: int(v.split(',')[1]))


# - Overlap
genes, samples = set(gdsc_fc.index).intersection(cnv_abs.index), set(gdsc_fc).intersection(cnv_abs).intersection(nexp)
print(len(genes), len(samples))


# - Estimate chromosome copies
chr_copies = cnv_abs[cnv_abs.index.isin(ginfo.index)].replace(-1, np.nan)
chr_copies = chr_copies.groupby(ginfo.loc[chr_copies.index, 'chr']).median()
chr_copies = chr_copies.drop(['X', 'Y'])
chr_copies = chr_copies.unstack()


# - Estimate gene copies ratio
df = pd.DataFrame({c: gdsc_fc.loc[nexp[c], c] for c in gdsc_fc if c in nexp})
df = pd.concat([
    df.unstack().rename('fc'),
    cnv_abs.loc[genes, samples].unstack().rename('cnv')
], axis=1).dropna().query('cnv != -1').reset_index().rename(columns={'level_0': 'sample', 'level_1': 'gene'}).query('cnv >= 2')
df = df.assign(chr=sgrna_lib.groupby('GENES')['CHRM'].first()[df['gene']].values)
df = df.assign(cnv=[str(int(i)) if i < 10 else '10+' for i in df['cnv']])
df = df[~df['chr'].isin(['X', 'Y'])]


# - Chromosome amplification per chromossome
thres = natsorted(set(df['cnv']))

plot_df = []
for s in set(df['sample']):
    for c in set(df['chr']):
        df_ = df.query("(sample == '%s') & (chr == '%s')" % (s, c))
        for t in thres:
            if (sum(df_['cnv'] == t) >= 5) and (len(set(df_['cnv'])) > 1):
                plot_df.append({
                    'sample': s,
                    'chr': c,
                    'cnv': t,
                    'chr_cnv': int(chr_copies.loc[(s, c)]),
                    'auc': roc_auc_score((df_['cnv'] == t).astype(int), -df_['fc'])
                })
plot_df = pd.DataFrame(plot_df)
plot_df = plot_df.assign(chr_cnv=[str(int(i)) if i < 6 else '6+' for i in plot_df['chr_cnv']])
plot_df = plot_df.assign(ploidy=ploidy[plot_df['sample']].values)

# Copy-number bias per chromosome
hue_order = natsorted(set(plot_df.query("chr_cnv != '1'")['chr_cnv']))
hue_color = sns.light_palette(bipal_dbgd[0], n_colors=len(hue_order)).as_hex()

sns.boxplot('cnv', 'auc', 'chr_cnv', data=plot_df, hue_order=hue_order, order=thres, sym='', palette=hue_color, linewidth=.3)
sns.stripplot('cnv', 'auc', 'chr_cnv', data=plot_df, hue_order=hue_order, order=thres, palette=hue_color, edgecolor='white', size=.5, jitter=.2, split=True)

handles = [mpatches.Circle([.0, .0], .25, facecolor=c, label=l) for c, l in zip(*(hue_color, hue_order))]
legend = plt.legend(handles=handles, title='Chr. copies', prop={'size': 5}).get_title().set_fontsize('5')

plt.title('CRISPR/Cas9 copy-number amplification bias\nnon-expressed genes')
plt.xlabel('Copy-number')
plt.ylabel('Copy-number AUC (chromossome)')
plt.axhline(0.5, ls='--', lw=.3, alpha=.5, c=bipal_dbgd[0])
plt.gcf().set_size_inches(3, 3)
plt.savefig('reports/processing_gdsc_copy_number_bias_per_chr.png', bbox_inches='tight', dpi=600)
plt.close('all')

# Chromosome copies association with cell ploidy
order = natsorted(set(plot_df['chr_cnv']))
order_color = sns.light_palette(bipal_dbgd[0], n_colors=len(order)).as_hex()

sns.boxplot('chr_cnv', 'ploidy', data=plot_df, order=order, palette=order_color, sym='', linewidth=.3, notch=True)
sns.stripplot('chr_cnv', 'ploidy', data=plot_df, order=order, palette=order_color, edgecolor='white', linewidth=.3, size=2., jitter=.2)
plt.xlabel('# chromosome copies')
plt.ylabel('Cell line ploidy')
plt.gcf().set_size_inches(2, 4)
plt.savefig('reports/chromosome_copies_ploidy.png', bbox_inches='tight', dpi=600)
plt.close('all')
