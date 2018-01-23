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
from crispy.ratio import GFF_HEADERS
from sklearn.metrics import roc_auc_score


# - Imports
# Gene genomic infomration
ginfo = pd.read_csv('data/gencode.v27lift37.annotation.sorted.gff', sep='\t', names=GFF_HEADERS, index_col='feature')

# Non-expressed genes
nexp = pickle.load(open('data/gdsc/nexp_pickle.pickle', 'rb'))

# Original CRISPR fold-changes
gdsc_fc = pd.read_csv('data/crispr_gdsc_logfc.csv', index_col=0)

# Copy-number
cnv = pd.read_csv('data/crispy_gene_copy_number_snp.csv', index_col=0)

# Chromosome copies
chrm = pd.read_csv('data/crispy_chr_copy_number_snp.csv', index_col=0)

# Ploidy
ploidy = pd.read_csv('data/crispy_ploidy_snp.csv', index_col=0, names=['sample', 'ploidy'])['ploidy']


# - Overlap
genes, samples = set(gdsc_fc.index).intersection(cnv.index), set(gdsc_fc).intersection(cnv).intersection(nexp)
print(len(genes), len(samples))


# - Build copy-number data-frame
# CRISPR fold-changes of non-expressed genes
df = pd.DataFrame({c: gdsc_fc.loc[nexp[c].intersection(genes), c] for c in samples})
df.index.rename('gene', inplace=True)
df.columns.rename('sample', inplace=True)

# Concat copy-number measurements
df = pd.concat([
    df.unstack().rename('fc'),
    cnv.loc[df.index, df.columns].unstack().rename('cnv'),
], axis=1).dropna().reset_index().query('cnv >= 2')

df = df.assign(cnv=[str(int(i)) if i < 10 else '10+' for i in df['cnv']])

df = df.assign(chr=ginfo.loc[df['gene'], 'chr'].values)


# - Enrichment analysis of copy-number alterations
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
                    'auc': roc_auc_score((df_['cnv'] == t).astype(int), -df_['fc'])
                })
plot_df = pd.DataFrame(plot_df)
plot_df = plot_df.assign(chr_cnv=chrm.unstack().loc[list(map(tuple, plot_df[['sample', 'chr']].values))].values)
plot_df = plot_df.assign(chr_cnv=[str(int(i)) if i < 6 else '6+' for i in plot_df['chr_cnv']])
plot_df = plot_df.assign(ploidy=ploidy[plot_df['sample']].values)


# - Plot
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
plt.savefig('reports/crispy/copy_number_bias_per_chr_gdsc.png', bbox_inches='tight', dpi=600)
plt.close('all')

# Chromosome copies association with cell ploidy
order = natsorted(set(plot_df['chr_cnv']))
order_color = sns.light_palette(bipal_dbgd[0], n_colors=len(order)).as_hex()

sns.boxplot('chr_cnv', 'ploidy', data=plot_df, order=order, palette=order_color, sym='', linewidth=.3, notch=True)
sns.stripplot('chr_cnv', 'ploidy', data=plot_df, order=order, palette=order_color, edgecolor='white', linewidth=.3, size=2., jitter=.2)
plt.xlabel('# chromosome copies')
plt.ylabel('Cell line ploidy')
plt.gcf().set_size_inches(2, 4)
plt.savefig('reports/crispy/ploidy_chromosome_copies_boxplot_gdsc.png', bbox_inches='tight', dpi=600)
plt.close('all')
