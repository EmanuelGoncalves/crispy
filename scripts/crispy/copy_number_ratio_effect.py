#!/usr/bin/env python
# Copyright (C) 2017 Emanuel Goncalves

import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from natsort import natsorted
from crispy import bipal_dbgd


# - Imports
# Essential genes
essential = pd.read_csv('data/resources/curated_BAGEL_essential.csv', sep='\t')['gene'].rename('essential')

# Samplesheet
ss = pd.read_csv('data/gdsc/samplesheet.csv', index_col=0)

# Gene HGNC info
ginfo = pd.read_csv('data/crispy_hgnc_info.csv', index_col=0)

# GDSC
c_gdsc_fc = pd.read_csv('data/crispr_gdsc_logfc.csv', index_col=0)

c_gdsc = pd.DataFrame({
    os.path.splitext(f)[0].replace('crispr_gdsc_crispy_', ''):
        pd.read_csv('data/crispy/' + f, index_col=0)['k_mean']
    for f in os.listdir('data/crispy/') if f.startswith('crispr_gdsc_crispy_')
}).dropna()

# Copy-number absolute counts
cnv_abs = pd.read_csv('data/crispy_gene_copy_number.csv', index_col=0)

# Chromosome copies
chr_copies = pd.read_csv('data/crispy_chrm_copy_number.csv', index_col=(0, 1))

# Gene copy-number ratio
cnv_ratios = pd.read_csv('data/crispy_copy_number_ratio.csv', index_col=0).replace(-1, np.nan)

# Cell line
ploidy = pd.read_csv('data/crispy_sample_ploidy.csv', index_col=0)['ploidy']


# - Overlap
genes, samples = set(c_gdsc.index).intersection(cnv_abs.index), set(c_gdsc).intersection(cnv_abs)
print(len(genes), len(samples))


# - Build data-frame
df = pd.concat([
    c_gdsc_fc.loc[genes, samples].unstack().rename('logfc'),
    c_gdsc.loc[genes, samples].unstack().rename('crispy'),
    cnv_abs.loc[genes, samples].unstack().rename('cnv'),
    cnv_ratios.loc[genes, samples].unstack().rename('ratio')
], axis=1).dropna()

df = df.reset_index().rename(columns={'level_0': 'sample', 'GENES': 'gene'})

df = df.assign(chr=ginfo.loc[df['gene'], 'chr'].values)

df = df.assign(chr_cnv=chr_copies.loc[[(s, int(c)) for s, c in df[['sample', 'chr']].values]].round(0).astype(int).values)

df = df.assign(ratio_bin=[str(i) if i < 4 else '4+' for i in df['ratio'].round(0).astype(int)])

df = df.assign(essential=df['gene'].isin(set(essential)).astype(int))

print(df.shape)


# - Targets for FISH validation
# g_target = ['ERBB2', 'MYC']
plot_df = df.query("gene == 'ERBB2'").set_index('sample')
plot_df = pd.concat([plot_df, ploidy.loc[plot_df.index], ss.loc[plot_df.index, 'Cancer Type']], axis=1).sort_values('crispy')
print(plot_df.sort_values('cnv'))


# - Copy-number ratio association with CRISPR/Cas9 bias: essential genes
order = natsorted(set(df['ratio_bin']))

sns.boxplot('logfc', 'ratio_bin', 'essential', data=df, orient='h', linewidth=.3, fliersize=1, order=order, palette=bipal_dbgd, notch=True)
plt.axvline(0, lw=.1, ls='-', c=bipal_dbgd[0])

plt.title('Gene copy-number ratio\neffect on CRISPR/Cas9 bias')
plt.xlabel('CRISPR/Cas9 fold-change bias (log2)')
plt.ylabel('Copy-number ratio')
plt.gcf().set_size_inches(2, 2)
plt.savefig('reports/ratio_boxplot_essential.png', bbox_inches='tight', dpi=600)
plt.close('all')
