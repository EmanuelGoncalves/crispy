#!/usr/bin/env python
# Copyright (C) 2017 Emanuel Goncalves

import os
import pandas as pd
from crispy.ratio import _GFF_HEADERS


# - Imports
# Samplesheet
ss = pd.read_csv('data/gdsc/samplesheet.csv', index_col=0)

# Gene genomic infomration
ginfo = pd.read_csv('data/gencode.v27lift37.annotation.sorted.gff', sep='\t', names=_GFF_HEADERS, index_col='feature')

# GDSC
c_gdsc_fc = pd.read_csv('data/crispr_gdsc_logfc.csv', index_col=0)

c_gdsc = pd.DataFrame({
    os.path.splitext(f)[0].replace('crispr_gdsc_crispy_', ''):
        pd.read_csv('data/crispy/' + f, index_col=0)['k_mean']
    for f in os.listdir('data/crispy/') if f.startswith('crispr_gdsc_crispy_')
}).dropna()

# Copy-number
cnv = pd.read_csv('data/crispy_gene_copy_number_snp.csv', index_col=0)

# Chromosome copies
chrm = pd.read_csv('data/crispy_chr_copy_number_snp.csv', index_col=0)

# Copy-number
ratios = pd.read_csv('data/crispy_gene_copy_number_ratio_snp.csv', index_col=0)

# Ploidy
ploidy = pd.read_csv('data/crispy_ploidy_snp.csv', index_col=0, names=['sample', 'ploidy'])['ploidy']

# WGS
wgs_cnv = pd.read_csv('data/crispy_gene_copy_number_wgs.csv', index_col=0)
wgs_chrm = pd.read_csv('data/crispy_chr_copy_number_wgs.csv', index_col=0)
wgs_ratios = pd.read_csv('data/crispy_gene_copy_number_ratio_wgs.csv', index_col=0)
wgs_ploidy = pd.read_csv('data/crispy_ploidy_wgs.csv', index_col=0, names=['sample', 'ploidy'])['ploidy']


# - Overlap
genes, samples = set(c_gdsc.index).intersection(cnv.index).intersection(wgs_cnv.index), set(c_gdsc).intersection(cnv).intersection(wgs_cnv)
print(len(genes), len(samples))


# - Build data-frame
df = pd.concat([
    c_gdsc_fc.loc[genes, samples].unstack().rename('logfc'),
    c_gdsc.loc[genes, samples].unstack().rename('crispy'),
    cnv.loc[genes, samples].unstack().rename('cnv'),
    ratios.loc[genes, samples].unstack().rename('ratio')
], axis=1).dropna()

df = df.reset_index().rename(columns={'level_0': 'sample', 'GENES': 'gene'})

df = df.assign(chr=ginfo.loc[df['gene'], 'chr'].values)
df = df.assign(chr_cnv=chrm.unstack().loc[list(map(tuple, df[['sample', 'chr']].values))].values)

df = df.assign(cnv_bin=[str(int(i)) if i < 10 else '10+' for i in df['cnv']])
df = df.assign(ratio_bin=[str(i) if i < 4 else '4+' for i in df['ratio'].round(0).astype(int)])
df = df.assign(chr_cnv_bin=[str(int(i)) if i < 6 else '6+' for i in df['chr_cnv']])
print(df.shape)


# - Targets for FISH validation
# g_target = ['ERBB2', 'MYC']
plot_df = df.query("gene == 'MYC'").set_index('sample')
plot_df = pd.concat([plot_df, ploidy.loc[plot_df.index], ss.loc[plot_df.index, 'Cancer Type']], axis=1).sort_values('crispy')
print(plot_df.sort_values('cnv'))
