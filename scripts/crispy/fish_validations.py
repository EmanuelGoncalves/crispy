#!/usr/bin/env python
# Copyright (C) 2017 Emanuel Goncalves

import pandas as pd
from crispy.ratio import _GFF_HEADERS


# - Imports
# Samplesheet
ss = pd.read_csv('data/gdsc/samplesheet.csv', index_col=0)

# Gene genomic infomration
ginfo = pd.read_csv('data/gencode.v27lift37.annotation.sorted.gff', sep='\t', names=_GFF_HEADERS, index_col='feature')

# WGS
wgs_cnv = pd.read_csv('data/crispy_gene_copy_number_wgs.csv', index_col=0)
wgs_chrm = pd.read_csv('data/crispy_chr_copy_number_wgs.csv', index_col=0)
wgs_ratios = pd.read_csv('data/crispy_gene_copy_number_ratio_wgs.csv', index_col=0)
wgs_ploidy = pd.read_csv('data/crispy_ploidy_wgs.csv', index_col=0, names=['sample', 'ploidy'])['ploidy']


# - Overlap
genes, samples = set(wgs_cnv.index), set(wgs_cnv)
print(len(genes), len(samples))


# - Build data-frame
def bin_cnv(value, thresold):
    return str(int(value)) if value < thresold else '%s+' % thresold


df = pd.concat([
    wgs_cnv.loc[genes, samples].unstack().rename('cnv'),
    wgs_ratios.loc[genes, samples].unstack().rename('ratio')

], axis=1).dropna()

df.index.rename(['sample', 'gene'], inplace=True)

df = df.reset_index()

df = df.assign(chr=ginfo.loc[df['gene'], 'chr'].values)

df = df.assign(chr_cnv=wgs_chrm.unstack().loc[list(map(tuple, df[['sample', 'chr']].values))].values)

df = df.assign(cnv_bin=[bin_cnv(i, 10) for i in df['cnv']])

df = df.assign(ratio_bin=[bin_cnv(i, 4) for i in df['ratio']])

df = df.assign(chr_cnv_bin=[bin_cnv(i, 6) for i in df['chr_cnv']])

df = df.assign(ploidy=wgs_ploidy.loc[df['sample']].values)
print(df.shape)


# - Targets for FISH validation
plot_df = []
for g in ['EGFR', 'MYC']:
    df_ = df.query("gene == '%s'" % g).set_index('sample')
    df_ = pd.concat([df_, ss.loc[df_.index, 'Cancer Type']], axis=1).sort_values('ratio')
    plot_df.append(df_)
plot_df = pd.concat(plot_df).sort_values(['gene', 'cnv'])
plot_df.to_csv('data/crispy_fish_validations.csv')
print(plot_df)
