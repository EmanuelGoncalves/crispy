#!/usr/bin/env python
# Copyright (C) 2017 Emanuel Goncalves

import pandas as pd
from crispy.utils import bin_cnv
from crispy.ratio import GFF_HEADERS


# - Imports
# Samplesheet
ss = pd.read_csv('data/gdsc/samplesheet.csv', index_col=0)

# Gene genomic infomration
ginfo = pd.read_csv('data/gene_sets/gencode.gene.annotation.gff', sep='\t', names=GFF_HEADERS, index_col='feature')

# CRISPR
c_gdsc_crispy = pd.read_csv('data/crispr_gdsc_logfc_corrected.csv', index_col=0)

# WGS
wgs_cnv = pd.read_csv('data/crispy_copy_number_gene_wgs.csv', index_col=0)
wgs_chrm = pd.read_csv('data/crispy_copy_number_chr_wgs.csv', index_col=0)
wgs_ratios = pd.read_csv('data/crispy_copy_number_gene_ratio_wgs.csv', index_col=0)
wgs_ploidy = pd.read_csv('data/crispy_copy_number_ploidy_wgs.csv', index_col=0, names=['sample', 'ploidy'])['ploidy']

# SNP6
snp_cnv = pd.read_csv('data/crispy_copy_number_gene_snp.csv', index_col=0)
snp_chrm = pd.read_csv('data/crispy_copy_number_chr_snp.csv', index_col=0)
snp_ratios = pd.read_csv('data/crispy_copy_number_gene_ratio_snp.csv', index_col=0)
snp_ploidy = pd.read_csv('data/crispy_copy_number_ploidy_snp.csv', index_col=0, names=['sample', 'ploidy'])['ploidy']

# - Overlap
genes, samples = set(wgs_cnv.index), set(wgs_cnv)
print(len(genes), len(samples))


# - Build data-frame
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

df = df.assign(in_crispr=df['sample'].isin(set(c_gdsc_crispy)).astype(int))
print(df.shape)


# - Targets for FISH validation
plot_df = []
for g in ['MYC']:
    df_ = df.query("gene == '%s'" % g).set_index('sample')
    df_ = pd.concat([df_, ss.loc[df_.index, 'Cancer Type']], axis=1).sort_values('ratio')
    plot_df.append(df_)
plot_df = pd.concat(plot_df).sort_values(['gene', 'cnv'])
plot_df.to_csv('data/crispy_fish_validations.csv')
print(plot_df)
