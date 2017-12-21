#!/usr/bin/env python
# Copyright (C) 2017 Emanuel Goncalves

import numpy as np
import pandas as pd


# - Setups
hgnc = 'data/resources/hgnc/non_alt_loci_set.txt'
segm = 'data/gdsc/copynumber/Summary_segmentation_data_994_lines_picnic.txt'
gcnv = 'data/gdsc/copynumber/Gene_level_CN.txt'


# - Imports
# HGNC gene information
ginfo = pd.read_csv(hgnc, sep='\t', index_col=1).dropna(subset=['location'])
ginfo = ginfo.assign(chr=ginfo['location'].apply(lambda x: x.replace('q', 'p').split('p')[0]))
ginfo = ginfo[~ginfo['chr'].isin(['X', 'Y'])]
ginfo.to_csv('data/crispy_hgnc_info.csv')

# Copy-number segments
cnv_seg = pd.read_csv(segm, sep='\t')
cnv_seg = cnv_seg[~cnv_seg['chr'].isin([23, 24])]
cnv_seg = cnv_seg.assign(chr=cnv_seg['chr'].astype(str))
cnv_seg = cnv_seg.assign(length=(cnv_seg['endpos'] - cnv_seg['startpos']))
cnv_seg = cnv_seg.assign(totalCN_by_length=cnv_seg['length'] * (cnv_seg['totalCN'] + 1))

# Gene copy-number
cnv_gene = pd.read_csv(gcnv, sep='\t', index_col=0)
cnv_gene = cnv_gene.drop(['chr', 'start', 'stop'], axis=1, errors='ignore').applymap(lambda v: int(v.split(',')[0]))
cnv_gene.to_csv('data/crispy_gene_copy_number.csv')


# - Estimate cell line ploidy
ploidy = cnv_seg.groupby('cellLine')['totalCN_by_length'].sum().divide(cnv_seg.groupby('cellLine')['length'].sum()) - 1
ploidy.reset_index().rename(columns={'cellLine': 'sample', 0: 'ploidy'}).to_csv('data/crispy_sample_ploidy.csv', index=False)


# - Estimate chromosome copies
chr_copies = cnv_seg.groupby(['cellLine', 'chr'])['totalCN_by_length'].sum().divide(cnv_seg.groupby(['cellLine', 'chr'])['length'].sum()) - 1
chr_copies_m = pd.pivot_table(chr_copies.reset_index().rename(columns={'cellLine': 'sample', 0: 'ploidy'}), index='chr', columns='sample', values='ploidy')
chr_copies.reset_index().rename(columns={'cellLine': 'sample', 0: 'ploidy'}).to_csv('data/crispy_chrm_copy_number.csv', index=False)


# - Estimate gene copy-number ratio
cnv_ratio = cnv_gene[cnv_gene.index.isin(ginfo.index)].replace(-1, np.nan)
cnv_ratio = cnv_ratio.divide(chr_copies_m.loc[ginfo.loc[cnv_ratio.index, 'chr'], cnv_ratio.columns].set_index(cnv_ratio.index))
cnv_ratio.to_csv('data/crispy_copy_number_ratio.csv')

