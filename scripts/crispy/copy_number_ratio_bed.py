#!/usr/bin/env python
# Copyright (C) 2018 Emanuel Goncalves

import os
import pandas as pd
from crispy.ratio import map_cn

# - Import copy-number and estimate gene-level cn, chromosome copies and ploidy
# WGS
wgs_folder = 'data/gdsc/wgs/ascat_bed/'
for f in os.listdir(wgs_folder):
    if f.endswith('.ascat.bed'):
        df = map_cn(wgs_folder + f)
        df.to_csv('data/crispy_copynumber/%s_crispy_copynumber_wgs.csv' % f.split('.')[0])

# SNP
snp_folder = 'data/gdsc/copynumber/snp6_bed/'
for f in os.listdir(snp_folder):
    if f.endswith('.picnic.bed'):
        df = map_cn(snp_folder + f)
        df.to_csv('data/crispy_copynumber/%s_crispy_copynumber_snp.csv' % f.split('.')[0])


# - Export matricies
for t in ['wgs', 'snp']:
    # Gene copy-number
    df = pd.concat([
            pd.read_csv('data/crispy_copynumber/%s' % i, index_col=0)['weight'].rename(i.split('_')[0])
        for i in os.listdir('data/crispy_copynumber/') if '_crispy_copynumber_%s.csv' % t in i
    ], axis=1)
    df.to_csv('data/crispy_gene_copy_number_%s.csv' % t)

    # Gene copy-number ratio
    df = pd.concat([
            pd.read_csv('data/crispy_copynumber/%s' % i, index_col=0)['ratio'].rename(i.split('_')[0])
        for i in os.listdir('data/crispy_copynumber/') if '_crispy_copynumber_%s.csv' % t in i
    ], axis=1)
    df.round(5).to_csv('data/crispy_gene_copy_number_ratio_%s.csv' % t)

    # Chromosome number of copies
    df = pd.concat([
            pd.read_csv('data/crispy_copynumber/%s' % i, index_col=0).groupby('chr').first()['chr_weight'].rename(i.split('_')[0])
        for i in os.listdir('data/crispy_copynumber/') if '_crispy_copynumber_%s.csv' % t in i
    ], axis=1)
    df.round(5).to_csv('data/crispy_chr_copy_number_%s.csv' % t)

    # Ploidy
    df = pd.Series({
            i.split('_')[0]: pd.read_csv('data/crispy_copynumber/%s' % i, index_col=0).iloc[0]['ploidy']
        for i in os.listdir('data/crispy_copynumber/') if '_crispy_copynumber_%s.csv' % t in i
    })
    df.round(5).to_csv('data/crispy_ploidy_%s.csv' % t)
