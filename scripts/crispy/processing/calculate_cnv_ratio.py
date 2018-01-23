#!/usr/bin/env python
# Copyright (C) 2018 Emanuel Goncalves

import os
import pandas as pd
from crispy.ratio import map_cn


# - Import copy-number and estimate gene-level cn, chromosome copies and ploidy
# WGS
def map_cnv_bed_wgs(bed_folder='data/gdsc/wgs/ascat_bed/'):
    for f in os.listdir(bed_folder):
        if f.endswith('.ascat.bed'):
            df = map_cn(bed_folder + f)
            df.to_csv('data/crispy_copynumber/%s_crispy_copynumber_wgs.csv' % f.split('.')[0])


# SNP6
def map_cnv_bed_snp(bed_folder='data/gdsc/copynumber/snp6_bed/'):
    for f in os.listdir(bed_folder):
        if f.endswith('.picnic.bed'):
            df = map_cn(bed_folder + f)
            df.to_csv('data/crispy_copynumber/%s_crispy_copynumber_snp.csv' % f.split('.')[0])


# - Export matricies
def write_cnv_matricies(data_type):
    # Gene copy-number
    df = pd.concat([
            pd.read_csv('data/crispy_copynumber/%s' % i, index_col=0)['weight'].rename(i.split('_')[0])
        for i in os.listdir('data/crispy_copynumber/') if '_crispy_copynumber_%s.csv' % data_type in i
    ], axis=1)
    df.to_csv('data/crispy_gene_copy_number_%s.csv' % data_type)

    # Gene copy-number ratio
    df = pd.concat([
            pd.read_csv('data/crispy_copynumber/%s' % i, index_col=0)['ratio'].rename(i.split('_')[0])
        for i in os.listdir('data/crispy_copynumber/') if '_crispy_copynumber_%s.csv' % data_type in i
    ], axis=1)
    df.round(5).to_csv('data/crispy_gene_copy_number_ratio_%s.csv' % data_type)

    # Chromosome number of copies
    df = pd.concat([
            pd.read_csv('data/crispy_copynumber/%s' % i, index_col=0).groupby('chr').first()['chr_weight'].rename(i.split('_')[0])
        for i in os.listdir('data/crispy_copynumber/') if '_crispy_copynumber_%s.csv' % data_type in i
    ], axis=1)
    df.round(5).to_csv('data/crispy_chr_copy_number_%s.csv' % data_type)

    # Ploidy
    df = pd.Series({
            i.split('_')[0]: pd.read_csv('data/crispy_copynumber/%s' % i, index_col=0).iloc[0]['ploidy']
        for i in os.listdir('data/crispy_copynumber/') if '_crispy_copynumber_%s.csv' % data_type in i
    })
    df.round(5).to_csv('data/crispy_ploidy_%s.csv' % data_type)


if __name__ == '__main__':
    # Intersect WGS copy-number segments with gene annotation
    map_cnv_bed_wgs()

    # Intersect SNP6 copy-number segments with gene annotation
    map_cnv_bed_snp()

    # Assemble and write copy-number matricies
    write_cnv_matricies('wgs')
    write_cnv_matricies('snp')
