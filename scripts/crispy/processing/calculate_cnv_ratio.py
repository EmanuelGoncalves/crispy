#!/usr/bin/env python
# Copyright (C) 2018 Emanuel Goncalves

import os
import pandas as pd
from crispy.ratio import map_cn


# - GDSC: Import copy-number and estimate gene-level cn, chromosome copies and ploidy
# WGS
def map_cnv_bed_wgs(bed_folder):
    # Generate annotated BED files
    print('[INFO] WGS annotating gene: {}'.format(bed_folder))
    for f in os.listdir(bed_folder):
        if f.endswith('.ascat.bed'):
            df = map_cn(bed_folder + f)
            df.to_csv('data/gdsc/wgs/ascat_bed/%s.bed.gene.wgs.csv' % f.split('.')[0])

    # Assemble single matricies
    print('[INFO] WGS exporting matricies')
    write_cnv_matricies('wgs', bed_folder)


# SNP6
def map_cnv_bed_snp(bed_folder):
    # Generate annotated BED files
    print('[INFO] SNP annotating gene: {}'.format(bed_folder))
    for f in os.listdir(bed_folder):
        if f.endswith('.picnic.bed'):
            df = map_cn(bed_folder + f)
            df.to_csv('data/gdsc/copynumber/snp6_bed/%s.bed.gene.snp.csv' % f.split('.')[0])

    # Assemble single matricies
    print('[INFO] SNP exporting matricies')
    write_cnv_matricies('snp', bed_folder)


# Export matricies
def write_cnv_matricies(data_type, bed_folder):
    # Gene copy-number
    df = pd.concat([
            pd.read_csv('{}/{}'.format(bed_folder, i), index_col=0)['weight'].rename(i.split('_')[0])
        for i in os.listdir(bed_folder) if '.bed.gene.%s.csv' % data_type in i
    ], axis=1)
    df.to_csv('data/crispy_copy_number_gene_%s.csv' % data_type)

    # Gene copy-number ratio
    df = pd.concat([
            pd.read_csv('{}/{}'.format(bed_folder, i), index_col=0)['ratio'].rename(i.split('_')[0])
        for i in os.listdir(bed_folder) if '.bed.gene.%s.csv' % data_type in i
    ], axis=1)
    df.round(5).to_csv('data/crispy_copy_number_gene_ratio_%s.csv' % data_type)

    # Chromosome number of copies
    df = pd.concat([
            pd.read_csv('{}/{}'.format(bed_folder, i), index_col=0).groupby('chr').first()['chr_weight'].rename(i.split('_')[0])
        for i in os.listdir(bed_folder) if '.bed.gene.%s.csv' % data_type in i
    ], axis=1)
    df.round(5).to_csv('data/crispy_copy_number_chr_%s.csv' % data_type)

    # Ploidy
    df = pd.Series({
            i.split('_')[0]: pd.read_csv('{}/{}'.format(bed_folder, i), index_col=0).iloc[0]['ploidy']
        for i in os.listdir(bed_folder) if '.bed.gene.%s.csv' % data_type in i
    })
    df.round(5).to_csv('data/crispy_copy_number_ploidy_%s.csv' % data_type)


if __name__ == '__main__':
    # - Map GDSC CNV segments to gene level
    # WGS
    map_cnv_bed_wgs('data/gdsc/wgs/ascat_bed/')

    # SNP6
    map_cnv_bed_snp('data/gdsc/copynumber/snp6_bed/')
