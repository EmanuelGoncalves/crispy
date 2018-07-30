#!/usr/bin/env python
# Copyright (C) 2018 Emanuel Goncalves

import os
import pandas as pd
import scripts as mp
from crispy.ratio import map_cn, SEX_CHR


# - GDSC: Import copy-number and estimate gene-level cn, chromosome copies and ploidy
def map_cnv_bed(bed_folder, dtype):
    # Generate annotated BED files
    print('[INFO] {} annotating gene: {}'.format(dtype, bed_folder))

    for f in os.listdir(bed_folder):
        if f.endswith('.ascat.bed' if dtype == 'wgs' else '.picnic.bed'):
            df = map_cn(bed_folder + f)
            df.to_csv('{}/{}.bed.gene.{}.csv'.format(bed_folder, f.split('.')[0], dtype))

    # Assemble single matrices
    write_cnv_matrices(dtype, bed_folder)


# Export matricies
def write_cnv_matrices(data_type, bed_folder):
    # Gene copy-number
    pd.concat([
        pd.read_csv('{}/{}'.format(bed_folder, i), index_col=0)['weight'].rename(i.split('.')[0]) for i in os.listdir(bed_folder)
        if '.bed.gene.{}.csv'.format(data_type) in i
    ], axis=1).to_csv(mp.CN_GENE.format(data_type))

    # Gene copy-number ratio
    pd.concat([
        pd.read_csv('{}/{}'.format(bed_folder, i), index_col=0)['ratio'].rename(i.split('.')[0]) for i in os.listdir(bed_folder)
        if '.bed.gene.{}.csv'.format(data_type) in i
    ], axis=1).round(5).to_csv(mp.CN_GENE_RATIO.format(data_type))

    # Chromosome number of copies
    pd.concat([
        pd.read_csv('{}/{}'.format(bed_folder, i), index_col=0).groupby('chr').first()['chr_weight'].rename(i.split('.')[0]) for i in os.listdir(bed_folder)
        if '.bed.gene.{}.csv'.format(data_type) in i
    ], axis=1).round(5).to_csv(mp.CN_CHR.format(data_type))

    # Ploidy
    pd.Series({
        i.split('.')[0]: pd.read_csv('{}/{}'.format(bed_folder, i), index_col=0).iloc[0]['ploidy'] for i in os.listdir(bed_folder)
        if '.bed.gene.{}.csv'.format(data_type) in i
    }).round(5).to_csv(mp.CN_PLOIDY.format(data_type))


def segment_ratios():
    # Chromosome copies
    chrm = mp.get_chr_copies()

    for f in os.listdir(mp.SNP_PICNIC_BED):
        if f.endswith('.picnic.bed'):
            s = f.split('.')[0]
            print(s)

            df = pd.read_csv(mp.SNP_PICNIC_BED + f, sep='\t')

            df = df[~df['#chr'].isin(SEX_CHR)]

            df = df.assign(chrm_cnv=chrm.loc[df['#chr'], s].values)

            df = df.assign(seg_ratio=df.eval('total_cn / chrm_cnv'))

            df.to_csv('{}/{}.snp6.seg.ratio.csv'.format(mp.SNP_PICNIC_BED, s), index=False)


if __name__ == '__main__':
    # WGS
    map_cnv_bed(mp.WGS_ASCAT_BED, 'wgs')

    # SNP6
    map_cnv_bed(mp.SNP_PICNIC_BED, 'snp')
