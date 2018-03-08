#!/usr/bin/env python
# Copyright (C) 2018 Emanuel Goncalves

import os
import pandas as pd
import scripts as mp
from crispy.ratio import map_cn


# - GDSC: Import copy-number and estimate gene-level cn, chromosome copies and ploidy
def map_cnv_bed(bed_folder, dtype):
    # Generate annotated BED files
    print('[INFO] {} annotating gene: {}'.format(dtype, bed_folder))

    for f in os.listdir(bed_folder):
        if f.endswith('.ascat.bed'):
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


if __name__ == '__main__':
    # WGS
    map_cnv_bed(mp.WGS_ASCAT_BED, 'wgs')

    # SNP6
    map_cnv_bed(mp.SNP_PICNIC_BED, 'snp')
