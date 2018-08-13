#!/usr/bin/env python
# Copyright (C) 2018 Emanuel Goncalves

import pandas as pd
import crispy as cy
import scripts as mp
from pybedtools import BedTool


BED_NAMES = ['#chr', 'start', 'end', 'cn', 'chr', 'sgrna_start', 'sgrna_end', 'fc', 'sgrna_id']
BED_NAMES_ORDER = ['chr', 'start', 'end', 'sgrna_id', 'sgrna_start', 'sgrna_end', 'cn', 'fc', 'chrm', 'ploidy', 'ratio', 'len']


def import_lib():
    # Import CRISPR library
    lib = cy.get_crispr_lib()\
        .dropna(subset=['STARTpos', 'ENDpos'])\
        .rename(columns={'CHRM': '#chr', 'STARTpos': 'start', 'ENDpos': 'end'})

    lib['#chr'] = lib['#chr'].apply(lambda v: 'chr{}'.format(v))
    lib['start'] = lib['start'].astype(int)
    lib['end'] = lib['end'].astype(int)

    lib['id'] = lib.index

    return lib


def calculate_ploidy(snp_bed):
    df = snp_bed.to_dataframe().rename(columns={'name': 'total_cn', 'chrom': 'chr'})
    df = df.assign(length=df['end'] - df['start'])
    df = df.assign(cn_by_length=df['length'] * (df['total_cn'] + 1))

    chrm = df.groupby('chr')['cn_by_length'].sum().divide(df.groupby('chr')['length'].sum()) - 1

    ploidy = (df['cn_by_length'].sum() / df['length'].sum()) - 1

    return chrm, ploidy


def intersect_bed(sample, fc):
    # Import CRISPR lib
    lib = import_lib()

    # Input/Output files
    file_snp_bed = '{}/{}.wgs.ascat.bed'.format(mp.WGS_ASCAT_BED, sample)
    file_crispr_bed = 'data/crispr_bed/{}.crispr.wgs.bed'.format(sample)
    file_bed = 'data/crispr_bed/{}.crispr.snp.wgs.bed'.format(sample)

    # Build CRISPR BED
    crispr_bed = pd.concat([fc[sample], lib], axis=1, sort=True).dropna()
    crispr_bed = crispr_bed[['#chr', 'start', 'end', sample, 'id']]
    crispr_bed.to_csv(file_crispr_bed, index=False, sep='\t')

    # Import BED files
    snp_bed = BedTool(file_snp_bed).sort()
    crispr_bed = BedTool(file_crispr_bed).sort()

    # Intersect SNP6 copy-number segments with CRISPR-Cas9 sgRNAs
    bed = snp_bed.intersect(crispr_bed, wa=True, wb=True).to_dataframe(names=BED_NAMES)

    # Calculate chromosome copies and cell ploidy
    chrm, ploidy = calculate_ploidy(snp_bed)

    bed = bed.assign(chrm=chrm[bed['chr']].values)
    bed = bed.assign(ploidy=ploidy)

    # Calculate copy-number ratio
    bed = bed.assign(ratio=bed.eval('cn / chrm'))

    # Calculate segment length
    bed = bed.assign(len=bed.eval('end - start'))

    # Round
    round_cols = ['fc', 'chrm', 'ploidy', 'ratio']
    bed[round_cols] = bed[round_cols].round(5)

    # Export
    bed[BED_NAMES_ORDER].to_csv(file_bed, index=False, sep='\t')

    return bed


if __name__ == '__main__':
    # - Import CRISPR-Cas9 sgRNAs
    crispr = mp.get_crispr_sgrna()

    # - Import sample list
    samples = mp.get_sample_list()

    # - Intersect SNP6 segments with CRISPR-Cas9
    for s in mp.BRCA_SAMPLES:
        print('[INFO] BedTools: {}'.format(s))
        intersect_bed(s, crispr)
