#!/usr/bin/env python
# Copyright (C) 2018 Emanuel Goncalves

import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pybedtools import BedTool
from scripts.plotting.segment.geckov2.calculate_fold_change import GECKOV2_SGRNA_MAP
from scripts.plotting.segment.geckov2.offtarget import lib_off_targets, import_seg_crispr_beds


BED_NAMES = ['#chr', 'start', 'end', 'cn', 'chr', 'sgrna_start', 'sgrna_end', 'sgrna_id']


def create_lib_fasta():
    lib = pd.read_csv(GECKOV2_SGRNA_MAP, sep='\t').dropna(subset=['Chromosome', 'Pos', 'Strand'])

    lib['sgrna'] = lib['Guide'].apply(lambda v: v.split('_')[0])

    lib = lib.rename(columns={'Chromosome': '#chr', 'Pos': 'start'})

    lib['start'] = lib['start'].astype(int)
    lib['end'] = lib['start'] + lib['sgrna'].apply(len) + lib['PAM'].apply(len)

    lib['id'] = lib['sgrna'] + '_' + lib['#chr'] + '_' + lib['start'].astype(str)

    lib = lib.groupby(['#chr', 'start', 'end'])['id'].first().reset_index()

    lib.to_csv('data/geckov2/geckov2_library.bed', index=False, sep='\t')


def calculate_ploidy(snp_bed):
    df = snp_bed.to_dataframe().rename(columns={'name': 'total_cn', 'chrom': 'chr'})
    df = df.assign(length=df['end'] - df['start'])
    df = df.assign(cn_by_length=df['length'] * (df['total_cn'] + 1))

    chrm = df.groupby('chr')['cn_by_length'].sum().divide(df.groupby('chr')['length'].sum()) - 1

    ploidy = (df['cn_by_length'].sum() / df['length'].sum()) - 1

    return chrm, ploidy


def intersect_bed(sample):
    # Input/Output files
    file_snp_bed = f'data/geckov2/crispr_bed/{sample}.snp6.ABSOLUTE.bed'
    file_lib_bed = 'data/geckov2/geckov2_library.bed'
    file_bed = f'data/geckov2/crispr_bed/{sample}.library.snp.bed'

    # Import BED files
    snp_bed = BedTool(file_snp_bed).sort()
    lib_bed = BedTool(file_lib_bed).sort()

    # Intersect SNP6 copy-number segments with CRISPR-Cas9 sgRNAs
    bed = snp_bed.intersect(lib_bed, wa=True, wb=True).to_dataframe(names=BED_NAMES)

    # Calculate chromosome copies and cell ploidy
    chrm, ploidy = calculate_ploidy(snp_bed)

    bed = bed.assign(chrm=chrm[bed['chr']].values)
    bed = bed.assign(ploidy=ploidy)

    # Calculate copy-number ratio
    bed = bed.assign(ratio=bed.eval('cn / chrm'))

    # Calculate segment length
    bed = bed.assign(len=bed.eval('end - start'))

    # Round
    round_cols = ['chrm', 'ploidy', 'ratio']
    bed[round_cols] = bed[round_cols].round(5)

    #
    bed = bed.groupby(['chr', 'start', 'end']).agg(dict(
        cn='first', chrm='first', ploidy='first', ratio='first', len='first', sgrna_id=lambda x: ';'.join(set(x))
    )).reset_index()

    # Export
    bed.to_csv(file_bed, index=False, sep='\t')

    return bed


if __name__ == '__main__':
    # Import libraries bed
    beds = import_seg_crispr_beds()

    # Off-targets
    sgrnas = lib_off_targets()

    #
    sgrnas.loc[beds[s]['sgrna_id']]