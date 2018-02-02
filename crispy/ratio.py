#!/usr/bin/env python
# Copyright (C) 2018 Emanuel Goncalves

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from crispy import bipal_dbgd
from pybedtools import BedTool

SEX_CHR = ['chrX', 'chrY']

GFF_FILE = 'data/gene_sets/gencode.gene.annotation.gff'

GFF_HEADERS = ['chr', 'db', 'type', 'start', 'end', 'score', 'strand', 'frame', 'feature']


def read_gff_dataframe(gff_file=None):
    gff = BedTool(GFF_FILE if gff_file is None else gff_file).sort()
    gff = gff.to_dataframe(names=GFF_HEADERS).set_index('feature')
    return gff


# TODO: Add Documentation
def map_cn(bed_file, method='min,max,mean,median,collapse,count', cn_field_pos=4, null=np.nan):
    # - Imports
    # Import Genes annotation bed file and specified bed file
    gff, bed = BedTool(GFF_FILE).sort(), BedTool(bed_file).sort()

    # BED headers
    bed_headers = list(map(lambda x: '%s_bed' % x, open(bed_file, 'r').readline().rstrip().split('\t')))

    # - Intersect regions of copy-number measurements with genes
    # Intersect Copy-Number BED to GFF
    gff_int_df = gff.intersect(bed, wb=True).to_dataframe(names=GFF_HEADERS + bed_headers)

    # Intersect length
    gff_int_df = gff_int_df.assign(int_length=(gff_int_df['end'] - gff_int_df['start']))

    # Total copy-number per segment
    gff_int_df = gff_int_df.assign(cn_by_length=gff_int_df['int_length'] * (gff_int_df['total_cn_bed'] + 1))

    # Weighted copy-number average per gene
    gff_int_df = gff_int_df.groupby('feature')['cn_by_length'].sum().divide(gff_int_df.groupby('feature')['int_length'].sum()) - 1

    # - Map gene annotation with copy-number bed file
    # Mapped bed Copy-Number to GFF
    gff_map_df = gff.map(bed, c=cn_field_pos, o=method, null=null)

    # Convert to data-frame
    gff_map_df = gff_map_df.to_dataframe(names=GFF_HEADERS + method.split(',')).set_index('feature')

    # Weighted gene copy-number
    gff_map_df = pd.concat([gff_map_df, gff_int_df.rename('weight')], axis=1)

    # Number of chromosome copies
    gff_map_df = gff_map_df.assign(chr_weight=chromosome_cn(bed_file)[gff_map_df['chr']].values)

    # Sample ploidy
    gff_map_df = gff_map_df.assign(ploidy=ploidy(bed_file))

    # Gene ratio
    gff_map_df = gff_map_df.assign(ratio=gff_map_df['weight'].divide(gff_map_df['chr_weight']).values)

    return gff_map_df


def _segments_cn(bed_file):
    bed = pd.read_csv(bed_file, sep='\t')

    bed = bed[~bed['#chr'].isin(SEX_CHR)]

    bed = bed.assign(length=(bed['end'] - bed['start']))

    bed = bed.assign(cn_by_length=bed['length'] * (bed['total_cn'] + 1))

    return bed


def chromosome_cn(bed_file):
    bed = _segments_cn(bed_file)

    bed = bed.groupby('#chr')['cn_by_length'].sum().divide(bed.groupby('#chr')['length'].sum()) - 1

    return bed


def ploidy(bed_file):
    bed = _segments_cn(bed_file)

    return (bed['cn_by_length'].sum() / bed['length'].sum()) - 1


def plot_gene(gene, bed_file, ax=None):
    # Import Genes annotation bed file and specified bed file
    gff, bed = BedTool(GFF_FILE).sort(), BedTool(bed_file).sort()

    # BED headers
    bed_headers = list(map(lambda x: '%s_bed' % x, open(bed_file, 'r').readline().rstrip().split('\t')))

    # Intersect Copy-Number BED to GFF
    gff_int_df = gff.intersect(bed, wb=True).to_dataframe(names=GFF_HEADERS + bed_headers).set_index('feature')

    # if ax is None:
    ax = plt.gca()

    # Plot original values
    for seg_start, seg_end, seg_cn in gff_int_df.loc[gene, ['start', 'end', 'total_cn_bed']].values:
        ax.plot([seg_start, seg_end], [seg_cn, seg_cn], lw=1., c=bipal_dbgd[0], alpha=1.)

    return ax
