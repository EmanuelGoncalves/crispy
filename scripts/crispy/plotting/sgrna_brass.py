#!/usr/bin/env python
# Copyright (C) 2018 Emanuel Goncalves

import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pybedtools import BedTool
from crispy.ratio import BRASS_HEADERS, GFF_FILE, GFF_HEADERS


def import_brass_bedpe(bedpe_file, bkdist, splitreads):
    # Import BRASS bedpe
    bedpe_df = pd.read_csv(bedpe_file, sep='\t', names=BRASS_HEADERS, comment='#')

    # Correct sample name
    bedpe_df['sample'] = bedpe_df['sample'].apply(lambda v: v.split(',')[0])

    # SV larger than threshold
    bedpe_df = bedpe_df[bedpe_df['bkdist'] > bkdist]

    # BRASS2 annotated SV
    if splitreads:
        bedpe_df = bedpe_df.query("assembly_score != '_'")

    return bedpe_df


def bin_bkdist(distance):
    if 1 < distance < 10e3:
        bin_distance = '1-10 kb'

    elif 10e3 < distance < 100e3:
        bin_distance = '1-100 kb'

    elif 100e3 < distance < 1e6:
        bin_distance = '0.1-1 Mb'

    elif 1e6 < distance < 10e6:
        bin_distance = '1-10 Mb'

    else:
        bin_distance = '>10 Mb'

    return bin_distance


if __name__ == '__main__':
    bedpe_dir = 'data/gdsc/wgs/brass_bedpe'

    samples = ['HCC1954', 'HCC1143', 'HCC38', 'HCC1187', 'HCC1937', 'HCC1395']

    # - BRASS
    bed_dfs = []
    for s in samples:
        bedpe_file = '{}/{}.brass.annot.bedpe'.format(bedpe_dir, s)
        print('[INFO] {}'.format(bedpe_file))

        # Import and filter bedpe
        bed_df = import_brass_bedpe(bedpe_file, bkdist=-2, splitreads=False)

        # Append df
        bed_df = bed_df.assign(sample=os.path.splitext(os.path.basename(bedpe_file))[0].split('.')[0])
        bed_dfs.append(bed_df)

    # Assemble annotated bed dataframes
    bed_dfs = pd.concat(bed_dfs).reset_index(drop=True)

    # - CRISPR/Cas9 sgRNA
    crispr_sgrna = pd.read_csv('data/crispr_gdsc_sgrna_logfc.csv', index_col=0)[samples].dropna()

    # -
    plot_df = bed_dfs[bed_dfs['bkdist'] > -1]
    plot_df = plot_df.assign(binbkdist=plot_df['bkdist'].map(bin_bkdist).values)
    plot_df = plot_df.groupby(['sample', 'svclass', 'binbkdist'])['bkdist'].count().reset_index()

    sns.barplot('svclass', 'bkdist', 'binbkdist', data=plot_df, hue_order=['1-10 kb', '1-100 kb', '0.1-1 Mb', '1-10 Mb', '>10 Mb'], palette='Blues', ci=None)
    plt.show()
