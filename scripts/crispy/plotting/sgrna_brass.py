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


if __name__ == '__main__':
    bedpe_dir = 'data/gdsc/wgs/brass_bedpe'

    # - BRASS
    bed_dfs = []
    for bedpe_file in map(lambda f: '{}/{}'.format(bedpe_dir, f), filter(lambda f: f.endswith('.brass.annot.bedpe'), os.listdir(bedpe_dir))):
        print('[INFO] {}'.format(bedpe_file))

        # Import and filter bedpe
        bed_df = import_brass_bedpe(bedpe_file, bkdist=0, splitreads=True)

        # Append df
        bed_df = bed_df.assign(sample=os.path.splitext(os.path.basename(bedpe_file))[0].split('.')[0])
        bed_dfs.append(bed_df)

    # Assemble annotated bed dataframes
    bed_dfs = pd.concat(bed_dfs)

    # - CRISPR/Cas9 sgRNA
    crispr_sgrna = pd.read_csv('data/crispr_gdsc_sgrna_logfc.csv', index_col=0)

    # - Overlap
    samples = list(set(crispr_sgrna).intersection(bed_dfs['sample']))
    print('Samples: {}'.format(len(samples)))

    # -
    bedpe = import_brass_bedpe('data/gdsc/wgs/brass_bedpe/HCC1187.brass.annot.bedpe', bkdist=0, splitreads=True)
    ngsc = pd.read_csv(
        'data/gdsc/wgs/brass_intermediates/HCC1187.ngscn.abs_cn.bg', sep='\t', header=None,
        names=['chr', 'start', 'end', 'cn', 'score'], dtype={'chr': str, 'start': int, 'end': int, 'cn': float, 'score': float}
    )

    chrm = '13'

    ngsc_ = ngsc[ngsc['chr'] == chrm]
    bedpe_ = bedpe[bedpe['chr1'] == chrm]

