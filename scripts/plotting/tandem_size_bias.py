#!/usr/bin/env python
# Copyright (C) 2018 Emanuel Goncalves

import os
import pickle
import numpy as np
import pandas as pd
import itertools as it
import seaborn as sns
import matplotlib.pyplot as plt
from crispy import PAL_DBGD
from natsort import natsorted
from pybedtools import BedTool
from crispy.utils import bin_cnv
from crispy.ratio import BRASS_HEADERS, GFF_FILE, GFF_HEADERS


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


def import_brass_bedpe(bedpe_file, bkdist, splitreads):
    # bedpe_file = 'data/gdsc/wgs/brass_bedpe/HCC1954.brass.annot.bedpe'
    # Import BRASS bedpe
    bedpe_df = pd.read_table(bedpe_file, sep='\t', names=BRASS_HEADERS, comment='#')

    # Correct sample name
    bedpe_df['sample'] = bedpe_df['sample'].apply(lambda v: v.split(',')[0])

    # SV larger than threshold
    bedpe_df = bedpe_df[bedpe_df['bkdist'] > bkdist]

    # BRASS2 annotated SV
    if splitreads:
        bedpe_df = bedpe_df.query("assembly_score != '_'")

    # # Fileter for tandem-duplications
    # bedpe_df = bedpe_df[bedpe_df['svclass'] == 'tandem-duplication']

    # Classify tandem-duplication based on size
    # bedpe_df = bedpe_df.assign(td_size=bedpe_df['bkdist'].apply(bin_bkdist).values)

    # Assemble bed file
    bed_df = pd.concat([
        bedpe_df['chr1'].rename('#chr').apply(lambda x: 'chr{}'.format(x)),
        bedpe_df[['start1', 'end1']].mean(1).astype(int).rename('start'),
        bedpe_df[['start2', 'end2']].mean(1).astype(int).rename('end'),
        bedpe_df['svclass'].rename('svclass')
    ], axis=1)

    return bed_df


def annotate_bed(bed_file, methods='collapse,count'):
    # Import Genes annotation bed file and specified bed file
    gff, bed = BedTool(GFF_FILE).sort(), BedTool(bed_file).sort()

    # Map GFF
    genes_sv = gff.map(bed, c=4, o='collapse,count').to_dataframe(names=GFF_HEADERS + methods.split(','))

    return genes_sv


def annotate_brass_bedpe(bedpe_dir, bkdist, splitreads, samples=None):
    bed_dfs = []

    for bedpe_file in map(lambda f: '{}/{}'.format(bedpe_dir, f), filter(lambda f: f.endswith('.brass.annot.bedpe'), os.listdir(bedpe_dir))):
        # Sample name
        sample = os.path.splitext(os.path.basename(bedpe_file))[0].split('.')[0]

        if (samples is not None) and (sample not in samples):
            continue

        print('[INFO] {}'.format(bedpe_file))

        # Import and filter bedpe
        bed_df = import_brass_bedpe(bedpe_file, bkdist, splitreads)

        if bed_df.shape[0] > 0:
            bed_file = '{}/{}.bed'.format(bedpe_dir, os.path.splitext(os.path.basename(bedpe_file))[0])
            bed_df.to_csv(bed_file, sep='\t', index=False)

            # Annotate SV with gene information
            bed_annot_df = annotate_bed(bed_file).sort_values('count', ascending=False)

            # Append df
            bed_annot_df = bed_annot_df.assign(sample=sample)
            bed_dfs.append(bed_annot_df)

    # Assemble annotated bed dataframes
    bed_dfs = pd.concat(bed_dfs)

    # Filter genes without SVs
    bed_dfs = bed_dfs.query("collapse != '.'")

    return bed_dfs


if __name__ == '__main__':
    bedpe_dir = 'data/gdsc/wgs/brass_bedpe'

    samples = ['HCC1954', 'HCC1143', 'HCC38', 'HCC1187', 'HCC1937', 'HCC1395']
    # samples = ['HCC1954', 'HCC1143', 'HCC38', 'HCC1187', 'HCC1937', 'HCC1395', 'NCI-H2087', 'LS-1034']

    # Annotate BRASS bedpes
    bed_dfs = annotate_brass_bedpe(bedpe_dir, bkdist=-2, splitreads=True, samples=samples)

    # Non-expressed genes
    nexp = pickle.load(open('data/gdsc/nexp_pickle.pickle', 'rb'))

    # - CRISPR
    c_gdsc_fc = pd.read_csv('data/crispr_gdsc_logfc.csv', index_col=0).dropna()
    c_gdsc_fc = pd.DataFrame({c: c_gdsc_fc.reindex(nexp[c])[c] for c in c_gdsc_fc if c in nexp})
    # c_gdsc_fc = dc.scale_crispr(c_gdsc_fc)

    # - Copy-number
    cnv = pd.read_csv('data/crispy_copy_number_gene_wgs.csv', index_col=0)
    cnv_ratios = pd.read_csv('data/crispy_copy_number_gene_ratio_wgs.csv', index_col=0)

    # -
    ploidy = pd.read_csv('data/crispy_copy_number_ploidy_snp.csv', index_col=0, names=['sample', 'ploidy'])['ploidy']

    # - Overlap
    genes = set(c_gdsc_fc.index).intersection(cnv.index)

    # -
    df = bed_dfs[bed_dfs['feature'].isin(c_gdsc_fc.index)]
    # df = df.assign(collapse=[natsorted(i.split(','))[0] for i in df['collapse']])
    df = df[df['count'] == 1]
    df = df.assign(ratio=[cnv_ratios.loc[g, s] for s, g in df[['sample', 'feature']].values])
    df = df.assign(ratio_bin=df['ratio'].apply(lambda v: bin_cnv(v, thresold=4)))
    df = df.assign(crispr=[c_gdsc_fc.loc[g, s] for s, g in df[['sample', 'feature']].values])
    df = df.dropna()

    # -
    order = natsorted(set(df['ratio_bin']))
    hue_order = natsorted(set(df['collapse']))
    # hue_order = ['1-10 kb', '1-100 kb', '0.1-1 Mb', '1-10 Mb', '>10 Mb']

    sns.boxplot(
        'ratio_bin', 'crispr', 'collapse', data=df, order=order, hue_order=hue_order, notch=True, palette=sns.light_palette(PAL_DBGD[0]), fliersize=1
    )
    plt.savefig('reports/crispy/td_size_bias.png', bbox_inches='tight', dpi=600)
    plt.close('all')

