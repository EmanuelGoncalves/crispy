#!/usr/bin/env python
# Copyright (C) 2018 Emanuel Goncalves

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from natsort import natsorted
from pybedtools import BedTool
from plotting import plot_rearrangements_seg, plot_chromosome_seg
from processing.bed_seg_crispr import calculate_ploidy
from scripts import BRCA_SAMPLES
from utils import bin_cnv

BRASS_HEADERS = [
    '# chr1', 'start1', 'end1', 'chr2', 'start2', 'end2', 'id/name', 'brass_score', 'strand1', 'strand2',
    'sample', 'svclass', 'bkdist', 'assembly_score', 'readpair names', 'readpair count', 'bal_trans', 'inv',
    'occL', 'occH', 'copynumber_flag', 'range_blat', 'Brass Notation', 'non-template', 'micro-homology',
    'assembled readnames', 'assembled read count', 'gene1', 'gene_id1', 'transcript_id1', 'strand1', 'end_phase1',
    'region1', 'region_number1', 'total_region_count1', 'first/last1', 'gene2', 'gene_id2', 'transcript_id2',
    'strand2', 'phase2', 'region2', 'region_number2', 'total_region_count2', 'first/last2', 'fusion_flag'
]

ASCAT_HEADERS = ['#chr', 'start', 'end', 'total_cn']


def brass_bed(brass_file, splitreads=True, copynumber_flag=True):
    # Import BRASS bedpe
    bedpe_df = pd.read_csv(brass_file, sep='\t', names=BRASS_HEADERS, comment='#').rename(columns={'# chr1': 'chr1'})
    bedpe_df['chr1'] = bedpe_df['chr1'].apply(lambda x: 'chr{}'.format(x))

    # Correct sample name
    bedpe_df['sample'] = bedpe_df['sample'].apply(lambda v: v.split(',')[0])

    # BRASS2 annotated SV
    if splitreads:
        bedpe_df = bedpe_df.query("assembly_score != '_'")

    # Copy-number flag positive
    if copynumber_flag:
        bedpe_df = bedpe_df.query('copynumber_flag == 1')

    # Assemble bed file
    bed_df = []

    # Expand translocations
    df = bedpe_df.query("svclass == 'translocation'")
    for i in [1, 2]:
        bed_df.append(pd.concat([
            df[f'chr{i}'].rename('#chr'),
            df[[f'start{i}', f'end{i}']].mean(1).astype(int).rename('start'),
            df[[f'start{i}', f'end{i}']].mean(1).astype(int).rename('end'),
            df['svclass']
        ], axis=1))

    # Other SVs
    df = bedpe_df.query("svclass != 'translocation'")
    bed_df.append(pd.concat([
        df['chr1'].rename('#chr'),
        df[['start1', 'end1']].mean(1).astype(int).rename('start'),
        df[['start2', 'end2']].mean(1).astype(int).rename('end'),
        df['svclass'].rename('svclass')
    ], axis=1))

    bed_df = pd.concat(bed_df)

    return bed_df


def map_ascat_brass(brass_bed_file, ascat_bed_file):
    svs, segs = BedTool(brass_bed_file).sort(), BedTool(ascat_bed_file).sort()

    names = ['#chr', 'start', 'end', 'cn', 'svs']
    bed = segs.map(svs, c=4, o='collapse').to_dataframe(names=names)

    return bed


if __name__ == '__main__':
    #
    bed_sv = {}
    for sample in BRCA_SAMPLES:
        # sample = 'HCC38'

        brass_file = f'data/wgs/{sample}.brass.annot.bedpe'
        brass_bed_file = f'data/wgs/{sample}.brass.bed'
        brass_bed(brass_file).to_csv(brass_bed_file, index=False, sep='\t')

        ascat_bed_file = f'data/wgs/{sample}.wgs.ascat.bed'
        bed_df = map_ascat_brass(brass_bed_file, ascat_bed_file)

        bed_df = bed_df.assign(sample=sample)

        bed_sv[sample] = bed_df

    #
    bed_crispr = {s: pd.read_csv(f'data/crispr_bed/{s}.crispr.snp.wgs.bed', sep='\t') for s in BRCA_SAMPLES}

    # - Aggregate per segment
    agg_fun = dict(fc=np.mean, cn=np.mean, chrm=np.mean, ploidy=np.mean, ratio=np.mean, len=np.mean, sgrna_id='count')

    df = pd.concat([
        bed_crispr[s].assign(sample=s).groupby(['sample', 'chr', 'start', 'end']).agg(agg_fun) for s in BRCA_SAMPLES
    ])

    df_svs = pd.concat([
        bed_sv[s].rename(columns={'#chr': 'chr'}).groupby(['sample', 'chr', 'start', 'end']).first()['svs'] for s in BRCA_SAMPLES
    ])
    df = pd.concat([df, df_svs], axis=1).dropna()

    df = df.assign(len_log2=np.log2(df['len']))

    df = df.assign(ratio_bin=df['ratio'].apply(lambda v: bin_cnv(v, thresold=4)))
    df = df.assign(cn_bin=df['cn'].apply(lambda v: bin_cnv(v, thresold=10)))
    df = df.assign(chrm_bin=df['chrm'].apply(lambda v: bin_cnv(v, thresold=6)))
    df = df.assign(ploidy_bin=df['ploidy'].apply(lambda v: bin_cnv(v, thresold=5)))

    #
    plot_df = df[df['svs'].apply(lambda v: len(v.split(',')) == 1)]
    plot_df = plot_df.reset_index()
    plot_df = plot_df[~plot_df['chr'].isin(['chrX', 'chrY'])]
    plot_df = plot_df.query('sgrna_id > 30')

    order = natsorted(set(plot_df['ratio_bin']))
    hue_order = ['.', 'inversion', 'deletion', 'tandem-duplication', 'translocation']

    sns.boxplot('ratio_bin', 'fc', 'svs', data=plot_df, notch=True, order=order, hue_order=hue_order, linewidth=.3, fliersize=.1)

    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), frameon=False)

    plt.gcf().set_size_inches(3, 3)
    plt.savefig('reports/wgs_svs_fc_boxplots.png', bbox_inches='tight', dpi=600)
    plt.close('all')

    #
    sample, chrm = 'HCC38', 'chr9'

    ax = plot_rearrangements_seg(sample, chrm)
    ax[2].plot((12884404/1e6, 13784707/1e6), (0, 0), lw=1., color='red')

    plt.suptitle(f'{sample}', fontsize=10)
    plt.gcf().set_size_inches(8, 3)
    plt.savefig('reports/chromosomeplot_svs.png', bbox_inches='tight', dpi=600)
    plt.close('all')
