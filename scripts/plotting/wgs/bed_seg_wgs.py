#!/usr/bin/env python
# Copyright (C) 2018 Emanuel Goncalves

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


def brass_bed(brass_file, splitreads=True, copynumber_flag=True, svclass='tandem-duplication', bkdist=0, assembly_score=90):
    # Import BRASS bedpe
    bedpe_df = pd.read_csv(brass_file, sep='\t', names=BRASS_HEADERS, comment='#')
    bedpe_df['# chr1'] = bedpe_df['# chr1'].apply(lambda x: 'chr{}'.format(x))

    # Correct sample name
    bedpe_df['sample'] = bedpe_df['sample'].apply(lambda v: v.split(',')[0])

    # SV larger than threshold
    bedpe_df = bedpe_df.query('bkdist > @bkdist')

    # BRASS2 annotated SV
    if splitreads:
        bedpe_df = bedpe_df.query("assembly_score != '_'")
        bedpe_df = bedpe_df[bedpe_df['assembly_score'].astype(int) >= assembly_score]

    # Copy-number flag positive
    if copynumber_flag:
        bedpe_df = bedpe_df.query('copynumber_flag == 1')

    # SVclass filter
    bedpe_df = bedpe_df.query('svclass == @svclass')

    # Assemble bed file
    bed_df = pd.concat([
        bedpe_df['# chr1'].rename('#chr'),
        bedpe_df[['start1', 'end1']].mean(1).astype(int).rename('start'),
        bedpe_df[['start2', 'end2']].mean(1).astype(int).rename('end'),
        bedpe_df['svclass'].rename('svclass')
    ], axis=1)

    return bed_df


def copy_number_intersection_with_svs(brass_bed_file, ascat_bed_file, svclass):
    svs, segs = BedTool(brass_bed_file).sort(), BedTool(ascat_bed_file).sort()

    bed_intersect = segs.intersect(svs).to_dataframe(names=['#chr', 'start', 'end', 'cn']).assign(svclass=svclass)
    bed_subtract = segs.subtract(svs).to_dataframe(names=['#chr', 'start', 'end', 'cn']).assign(svclass='none')

    bed_segs_svs = pd.concat([bed_intersect, bed_subtract])

    return bed_segs_svs


def segments_intersection_with_crispr(crispr_bed_file, ascat_brass_bed_file):
    # Import BED files
    seg_bed, crispr_bed = BedTool(ascat_brass_bed_file).sort(), BedTool(crispr_bed_file).sort()

    # Intersect SNP6 copy-number segments with CRISPR-Cas9 sgRNAs
    names = ['#chr', 'start', 'end', 'cn', 'svclass', 'chr', 'sgrna_start', 'sgrna_end', 'fc', 'sgrna_id']
    bed = seg_bed.intersect(crispr_bed, wa=True, wb=True).to_dataframe(names=names)

    # Calculate chromosome copies and cell ploidy
    chrm, ploidy = calculate_ploidy(seg_bed)

    bed = bed.assign(chrm=chrm[bed['chr']].values)
    bed = bed.assign(ploidy=ploidy)

    # Calculate copy-number ratio
    bed = bed.assign(ratio=bed.eval('cn / chrm'))

    # Calculate segment length
    bed = bed.assign(len=bed.eval('end - start'))

    return bed


def brass_translocations_bed(brass_file, splitreads=True, copynumber_flag=True):
    # Import BRASS bedpe
    bedpe_df = pd.read_csv(brass_file, sep='\t', names=BRASS_HEADERS, comment='#').rename(columns={'# chr1': 'chr1'})
    bedpe_df['chr1'] = bedpe_df['chr1'].apply(lambda x: 'chr{}'.format(x))
    bedpe_df['chr2'] = bedpe_df['chr2'].apply(lambda x: 'chr{}'.format(x))

    # Correct sample name
    bedpe_df['sample'] = bedpe_df['sample'].apply(lambda v: v.split(',')[0])

    # SV larger than threshold
    bedpe_df = bedpe_df.query('bkdist == -1')

    # BRASS2 annotated SV
    if splitreads:
        bedpe_df = bedpe_df.query("assembly_score != '_'")

    # Copy-number flag positive
    if copynumber_flag:
        bedpe_df = bedpe_df.query('copynumber_flag == 1')

    # SVclass filter
    bedpe_df = bedpe_df.query("svclass == 'translocation'")

    # Assemble bed file
    bed_df = []

    for i in [1, 2]:
        bed_df.append(pd.concat([
            bedpe_df[f'chr{i}'].rename('#chr'),
            bedpe_df[[f'start{i}', f'end{i}']].mean(1).astype(int).rename('start'),
            bedpe_df[[f'start{i}', f'end{i}']].mean(1).astype(int).rename('end'),
            bedpe_df['svclass']
        ], axis=1))

    bed_df = pd.concat(bed_df)

    return bed_df


def copy_number_intersection_with_translocations(brass_bed_file, ascat_bed_file, distance=1e5):
    svs, segs = BedTool(brass_bed_file).sort(), BedTool(ascat_bed_file).sort()

    names = ['#chr', 'start', 'end', 'cn', 'sv_chr', 'sv_start', 'sv_end', 'svclass', 'distance']

    bed = segs.closest(svs, d=True, io=True).to_dataframe(names=names)
    bed = bed.query('distance != -1')
    bed = bed[bed['distance'] <= distance]

    return bed


def translocations_intersection_with_crispr(crispr_bed_file, ascat_brass_bed_file):
    # Import BED files
    seg_bed, crispr_bed = BedTool(ascat_brass_bed_file).sort(), BedTool(crispr_bed_file).sort()

    # Intersect SNP6 copy-number segments with CRISPR-Cas9 sgRNAs
    names = ['#chr', 'start', 'end', 'cn', 'chr', 'sgrna_start', 'sgrna_end', 'svclass', 'fc', 'sgrna_id']
    bed = seg_bed.intersect(crispr_bed, wa=True, wb=True).to_dataframe()

    # Calculate chromosome copies and cell ploidy
    chrm, ploidy = calculate_ploidy(seg_bed)

    bed = bed.assign(chrm=chrm[bed['chr']].values)
    bed = bed.assign(ploidy=ploidy)

    # Calculate copy-number ratio
    bed = bed.assign(ratio=bed.eval('cn / chrm'))

    # Calculate segment length
    bed = bed.assign(len=bed.eval('end - start'))

    return bed


if __name__ == '__main__':
    beds_df = []
    for sample in ['HCC38']:
        svclass = 'tandem-duplication'
        # svclass = 'deletion'

        brass_file = f'data/wgs/{sample}.brass.annot.bedpe'
        brass_bed_file = f'data/wgs/{sample}.brass.annot.{svclass}.bed'

        ascat_bed_file = f'data/wgs/{sample}.wgs.ascat.bed'

        ascat_brass_bed_file = f'data/wgs/{sample}.wgs.ascat.brass.{svclass}.bed'

        crispr_bed_file = f'data/crispr_bed/{sample}.crispr.bed'

        brass_bed(brass_file, svclass=svclass).to_csv(brass_bed_file, index=False, sep='\t')

        copy_number_intersection_with_svs(brass_bed_file, ascat_bed_file, svclass).to_csv(ascat_brass_bed_file, index=False, sep='\t')

        bed = segments_intersection_with_crispr(crispr_bed_file, ascat_brass_bed_file)

        beds_df.append(bed.assign(sample=sample))

    #
    plot_df = pd.concat(beds_df)
    plot_df = plot_df.assign(cn_bin=plot_df['cn'].apply(lambda v: bin_cnv(v, thresold=10)))
    plot_df = plot_df.assign(ratio_bin=plot_df['ratio'].apply(lambda v: bin_cnv(v, thresold=4)))

    seg_sgrnas_count = plot_df.groupby(['#chr', 'start', 'end', 'sample'])['svclass'].count()
    plot_df = plot_df.groupby(['#chr', 'start', 'end', 'sample']).agg({'fc': 'median', 'svclass': 'first', 'cn_bin': 'first', 'ratio_bin': 'first'})
    plot_df = plot_df.loc[seg_sgrnas_count[seg_sgrnas_count > 30].index]

    order = natsorted(set(plot_df['cn_bin']))
    sns.boxplot('cn_bin', 'fc', 'svclass', data=plot_df, notch=True, order=order, linewidth=.3, fliersize=.1)
    plt.legend(frameon=False)
    plt.gcf().set_size_inches(3, 3)
    plt.savefig('reports/wgs_svs_fc_boxplots.png', bbox_inches='tight', dpi=600)
    plt.close('all')

    #
    sample, chrm = 'HCC38', 'chr16'

    #
    plot_rearrangements_seg(sample, chrm)
    plt.suptitle(f'{sample}', fontsize=10)
    plt.gcf().set_size_inches(8, 3)
    plt.savefig('reports/chromosomeplot_svs.png', bbox_inches='tight', dpi=600)
    plt.close('all')

    #
    ax, plot_df = plot_chromosome_seg(sample, chrm)
    plt.gcf().set_size_inches(8, 2)
    plt.savefig('reports/chromosomeplot.png', bbox_inches='tight', dpi=600)
    plt.close('all')
