#!/usr/bin/env python
# Copyright (C) 2018 Emanuel Goncalves

import numpy as np
import pandas as pd
import scripts as mp
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as plticker
from pybedtools import BedTool
from crispy import PAL_DBGD
from crispy.utils import bin_cnv
from matplotlib.patches import Arc
from sklearn.metrics import roc_auc_score
from crispy.ratio import BRASS_HEADERS, GFF_FILE, GFF_HEADERS


PALETTE = {
    'tandem-duplication': '#377eb8',
    'deletion': '#e41a1c',
    'translocation': '#984ea3',
    'inversion': '#4daf4a',
    'inversion_h_h': '#4daf4a',
    'inversion_t_t': '#ff7f00',
}

CHR_SIZES_HG19 = {
    'chr1': 249250621, 'chr2': 243199373, 'chr3': 198022430, 'chr4': 191154276, 'chr5': 180915260, 'chr6': 171115067, 'chr7': 159138663, 'chr8': 146364022,
    'chr9': 141213431, 'chr10': 135534747, 'chr11': 135006516, 'chr12': 133851895, 'chr13': 115169878, 'chr14': 107349540, 'chr15': 102531392, 'chr16': 90354753,
    'chr17': 81195210, 'chr18': 78077248, 'chr19': 59128983, 'chr20': 63025520, 'chr21': 48129895, 'chr22': 51304566, 'chrX': 155270560, 'chrY': 59373566
}

BKDIST_ORDER = ['1-10 kb', '1-100 kb', '0.1-1 Mb', '1-10 Mb', '>10 Mb']


def svcount_barplot(brass):
    plot_df = brass.query("assembly_score != '_'")['svclass'].value_counts().to_frame().sort_values('svclass')
    plot_df = plot_df.assign(ypos=np.arange(plot_df.shape[0]))
    plot_df.index = [i.capitalize() for i in plot_df.index]

    plt.barh(plot_df['ypos'], plot_df['svclass'], .8, color=PAL_DBGD[0], align='center')

    plt.xticks(np.arange(0, 1001, 250))
    plt.yticks(plot_df['ypos'])
    plt.yticks(plot_df['ypos'], plot_df.index)

    plt.xlabel('Count')
    plt.title('Breast carcinoma cell lines\nstructural rearrangements')

    plt.legend(loc=4, prop={'size': 4})

    plt.grid(color=PAL_DBGD[0], ls='-', lw=.1, axis='x')


def aucs_samples(plot_df, order, min_events=5):
    y_aucs = []
    for sample in samples:
        df = plot_df.query("sample == '{}'".format(sample)).dropna()

        for t in order:
            y_true, y_score = df['svclass'].map(lambda x: t in set(x.split(','))).astype(int), df['ratio']

            if sum(y_true) >= min_events and len(set(y_true)) > 1:
                y_aucs.append({
                    'sample': sample, 'svclass': t, 'auc': roc_auc_score(y_true, y_score)
                })

    y_aucs = pd.DataFrame(y_aucs)

    pal = sns.light_palette(PAL_DBGD[0], len(order) + 1).as_hex()[1:]

    sns.boxplot(
        'auc', 'svclass', data=y_aucs, orient='h', order=order, palette=pal, notch=True, linewidth=.5, fliersize=1
    )
    plt.axvline(.5, lw=.3, c=PAL_DBGD[0])
    plt.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), prop={'size': 6})
    plt.title('Structural rearrangements relation with\ncopy-number ratio')
    plt.xlabel('AROC')
    plt.ylabel('')


if __name__ == '__main__':
    # - Copy-number
    ratios = pd.read_csv(mp.CN_GENE_RATIO.format('wgs'), index_col=0)
    ploidy = pd.read_csv(mp.CN_PLOIDY.format('wgs'), index_col=0, names=['sample', 'ploidy'])['ploidy']

    # - BRASS
    brass = pd.concat([
        import_brass_bedpe('{}/{}.brass.annot.bedpe'.format(mp.WGS_BRASS_BEDPE, sample)).assign(sample=sample) for sample in mp.BRCA_SAMPLES
    ])

    # Count number of SVs
    svcount_barplot(brass)
    plt.gcf().set_size_inches(2, 1)
    plt.savefig('reports/crispy/brass_svs_counts_barplot_brca.png', bbox_inches='tight', dpi=600)
    plt.close('all')

    # - Samples
    samples = list(ratios)

    # - Map structural rearrangements to copy-number ratio segments
    # Gene gff file
    gff = BedTool(GFF_FILE).sort()

    # Map to samples
    ratio_svs = []
    # sample = 'HCC38'
    for sample in samples:
        # Import BRASS SV
        svs_file = '{}/{}.brass.annot.bedpe.{}'.format(mp.WGS_BRASS_BEDPE, sample, 'bed')

        svs_df = import_brass_bedpe('{}/{}.brass.annot.bedpe'.format(mp.WGS_BRASS_BEDPE, sample), bkdist=0, splitreads=True, convert_to_bed=True)
        svs_df.to_csv(svs_file, sep='\t', index=False)

        svs = BedTool(svs_file).sort()

        # Map ASCAT copy-number ratio segments to BRASS structural rearrangements
        gene_bkdist = gff.map(svs, c=5, o='collapse').to_dataframe(names=GFF_HEADERS + ['bkdist'])

        df = gff.map(svs, c=4, o='collapse,count').to_dataframe(names=GFF_HEADERS + ['svclass', 'count'])

        df = df.assign(bkdist=gene_bkdist['bkdist'].values)

        df = df.assign(ploidy=bin_cnv(ploidy[sample], 5))

        df = df.assign(ratio=ratios.loc[df['feature'], sample].values)

        df = df.assign(sample=sample)

        ratio_svs.append(df)

    ratio_svs = pd.concat(ratio_svs).reset_index(drop=True).dropna()

    # - Copy-number ratio structural rearrangements enrichment
    order = ['deletion', 'inversion', 'tandem-duplication']
    plot_df = ratio_svs.query('count <= 1').copy()
    print(plot_df.sort_values('ratio', ascending=False).head(30))

    # - Plot AUC boxplots
    aucs_samples(plot_df, order)
    plt.gcf().set_size_inches(3, 3)
    plt.savefig('reports/crispy/brass_sv_ratio_boxplot_auc.png', bbox_inches='tight', dpi=600)
    plt.close('all')

