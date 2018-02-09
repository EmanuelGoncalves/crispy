#!/usr/bin/env python
# Copyright (C) 2018 Emanuel Goncalves

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from crispy import bipal_dbgd
from collections import OrderedDict
from crispy.ratio import BRASS_HEADERS

PALETTE = {
    'tandem-duplication': '#377eb8', 'deletion': '#e41a1c', 'translocation': '#984ea3', 'inversion_h_h': '#4daf4a', 'inversion_t_t': '#ff7f00'
}

CHR_SIZES_HG19 = {
    'chr1': 249250621, 'chr2': 243199373, 'chr3': 198022430, 'chr4': 191154276, 'chr5': 180915260, 'chr6': 171115067, 'chr7': 159138663, 'chr8': 146364022,
    'chr9': 141213431, 'chr10': 135534747, 'chr11': 135006516, 'chr12': 133851895, 'chr13': 115169878, 'chr14': 107349540, 'chr15': 102531392, 'chr16': 90354753,
    'chr17': 81195210, 'chr18': 78077248, 'chr19': 59128983, 'chr20': 63025520, 'chr21': 48129895, 'chr22': 51304566, 'chrX': 155270560, 'chrY': 59373566
}


def svtype(strand1, strand2, svclass):
    svtype = None

    if svclass == 'translocation':
        svtype = 'translocation'

    elif strand1 == '+' and strand2 == '+':
        svtype = 'deletion'

    elif strand1 == '-' and strand2 == '-':
        svtype = 'tandem-duplication'

    elif strand1 == '+' and strand2 == '-':
        svtype = 'inversion_h_h'

    elif strand1 == '-' and strand2 == '+':
        svtype = 'inversion_t_t'

    else:
        assert True, 'SV class not recognised: strand1 == {}; strand2 == {}; svclass == {}'.format(strand1, strand2, svclass)

    return svtype


def import_brass_bedpe(bedpe_file, bkdist, splitreads):
    # Import BRASS bedpe
    bedpe_df = pd.read_csv(bedpe_file, sep='\t', names=BRASS_HEADERS, comment='#')

    # Correct sample name
    bedpe_df['sample'] = bedpe_df['sample'].apply(lambda v: v.split(',')[0])

    # SV larger than threshold
    if bkdist is not None:
        bedpe_df = bedpe_df[bedpe_df['bkdist'] >= bkdist]

    # BRASS2 annotated SV
    if splitreads:
        bedpe_df = bedpe_df.query("assembly_score != '_'")

    return bedpe_df


def plot_rearrangements(brass, crispr, ngsc, chrm, winsize=1e5, chrm_size=None):
    chrm_size = CHR_SIZES_HG19 if chrm_size is None else chrm_size

    #
    intervals = np.arange(1, chrm_size[chrm] + winsize, winsize)

    # BRASS
    brass_ = brass[(brass['chr1'] == chrm) | (brass['chr2'] == chrm)]

    # NGSC
    ngsc_ = ngsc[ngsc['chr'] == chrm]
    ngsc_ = ngsc_[ngsc_['norm_logratio'] <= 2]
    ngsc_ = ngsc_.assign(location=ngsc_[['start', 'end']].mean(1))
    ngsc_ = ngsc_.assign(interval=pd.cut(ngsc_['location'], intervals, include_lowest=True, right=False))
    ngsc_ = ngsc_.groupby('interval').mean()

    # CRISPR
    crispr_ = crispr[crispr['chr'] == chrm]
    crispr_ = crispr_.assign(location=crispr_[['start', 'end']].mean(1))
    # crispr_ = crispr_.assign(interval=pd.cut(crispr_['location'], intervals, include_lowest=True, right=False))
    # crispr_ = crispr_.groupby('interval').mean()

    #
    if brass_.shape[0] == 0:
        return None

    # -
    f, (ax1, ax2) = plt.subplots(2, 1, sharex=True, sharey=False)

    #
    ax1.scatter(ngsc_['location'], ngsc_['absolute_cn'], s=2, alpha=.5, c=bipal_dbgd[0], label='copy-number')
    ax1.set_ylim(0.0, np.ceil(ngsc_['absolute_cn'].quantile(0.99)))

    #
    ax2.scatter(crispr_[['start', 'end']].mean(1), crispr_['crispr'], s=2, alpha=.5, c=bipal_dbgd[1], label='crispr')

    #
    for c1, s1, e1, c2, s2, e2, st1, st2, sv in brass_[['chr1', 'start1', 'end1', 'chr2', 'start2', 'end2', 'strand1', 'strand2', 'svclass']].values:
        stype = svtype(st1, st2, sv)
        stype_col = PALETTE[stype]

        for ymin, ymax, ax in [(-1.2, 1, ax1), (0, 1.2, ax2)]:

            if c1 == chrm:
                ax.axvline(
                    x=np.mean([s1, e1]), ymin=ymin, ymax=ymax, c=stype_col, linewidth=.3, zorder=0, clip_on=False, label=stype
                )

            if c2 == chrm:
                ax.axvline(
                    x=np.mean([s2, e2]), ymin=ymin, ymax=ymax, c=stype_col, linewidth=.3, zorder=0, clip_on=False, label=stype
                )

    #
    plt.xlim(-winsize, chrm_size[chrm] + winsize)

    #
    by_label = {l: p for ax in [ax1, ax2] for p, l in zip(*(ax.get_legend_handles_labels()))}
    plt.legend(by_label.values(), by_label.keys(), loc='center left', bbox_to_anchor=(1.04, 0.5))

    #
    plt.xlabel('{}'.format(chrm))

    return plt.gcf()


if __name__ == '__main__':
    # - Import CRISPR
    sgrna_lib = pd.read_csv('data/crispr_libs/KY_Library_v1.1_updated.csv', index_col=0)
    crispr_sgrna = pd.read_csv('data/crispr_gdsc_sgrna_logfc.csv', index_col=0)

    samples = ['NCI-H2087', 'LS-1034', 'HCC1954', 'HCC1143', 'HCC38', 'HCC1187', 'HCC1937', 'HCC1395']

    # sample, chrm, = 'HCC38', 'chr1'
    for sample in samples:
        #
        bedpe = import_brass_bedpe('data/gdsc/wgs/brass_bedpe/HCC1187.brass.annot.bedpe', bkdist=None, splitreads=False)
        bedpe = bedpe.assign(chr1=bedpe['chr1'].apply(lambda x: 'chr{}'.format(x)).values)
        bedpe = bedpe.assign(chr2=bedpe['chr2'].apply(lambda x: 'chr{}'.format(x)).values)

        #
        ngsc = pd.read_csv(
            'data/gdsc/wgs/brass_intermediates/HCC1187.ngscn.abs_cn.bg', sep='\t', header=None,
            names=['chr', 'start', 'end', 'absolute_cn', 'norm_logratio'], dtype={'chr': str, 'start': int, 'end': int, 'cn': float, 'score': float}
        )
        ngsc = ngsc.assign(chr=ngsc['chr'].apply(lambda x: 'chr{}'.format(x)).values)

        # -
        for chrm in set(bedpe['chr1']):
            print('Sample: {}; Chrm: {}'.format(sample, chrm))

            crispr = pd.concat([crispr_sgrna[sample].rename('crispr'), sgrna_lib], axis=1).dropna()
            # crispr = crispr.groupby('gene').agg({'start': np.min, 'end': np.max, 'chr': 'first', 'crispr': np.mean})

            svp = plot_rearrangements(bedpe, crispr, ngsc, chrm)

            plt.xlim(0.25e8, 0.5e8)

            if svp is not None:
                plt.suptitle('{}'.format(sample))

                #
                plt.gcf().set_size_inches(6, 3)
                plt.savefig('reports/crispy/svplots/svplot_{}_{}.png'.format(sample, chrm), bbox_inches='tight', dpi=600)
                plt.close('all')
