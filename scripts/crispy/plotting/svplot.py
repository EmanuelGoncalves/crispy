#!/usr/bin/env python
# Copyright (C) 2018 Emanuel Goncalves

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from crispy import bipal_dbgd
from matplotlib.patches import Arc
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
        assert False, 'SV class not recognised: strand1 == {}; strand2 == {}; svclass == {}'.format(strand1, strand2, svclass)

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


def plot_rearrangements(brass, crispr, ngsc, chrm, winsize=1e5, chrm_size=None, xlim=None, scale=1e6):
    chrm_size = CHR_SIZES_HG19 if chrm_size is None else chrm_size

    #
    winsize /= scale
    chrm_size = {k: v / scale for k, v in chrm_size.items()}

    #
    xlim = (0, chrm_size[chrm]) if xlim is None else xlim

    #
    intervals = np.arange(1, chrm_size[chrm] + winsize, winsize)

    # BRASS
    brass_ = brass[(brass['chr1'] == chrm) | (brass['chr2'] == chrm)]
    brass_['start1'] /= scale
    brass_['start2'] /= scale
    brass_['end1'] /= scale
    brass_['end2'] /= scale

    # NGSC
    ngsc_ = ngsc[ngsc['chr'] == chrm]
    ngsc_ = ngsc_[ngsc_['norm_logratio'] <= 2]
    ngsc_['start'] /= scale
    ngsc_['end'] /= scale
    ngsc_ = ngsc_.assign(location=ngsc_[['start', 'end']].mean(1))
    ngsc_ = ngsc_.assign(interval=pd.cut(ngsc_['location'], intervals, include_lowest=True, right=False))
    ngsc_ = ngsc_.groupby('interval').mean()

    # CRISPR
    crispr_ = crispr[crispr['chr'] == chrm]
    crispr_['start'] /= scale
    crispr_['end'] /= scale
    crispr_ = crispr_.assign(location=crispr_[['start', 'end']].mean(1))
    # crispr_ = crispr_.assign(interval=pd.cut(crispr_['location'], intervals, include_lowest=True, right=False))
    # crispr_ = crispr_.groupby('interval').mean()

    #
    if brass_.shape[0] == 0:
        return None

    # -
    f, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True, sharey=False, gridspec_kw={'height_ratios': [1, 2, 2]})

    #
    ax1.axhline(0.0, lw=.3, color=bipal_dbgd[0])
    ax1.set_ylim(-1, 1)

    #
    ax2.scatter(ngsc_['location'], ngsc_['absolute_cn'], s=2, alpha=.5, c=bipal_dbgd[0], label='copy-number')
    ax2.set_ylim(0.0, np.ceil(ngsc_['absolute_cn'].quantile(0.99)))

    #
    ax3.scatter(crispr_[['start', 'end']].mean(1), crispr_['crispr'], s=2, alpha=.5, c=bipal_dbgd[1], label='crispr')

    #
    for c1, s1, e1, c2, s2, e2, st1, st2, sv in brass_[['chr1', 'start1', 'end1', 'chr2', 'start2', 'end2', 'strand1', 'strand2', 'svclass']].values:
        stype = svtype(st1, st2, sv)
        stype_col = PALETTE[stype]

        x1_mean, x2_mean = np.mean([s1, e1]), np.mean([s2, e2])

        # Plot arc
        if c1 == c2:
            angle = 0 if stype in ['tandem-duplication', 'deletion'] else 180

            xy = (np.mean([x1_mean, x2_mean]), 0)

            ax1.add_patch(Arc(xy, x2_mean - x1_mean, 1., angle=angle, theta1=0, theta2=180, edgecolor=stype_col))

        # Plot segments
        for ymin, ymax, ax in [(-1, 0.5, ax1), (-1, 1, ax2), (0, 1, ax3)]:
            if (c1 == chrm) and (xlim[0] <= x1_mean <= xlim[1]):
                ax.axvline(
                    x=x1_mean, ymin=ymin, ymax=ymax, c=stype_col, linewidth=.3, zorder=0, clip_on=False, label=stype
                )

            if (c2 == chrm) and (xlim[0] <= x1_mean <= xlim[1]):
                ax.axvline(
                    x=x2_mean, ymin=ymin, ymax=ymax, c=stype_col, linewidth=.3, zorder=0, clip_on=False, label=stype
                )

    #
    plt.xlim(-winsize, chrm_size[chrm] + winsize)

    #
    by_label = {l: p for ax in [ax2, ax3] for p, l in zip(*(ax.get_legend_handles_labels()))}
    ax2.legend(by_label.values(), by_label.keys(), loc='center left', bbox_to_anchor=(1.04, 0.5))

    #
    plt.xlabel('{}'.format(chrm))

    #
    plt.xlim(xlim[0], xlim[1])

    return plt.gcf()


svp = plot_rearrangements(bedpe, crispr, ngsc, chrm)

plt.suptitle('{}'.format(sample))

plt.gcf().set_size_inches(6, 3)
plt.savefig('reports/crispy/svplot_{}_{}.pdf'.format(sample, chrm), bbox_inches='tight')
plt.close('all')


if __name__ == '__main__':
    # - Import CRISPR
    sgrna_lib = pd.read_csv('data/crispr_libs/KY_Library_v1.1_updated.csv', index_col=0)
    crispr_sgrna = pd.read_csv('data/crispr_gdsc_sgrna_logfc.csv', index_col=0)

    samples = ['NCI-H2087', 'LS-1034', 'HCC1954', 'HCC1143', 'HCC38', 'HCC1187', 'HCC1937', 'HCC1395']

    # sample, chrm, = 'HCC1143', 'chr17'
    for sample in samples:
        #
        bedpe = import_brass_bedpe('data/gdsc/wgs/brass_bedpe/{}.brass.annot.bedpe'.format(sample), bkdist=None, splitreads=False)
        bedpe = bedpe.assign(chr1=bedpe['chr1'].apply(lambda x: 'chr{}'.format(x)).values)
        bedpe = bedpe.assign(chr2=bedpe['chr2'].apply(lambda x: 'chr{}'.format(x)).values)

        #
        ngsc = pd.read_csv(
            'data/gdsc/wgs/brass_intermediates/{}.ngscn.abs_cn.bg'.format(sample), sep='\t', header=None,
            names=['chr', 'start', 'end', 'absolute_cn', 'norm_logratio'], dtype={'chr': str, 'start': int, 'end': int, 'cn': float, 'score': float}
        )
        ngsc = ngsc.assign(chr=ngsc['chr'].apply(lambda x: 'chr{}'.format(x)).values)

        # -
        for chrm in set(bedpe['chr1']):
            print('Sample: {}; Chrm: {}'.format(sample, chrm))

            crispr = pd.concat([crispr_sgrna[sample].rename('crispr'), sgrna_lib], axis=1).dropna()
            # crispr = crispr.groupby('gene').agg({'start': np.min, 'end': np.max, 'chr': 'first', 'crispr': np.mean})

            svp = plot_rearrangements(bedpe, crispr, ngsc, chrm)

            if svp is not None:
                plt.suptitle('{}'.format(sample))

                #
                plt.gcf().set_size_inches(6, 3)
                plt.savefig('reports/crispy/svplot_{}_{}.png'.format(sample, chrm), bbox_inches='tight', dpi=600)
                plt.close('all')
