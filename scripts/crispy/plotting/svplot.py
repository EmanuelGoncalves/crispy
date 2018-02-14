#!/usr/bin/env python
# Copyright (C) 2018 Emanuel Goncalves

import pickle
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as plticker
from matplotlib.gridspec import GridSpec
from crispy import bipal_dbgd
from matplotlib.patches import Arc
from collections import OrderedDict
from crispy.ratio import BRASS_HEADERS
from scripts.crispy.processing.correct_cnv_bias import assemble_matrix


PALETTE = {
    'tandem-duplication': '#377eb8', 'deletion': '#e41a1c', 'translocation': '#984ea3', 'inversion_h_h': '#4daf4a', 'inversion_t_t': '#ff7f00'
}

CHR_SIZES_HG19 = {
    'chr1': 249250621, 'chr2': 243199373, 'chr3': 198022430, 'chr4': 191154276, 'chr5': 180915260, 'chr6': 171115067, 'chr7': 159138663, 'chr8': 146364022,
    'chr9': 141213431, 'chr10': 135534747, 'chr11': 135006516, 'chr12': 133851895, 'chr13': 115169878, 'chr14': 107349540, 'chr15': 102531392, 'chr16': 90354753,
    'chr17': 81195210, 'chr18': 78077248, 'chr19': 59128983, 'chr20': 63025520, 'chr21': 48129895, 'chr22': 51304566, 'chrX': 155270560, 'chrY': 59373566
}


def svtype(strand1, strand2, svclass):
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

    # Parse chromosome name
    bedpe_df = bedpe_df.assign(chr1=bedpe_df['chr1'].apply(lambda x: 'chr{}'.format(x)).values)
    bedpe_df = bedpe_df.assign(chr2=bedpe_df['chr2'].apply(lambda x: 'chr{}'.format(x)).values)

    return bedpe_df


def plot_rearrangements(brass, crispr, ngsc, chrm, winsize=1e5, chrm_size=None, xlim=None, scale=1e6):
    chrm_size = CHR_SIZES_HG19 if chrm_size is None else chrm_size

    #
    xlim = (0, chrm_size[chrm]) if xlim is None else xlim

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
    f, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True, sharey=False, gridspec_kw={'height_ratios': [1, 2, 2]})

    #
    ax1.axhline(0.0, lw=.3, color=bipal_dbgd[0])
    ax1.set_ylim(-1, 1)

    #
    ax2.scatter(ngsc_['location'] / scale, ngsc_['absolute_cn'], s=2, alpha=.5, c=bipal_dbgd[0], label='copy-number', zorder=1)
    ax2.set_ylim(0.0, np.ceil(ngsc_['absolute_cn'].quantile(0.99)))

    #
    ax3.scatter(crispr_['location'] / scale, crispr_['crispr'], s=2, alpha=.5, c=bipal_dbgd[1], label='crispr', zorder=1)
    ax3.axhline(0.0, lw=.3, color=bipal_dbgd[0])

    #
    if 'cnv' in crispr.columns:
        ax2.scatter(crispr_['location'] / scale, crispr_['cnv'], s=3, alpha=.5, c='#d9d9d9', zorder=2, label='ASCAT')

    if 'k_mean' in crispr.columns:
        ax3.scatter(crispr_['location'] / scale, crispr_['k_mean'], s=3, alpha=.5, c='#d9d9d9', zorder=2, label='Crispy')

    #
    for c1, s1, e1, c2, s2, e2, st1, st2, sv in brass_[['chr1', 'start1', 'end1', 'chr2', 'start2', 'end2', 'strand1', 'strand2', 'svclass']].values:
        stype = svtype(st1, st2, sv)
        stype_col = PALETTE[stype]

        x1_mean, x2_mean = np.mean([s1, e1]), np.mean([s2, e2])

        # Plot arc
        if c1 == c2:
            angle = 0 if stype in ['tandem-duplication', 'deletion'] else 180

            xy = (np.mean([x1_mean, x2_mean]) / scale, 0)

            ax1.add_patch(Arc(xy, (x2_mean - x1_mean) / scale, 1., angle=angle, theta1=0, theta2=180, edgecolor=stype_col, lw=.1))

        # Plot segments
        for ymin, ymax, ax in [(-1, 0.5, ax1), (-1, 1, ax2), (0, 1, ax3)]:
            if (c1 == chrm) and (xlim[0] <= x1_mean <= xlim[1]):
                ax.axvline(
                    x=x1_mean / scale, ymin=ymin, ymax=ymax, c=stype_col, linewidth=.1, zorder=0, clip_on=False, label=stype
                )

            if (c2 == chrm) and (xlim[0] <= x2_mean <= xlim[1]):
                ax.axvline(
                    x=x2_mean / scale, ymin=ymin, ymax=ymax, c=stype_col, linewidth=.1, zorder=0, clip_on=False, label=stype
                )

        # Translocation label
        if stype == 'translocation':
            if (c1 == chrm) and (xlim[0] <= x1_mean <= xlim[1]):
                ax1.text(x1_mean / scale, 0, ' to {}'.format(c2), color=stype_col, ha='center', fontsize=3, rotation=90, va='bottom')

            if (c2 == chrm) and (xlim[0] <= x2_mean <= xlim[1]):
                ax1.text(x2_mean / scale, 0, ' to {}'.format(c1), color=stype_col, ha='center', fontsize=3, rotation=90, va='bottom')

    #
    by_label = {l: p for p, l in zip(*(ax2.get_legend_handles_labels())) if l in PALETTE}
    ax1.legend(by_label.values(), by_label.keys(), loc='center left', bbox_to_anchor=(1.02, 0.5), prop={'size': 4})

    by_label = {l: p for p, l in zip(*(ax2.get_legend_handles_labels())) if l not in PALETTE}
    ax2.legend(by_label.values(), by_label.keys(), loc='center left', bbox_to_anchor=(1.02, 0.5), prop={'size': 4})

    by_label = {l: p for p, l in zip(*(ax3.get_legend_handles_labels())) if l not in PALETTE}
    ax3.legend(by_label.values(), by_label.keys(), loc='center left', bbox_to_anchor=(1.02, 0.5), prop={'size': 4})

    #
    ax1.axis('off')

    #
    ax2.yaxis.set_major_locator(plticker.MultipleLocator(base=2.))
    ax3.yaxis.set_major_locator(plticker.MultipleLocator(base=2.))

    #
    ax1.set_ylabel('SV')
    ax2.set_ylabel('Copy-number')
    ax3.set_ylabel('CRISPR')

    #
    plt.xlabel('Position on chromosome {} (Mb)'.format(chrm.replace('chr', '')))

    #
    plt.xlim(xlim[0] / scale, xlim[1] / scale)

    return ax1, ax2, ax3


if __name__ == '__main__':
    # - Import CRISPR
    cnv = pd.read_csv('data/crispy_copy_number_gene_snp.csv', index_col=0)
    kmean = assemble_matrix('data/crispy/gdsc/', 'k_mean')
    sgrna_lib = pd.read_csv('data/crispr_libs/KY_Library_v1.1_updated.csv', index_col=0)
    crispr_sgrna = pd.read_csv('data/crispr_gdsc_sgrna_logfc.csv', index_col=0)
    nexp = pickle.load(open('data/gdsc/nexp_pickle.pickle', 'rb'))
    ss = pd.read_csv('data/gdsc/samplesheet.csv', index_col=0)

    samples = ['HCC1954', 'HCC1143', 'HCC38', 'HCC1187', 'HCC1937', 'HCC1395']
    # samples = ['LS-1034', 'HCC1143', 'HCC38', 'HCC1187', 'HCC1937', 'HCC1395']
    # samples = set(samples).intersection(nexp)

    #
    sample, chrm, = 'HCC1187', 'chr1'

    #
    bedpe = import_brass_bedpe('data/gdsc/wgs/brass_bedpe/{}.brass.annot.bedpe'.format(sample), bkdist=None, splitreads=True)

    #
    ngsc = pd.read_csv(
        'data/gdsc/wgs/brass_intermediates/{}.ngscn.abs_cn.bg'.format(sample), sep='\t', header=None,
        names=['chr', 'start', 'end', 'absolute_cn', 'norm_logratio'], dtype={'chr': str, 'start': int, 'end': int, 'cn': float, 'score': float}
    )
    ngsc = ngsc.assign(chr=ngsc['chr'].apply(lambda x: 'chr{}'.format(x)).values)

    # -
    print('Sample: {}; Chrm: {}'.format(sample, chrm))

    crispr = pd.concat([crispr_sgrna[sample].rename('crispr'), sgrna_lib], axis=1).dropna()
    # crispr = crispr[crispr['gene'].isin(nexp[sample])]
    crispr = crispr.assign(snp=cnv[sample].reindex(crispr['gene']).values)
    crispr = crispr.assign(kmean=kmean[sample].reindex(crispr['gene']).values)
    # crispr = crispr.groupby('gene').agg({'start': np.min, 'end': np.max, 'chr': 'first', 'crispr': np.mean})

    ax1, ax2, ax3 = plot_rearrangements(bedpe, crispr, ngsc, chrm)

    plt.suptitle('{}'.format(sample))

    #
    plt.gcf().set_size_inches(8, 3)
    plt.savefig('reports/crispy/svplot_{}_{}.pdf'.format(sample, chrm), bbox_inches='tight')
    plt.close('all')
