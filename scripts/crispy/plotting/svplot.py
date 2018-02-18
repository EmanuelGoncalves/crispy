#!/usr/bin/env python
# Copyright (C) 2018 Emanuel Goncalves

import numpy as np
import pandas as pd
import seaborn as sns
import crispy.ratio as cr
import matplotlib.pyplot as plt
import matplotlib.ticker as plticker
from pybedtools import BedTool
from crispy import bipal_dbgd
from matplotlib.patches import Arc
from crispy.ratio import BRASS_HEADERS
from sklearn.metrics import roc_curve, auc


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


def svtype(strand1, strand2, svclass, unfold_inversions):
    if svclass == 'translocation':
        svtype = 'translocation'

    elif strand1 == '+' and strand2 == '+':
        svtype = 'deletion'

    elif strand1 == '-' and strand2 == '-':
        svtype = 'tandem-duplication'

    elif strand1 == '+' and strand2 == '-':
        svtype = 'inversion_h_h' if unfold_inversions else 'inversion'

    elif strand1 == '-' and strand2 == '+':
        svtype = 'inversion_t_t' if unfold_inversions else 'inversion'

    else:
        assert False, 'SV class not recognised: strand1 == {}; strand2 == {}; svclass == {}'.format(strand1, strand2, svclass)

    return svtype


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


def plot_rearrangements(brass, crispr, ngsc, chrm, winsize=1e5, chrm_size=None, xlim=None, scale=1e6, unfold_inversions=False):
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
    ax2.scatter(ngsc_['location'] / scale, ngsc_['absolute_cn'], s=1, alpha=.5, c=bipal_dbgd[0], label='copy-number', zorder=1)
    ax2.set_ylim(0.0, np.ceil(ngsc_['absolute_cn'].quantile(0.99)))

    #
    ax3.scatter(crispr_['location'] / scale, crispr_['crispr'], s=1, alpha=.5, c=bipal_dbgd[1], label='crispr', zorder=1)
    ax3.axhline(0.0, lw=.3, color=bipal_dbgd[0])

    #
    if 'cnv' in crispr.columns:
        ax2.scatter(crispr_['location'] / scale, crispr_['cnv'], s=3, alpha=.5, c='#d9d9d9', zorder=2, label='ASCAT')

    if 'k_mean' in crispr.columns:
        ax3.scatter(crispr_['location'] / scale, crispr_['k_mean'], s=3, alpha=.5, c='#d9d9d9', zorder=2, label='Crispy')

    #
    for c1, s1, e1, c2, s2, e2, st1, st2, sv in brass_[['chr1', 'start1', 'end1', 'chr2', 'start2', 'end2', 'strand1', 'strand2', 'svclass']].values:
        stype = svtype(st1, st2, sv, unfold_inversions)
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


def svcount_barplot(brass):
    plot_df = pd.concat([
        brass['svclass'].value_counts().to_frame(),
        brass.query("assembly_score != '_'")['svclass'].value_counts().rename('svclass2')
    ], axis=1).sort_values('svclass')
    plot_df = plot_df.assign(ypos=np.arange(plot_df.shape[0]))

    pal = sns.light_palette(bipal_dbgd[0], n_colors=3, reverse=False).as_hex()[1:]

    for i, c in enumerate(['svclass', 'svclass2']):
        plt.barh(plot_df['ypos'], plot_df[c], .8, color=pal[i], align='center', label='BRASS' if c == 'svclass' else 'BRASS2')

    plt.yticks(plot_df['ypos'])
    plt.yticks(plot_df['ypos'], plot_df.index)

    plt.xlabel('Count')
    plt.title('BRCA cell lines\nstructural rearrangements')

    plt.legend(loc=4, prop={'size': 4})

    plt.grid(color=pal[0], ls='-', lw=.1, axis='x')


def brass_bedpe_to_bed(bedpe_df):
    return pd.concat([
        bedpe_df['chr1'].rename('#chr'),
        bedpe_df[['start1', 'end1']].mean(1).astype(int).rename('start'),
        bedpe_df[['start2', 'end2']].mean(1).astype(int).rename('end'),
        bedpe_df['svclass'].rename('svclass')
    ], axis=1)


def import_brass_bedpe(bedpe_file, bkdist=None, splitreads=False, convert_to_bed=False):
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

    # Conver SVs to bed
    if convert_to_bed:
        bedpe_df = brass_bedpe_to_bed(bedpe_df)

    return bedpe_df


def plot_sv_ratios_arocs(plot_df):
    order = ['deletion', 'inversion', 'tandem-duplication', '.']

    ax = plt.gca()

    for t, c in zip(order, sns.light_palette(bipal_dbgd[0], len(order) + 1).as_hex()[1:]):
        y_true, y_score = plot_df['svclass'].map(lambda x: t in set(x.split(','))).astype(int), plot_df['ratio']

        fpr, tpr, _ = roc_curve(y_true, y_score)

        ax.plot(fpr, tpr, label='{0:} (AUC={1:.2f})'.format(t.capitalize(), auc(fpr, tpr)), lw=1., c=c)

    ax.plot((0, 1), (0, 1), 'k--', lw=.3, alpha=.5)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel('False positive rate')
    ax.set_ylabel('True positive rate')
    ax.set_title('Structural rearrangements relation with\ncopy-number ratio')

    ax.legend(loc=4, prop={'size': 6})

    return ax


if __name__ == '__main__':
    samples = ['HCC1954', 'HCC1143', 'HCC38', 'HCC1187', 'HCC1937', 'HCC1395']

    ascat_dir = 'data/gdsc/wgs/ascat_bed/{}.wgs.ascat.bed'
    brass_dir = 'data/gdsc/wgs/brass_bedpe/{}.brass.annot.bedpe'

    # - BRASS
    brass = pd.concat([import_brass_bedpe(brass_dir.format(sample)).assign(sample=sample) for sample in samples])

    # Count number of SVs
    svcount_barplot(brass)
    plt.gcf().set_size_inches(2, 1)
    plt.savefig('reports/crispy/brass_svs_counts_barplot.png', bbox_inches='tight', dpi=600)
    plt.close('all')

    # - Map structural rearrangements to copy-number ratio segments
    ratio_svs = []
    # sample = 'HCC38'
    for sample in samples:
        # Import ASCAT segments and calculate ratios
        seg_file = ascat_dir.format(sample) + '.ratio.bed'

        seg_df = cr.import_bed_datframe(ascat_dir.format(sample)).assign(sample=sample)
        seg_df.to_csv(seg_file, sep='\t', index=False)

        seg = BedTool(seg_file).sort()

        # Import BRASS SV
        svs_file = brass_dir.format(sample) + '.bed'

        svs_df = import_brass_bedpe(brass_dir.format(sample), bkdist=0, splitreads=False, convert_to_bed=True)
        svs_df.to_csv(svs_file, sep='\t', index=False)

        svs = BedTool(svs_file).sort()

        # Map ASCAT copy-number ratio segments to BRASS structural rearrangements
        seg_svs = seg.map(svs, c=4, o='collapse,count').to_dataframe(names=list(seg_df.columns) + ['svclass', 'count'])

        # Append
        ratio_svs.append(seg_svs)

    ratio_svs = pd.concat(ratio_svs).reset_index()
    ratio_svs = ratio_svs.assign(length=ratio_svs.eval('end - start').apply(bin_bkdist).values)
    print(ratio_svs.sort_values('ratio'))

    # - Copy-number ratio structural rearrangements enrichment
    # AROCs
    ax = plot_sv_ratios_arocs(ratio_svs)

    plt.gcf().set_size_inches(3, 3)
    plt.savefig('reports/crispy/brass_sv_ratio_cumdist.png', bbox_inches='tight', dpi=600)
    plt.close('all')
