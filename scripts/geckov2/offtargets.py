#!/usr/bin/env python
# Copyright (C) 2018 Emanuel Goncalves

import geckov2
import numpy as np
import crispy as cy
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def offtarget_boxplots(plot_df):
    plot_df = plot_df.assign(ncuts_bin=plot_df['ncuts'].apply(cy.Utils.bin_cnv, args=(6,)))
    plot_df = plot_df[[ncuts == nchrs if dist == -1 else True for ncuts, nchrs, dist in plot_df[['ncuts', 'nchrs', 'distance']].values]]
    plot_df = plot_df[plot_df['ngenes'] == 1]

    hue_order = ['1-10 kb', '10-100 kb', '0.1-1 Mb', '1-10 Mb', '>10 Mb', 'diff. chr.']

    ax = cy.QCplot.bias_boxplot(plot_df, x='ncuts_bin', y='mean_fc', hue='distance_bin', hue_order=hue_order, tick_base=1, add_n=True)

    ax.axhline(0, ls=':', c=cy.QCplot.PAL_DBGD[0], lw=.3, zorder=0)
    ax.axhline(-1, ls=':', c=cy.QCplot.PAL_DBGD[0], lw=.3, zorder=0)

    ax.set_xlabel('Number of cuts')
    ax.set_ylabel('CRISPR-Cas9 fold-change\n(scaled)')

    ax.set_title('GeCKOv2 guides with off-target activity')

    ax.legend(loc='center left', title='Cuts distance', bbox_to_anchor=(1, 0.5), frameon=False, prop={'size': 7})\
        .get_title().set_fontsize('7')


if __name__ == '__main__':
    # - Imports
    # Samplesheet
    samplesheet = geckov2.get_samplesheet()

    # GeCKOv2 library
    lib = geckov2.get_crispr_library()

    # Geckov2 off-targets
    offtargets = geckov2.lib_off_targets()

    # Data-sets
    raw_counts = geckov2.get_raw_counts()

    # Gene-sets
    essential = cy.Utils.get_essential_genes()
    essential_guides = set(lib[lib['gene'].isin(essential)]['sgrna'])

    nessential = cy.Utils.get_non_essential_genes()
    nessential_guides = set(lib[lib['gene'].isin(nessential)]['sgrna'])

    # - Sample manifest
    manifest = {
        v: [c for c in raw_counts if c.split(' Rep')[0] == i] for i, v in samplesheet['name'].iteritems()
    }

    # - sgRNA fold-change
    fc = {
        s: cy.Crispy(
            raw_counts=raw_counts[manifest[s] + [geckov2.PLASMID]],
            plasmid=geckov2.PLASMID
        ).fold_changes(qc_replicates_thres=None) for s in manifest
    }
    fc = pd.DataFrame(fc)

    # - Scale fold-changes
    fc = cy.Crispy.scale_crispr(fc, ess=essential_guides, ness=nessential_guides)

    # - Geckov2 offtarget guides fold-changes grouped by distance
    plot_df = pd.concat([offtargets, fc.loc[offtargets.index].mean(1).rename('mean_fc')], axis=1)
    offtarget_boxplots(plot_df)
    plt.gcf().set_size_inches(3, 3)
    plt.savefig(f'reports/offtargets_distance_boxplot.pdf', bbox_inches='tight', transparent=True)
    plt.close('all')
