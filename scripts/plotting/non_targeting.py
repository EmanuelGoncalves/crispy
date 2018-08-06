#!/usr/bin/env python
# Copyright (C) 2018 Emanuel Goncalves

import numpy as np
import pandas as pd
import crispy as cy
import scripts as mp
import seaborn as sns
import scipy.stats as st
import matplotlib.pyplot as plt


if __name__ == '__main__':
    # - Import
    growth = pd.read_csv('/Users/eg14/Projects/dtrace/data/gdsc/growth/growth_rate.csv', index_col=0)
    growth = growth['growth_rate_median'].dropna()

    ss = mp.get_samplesheet()

    # Manifest
    manifest = pd.read_csv(mp.MANIFEST, index_col='header').query('to_use')
    manifest = manifest[manifest['name'].isin(mp.get_sample_list())]

    # CRISPR-Cas9 lib
    lib = cy.get_crispr_lib()

    # Non-expressed genes
    nexp = mp.get_non_exp()

    # Copy-number ratios
    cnv_ratios = mp.get_copynumber_ratios()
    cnv_ratios = cnv_ratios.round(0).dropna().astype(int)

    # Ploidy
    ploidy = mp.get_ploidy()

    # - Un-normalised fold-changes
    # CRISPR-Cas9 raw counts
    counts = pd.read_csv(mp.CRISPR_RAW_COUNTS, index_col=0)
    counts.columns = [i.split('.')[0] for i in counts]

    # Build counts data-frame (discard sgRNAs with problems)
    counts = counts.drop(mp.QC_FAILED_SGRNAS, errors='ignore')

    # Add constant
    counts = counts + 1

    # Calculate log2 fold-changes: log2(sample / plasmid)
    fc = pd.DataFrame({c: np.log2(counts[c] / counts[manifest.loc[c, 'plasmid']]) for c in manifest.index})

    # Scale
    ess = cy.get_essential_genes()
    ess_metric = np.median(fc.reindex(lib[lib['GENES'].isin(ess)].index).dropna(), axis=0)

    ness = cy.get_non_essential_genes()
    ness_metric = np.median(fc.reindex(lib[lib['GENES'].isin(ness)].index).dropna(), axis=0)

    # -
    c_ctrl = fc.loc[[i for i in lib.index if i.startswith('CTRL0')]].dropna(how='all').dropna(axis=1)
    c_ctrl = c_ctrl.drop(set(manifest['plasmid']), axis=1, errors='ignore')
    c_ctrl = c_ctrl.loc[:, [c in manifest.index for c in c_ctrl]]

    c_nexp = pd.DataFrame({
        s: fc.loc[
            lib[lib['GENES'].isin(
                set(nexp[nexp[manifest.loc[s, 'name']] == 1].index).intersection(cnv_ratios[cnv_ratios[manifest.loc[s, 'name']] == 1].index)
            )].index, s
        ] for s in c_ctrl
    })

    c_nexp_amp = pd.DataFrame({
        s: fc.loc[
            lib[lib['GENES'].isin(
                set(nexp[nexp[manifest.loc[s, 'name']] == 1].index).intersection(cnv_ratios[cnv_ratios[manifest.loc[s, 'name']] > 1].index)
            )].index, s
        ] for s in c_ctrl
    })

    # -
    plot_df = pd.concat([
        c_ctrl.median().rename('control'),
        c_nexp.median().rename('nexp'),
        c_nexp_amp.median().rename('amp'),
    ], axis=1)
    plot_df = plot_df.assign(diff=plot_df.eval('control - nexp'))
    plot_df = plot_df.assign(diff_amp=plot_df.eval('control - amp'))
    plot_df = plot_df.assign(ploidy=ploidy[manifest.loc[plot_df.index, 'name']].values)
    plot_df = plot_df.assign(growth=growth[manifest.loc[plot_df.index, 'name']].values)

    # -
    sns.distplot(plot_df['diff'], color=cy.PAL_DBGD[0], kde=False, bins=15)

    plt.title('Non-targeting vs Non-expressed (ratio=1)\n(median log2 fold-changes)')
    plt.xlabel('Log2 fold-change median difference')

    plt.gcf().set_size_inches(3, 2)
    plt.savefig('reports/non_targeting_histogram.png', bbox_inches='tight', dpi=600)
    plt.close('all')

    # -
    marginal_kws, annot_kws = dict(kde=False), dict(stat='R')
    scatter_kws, line_kws = dict(edgecolor='w', lw=.3, s=10, alpha=.6), dict(lw=1., color=cy.PAL_DBGD[0], alpha=1.)

    joint_kws = dict(scatter_kws=scatter_kws, line_kws=line_kws)

    g = sns.jointplot(
        'diff', 'ploidy', data=plot_df, kind='reg', space=0, color=cy.PAL_DBGD[0], joint_kws=joint_kws,
        marginal_kws=marginal_kws, annot_kws=annot_kws
    )

    g.set_axis_labels('Non-targeting vs Non-expressed (ratio=1)\n(median log2 fold-changes)', 'Ploidy')

    g = g.annotate(st.pearsonr, frameon=False)

    plt.gcf().set_size_inches(3, 3)
    plt.savefig('reports/non_targeting_jointplot.png', bbox_inches='tight', dpi=600)
    plt.close('all')

    # -
    marginal_kws, annot_kws = dict(kde=False), dict(stat='R')
    scatter_kws, line_kws = dict(edgecolor='w', lw=.3, s=10, alpha=.6), dict(lw=1., color=cy.PAL_DBGD[0], alpha=1.)

    joint_kws = dict(scatter_kws=scatter_kws, line_kws=line_kws)

    g = sns.jointplot(
        'diff', 'growth', data=plot_df, kind='reg', space=0, color=cy.PAL_DBGD[0], joint_kws=joint_kws,
        marginal_kws=marginal_kws, annot_kws=annot_kws
    )

    g.set_axis_labels('Non-targeting vs Non-expressed (ratio=1)\n(median log2 fold-changes)', 'Growth rate')

    g = g.annotate(st.pearsonr, frameon=False)

    plt.gcf().set_size_inches(3, 3)
    plt.savefig('reports/non_targeting_jointplot_growth.png', bbox_inches='tight', dpi=600)
    plt.close('all')
