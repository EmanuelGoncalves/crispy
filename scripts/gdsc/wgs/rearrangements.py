#!/usr/bin/env python
# Copyright (C) 2018 Emanuel Goncalves

import gdsc
import numpy as np
import crispy as cy
import pandas as pd
import matplotlib.pyplot as plt


def svcount_barplot(dfs):
    plot_df = pd.concat([dfs[k] for k in dfs])

    plot_df = plot_df.query("assembly_score != '_'")['svclass'].value_counts().to_frame().sort_values('svclass')
    plot_df = plot_df.assign(ypos=np.arange(plot_df.shape[0]))
    plot_df.index = [i.capitalize() for i in plot_df.index]

    plt.barh(plot_df['ypos'], plot_df['svclass'], .8, color=cy.QCplot.PAL_DBGD[0], align='center')

    plt.xticks(np.arange(0, 1001, 250))
    plt.yticks(plot_df['ypos'])
    plt.yticks(plot_df['ypos'], plot_df.index)

    plt.xlabel('Count')
    plt.title('Breast carcinoma cell lines\nstructural rearrangements')

    plt.grid(color=cy.QCplot.PAL_DBGD[0], ls='-', lw=.1, axis='x')


if __name__ == '__main__':
    # - Imports
    crispy_beds = gdsc.import_crispy_beds(wgs=True)
    brass_bedpes = gdsc.import_brass_bedpes()
    ascat_beds = gdsc.get_copy_number_segments_wgs(as_dict=True)

    #
    svcount_barplot(brass_bedpes)
    plt.gcf().set_size_inches(2, 1)
    plt.savefig('reports/gdsc/wgs/brass_svs_counts_barplot_brca.pdf', bbox_inches='tight', transparent=True)
    plt.close('all')

    #
    svs = [
        ('HCC1143', 'chr6', (29.5e6, 35e6), -1.5, None),
        ('HCC1937', 'chr2', (120e6, 210e6), -2.5, 14),
        ('HCC1143', 'chr12', (45e6, 85e6), -2., 14),
        ('HCC1395', 'chr3', (100e6, 200e6), -2., 8),
        ('HCC1954', 'chr8', (60e6, 150e6), -2., 12),
        ('HCC1954', 'chr5', None, -2., 20),
        ('HCC38', 'chr5', (0, 100e6), -2., 8)
    ]

    #
    for s, chrm, xlim, ax3_ylim, ax2_ylim in svs:
        crispy_bed, brass_bedpe, ascat_bed = crispy_beds[s], brass_bedpes[s], ascat_beds[s]

        ax1, ax2, ax3 = cy.QCplot.plot_rearrangements(
            brass_bedpe, ascat_bed, crispy_bed, chrm, show_legend=False, xlim=xlim
        )

        if ax2_ylim is not None:
            ax2.set_ylim(0, ax2_ylim)

        y_lim = ax3.get_ylim()
        ax3.set_ylim(ax3_ylim, y_lim[1])

        plt.suptitle(f'{s}', y=1.05)

        plt.gcf().set_size_inches(3, 2)
        plt.savefig(f'reports/gdsc/wgs/svs_{s}_{chrm}.pdf', bbox_inches='tight', transparent=True)
        plt.close('all')

    # -
    chrms = [
        'chr1', 'chr2', 'chr3', 'chr4', 'chr5', 'chr6', 'chr7', 'chr8', 'chr9', 'chr10',
        'chr11', 'chr12', 'chr13', 'chr14', 'chr15', 'chr16', 'chr17', 'chr18', 'chr19',
        'chr20', 'chr21', 'chr22'
    ]

    for s in crispy_beds:
        for chrm in chrms:
            crispy_bed, brass_bedpe, ascat_bed = crispy_beds[s], brass_bedpes[s], ascat_beds[s]

            ax1, ax2, ax3 = cy.QCplot.plot_rearrangements(
                brass_bedpe, ascat_bed, crispy_bed, chrm, show_legend=False
            )

            if ax1 is not None:
                y_lim = ax3.get_ylim()
                ax3.set_ylim(-2.5, y_lim[1])

                plt.suptitle(f'{s}', y=1.05)

                plt.gcf().set_size_inches(3, 2)
                plt.savefig(f'reports/gdsc/wgs/all_svs/svs_{s}_{chrm}.pdf', bbox_inches='tight', transparent=True)
                plt.close('all')
