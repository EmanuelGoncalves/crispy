#!/usr/bin/env python
# Copyright (C) 2018 Emanuel Goncalves

import gdsc
import numpy as np
import pandas as pd
import crispy as cy
import seaborn as sns
import matplotlib.pyplot as plt


if __name__ == '__main__':
    # - Crispy bed files
    crispy_beds = gdsc.import_crispy_beds()
    ascat_beds = gdsc.get_copy_number_segments()

    # - Chromosome plots
    chrs_to_plot = [
        ('HCC1954', 'chr8', (100, 145), ['MYC']),
        ('NCI-H2087', 'chr8', (100, 145), ['MYC'])
    ]

    for s, chrm, xlim, genes in chrs_to_plot:
        crispy_bed, ascat_bed = crispy_beds[s], ascat_beds.query(f"sample == '{s}'")

        ax = cy.QCplot.plot_chromosome(crispy_bed, ascat_bed, chrm, legend=True, highlight=genes)

        if xlim is not None:
            ax.set_xlim(xlim)

        ax.set_title(s)
        ax.set_xlabel(f'Position on chromosome {chrm} (Mb)')

        plt.gcf().set_size_inches(3, 2)
        plt.savefig(f'reports/gdsc/chrm_plot_{s}_{chrm}.pdf', bbox_inches='tight', transparent=True)
        plt.close('all')
