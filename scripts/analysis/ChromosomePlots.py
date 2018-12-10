#!/usr/bin/env python
# Copyright (C) 2018 Emanuel Goncalves

import crispy as cy
import matplotlib.pyplot as plt
from analysis.DataImporter import DataImporter

if __name__ == '__main__':
    # - Crispy bed files
    data = DataImporter()

    samples = ['HCC1954', 'NCI-H2087']

    beds = data.get_crispy_beds(subset=samples)
    cnvs = data.get_copy_number()

    # - Chromosome plots
    chrs_to_plot = [
        ('HCC1954', 'chr8', (100, 145), ['MYC']),
        ('NCI-H2087', 'chr8', (100, 145), ['MYC'])
    ]

    for s, chrm, xlim, genes in chrs_to_plot:
        crispy_bed, ascat_bed = beds[s], cnvs.query(f"sample == '{s}'")

        ax = cy.QCplot.plot_chromosome(crispy_bed, ascat_bed, chrm, legend=True, highlight=genes)

        if xlim is not None:
            ax.set_xlim(xlim)

        ax.set_title(s)
        ax.set_xlabel(f'Position on chromosome {chrm} (Mb)')

        plt.gcf().set_size_inches(3, 2)
        plt.savefig(f'reports/analysis/chromosome_plot_{s}_{chrm}.pdf', bbox_inches='tight', transparent=True)
        plt.close('all')
