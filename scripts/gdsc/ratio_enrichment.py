#!/usr/bin/env python
# Copyright (C) 2018 Emanuel Goncalves

import gdsc
import crispy as cy
import pandas as pd
import matplotlib.pyplot as plt


if __name__ == '__main__':
    # - Cancer gene list
    cgenes = pd.read_csv('data/cancer_genes.txt', sep='\t')
    cgenes = cgenes[cgenes['role'].isin(['Loss of function', 'Activating'])]
    cgenes = cgenes.groupby('role')['SYM'].agg(set).to_dict()

    # - Crispy bed files
    beds = gdsc.import_crispy_beds()

    # - Aggregate by gene
    ratios = pd.DataFrame({s: beds[s].groupby('gene')['ratio'].median() for s in beds})

    # - Enrichment of ratios
    dataset = cy.Utils.gkn(ratios.dropna().mean(1)).to_dict()

    for gset in cgenes:
        print(f'[INFO] {gset}')

        e_score, p_value, hits, running_hit = cy.SSGSEA.gsea(dataset, cgenes[gset], 10000)

        ax = cy.SSGSEA.plot_gsea(hits, running_hit)

        ax.set_ylabel('ssGSEA\nenrichment score')
        ax.set_title(f'{gset}\nscore={e_score:.2f}, p-value={p_value:.1e}')

        plt.gcf().set_size_inches(3, 2)
        plt.savefig(f'reports/gsea_ratio_{gset}.pdf', bbox_inches='tight', transparent=True)
        plt.close('all')

