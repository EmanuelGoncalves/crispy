#!/usr/bin/env python
# Copyright (C) 2018 Emanuel Goncalves

import crispy as cy
import pandas as pd
import matplotlib.pyplot as plt
from analysis.DataImporter import DataImporter

if __name__ == '__main__':
    # - Cancer gene list
    cgenes = pd.read_csv('data/analysis/genesets/cancer_genes.txt', sep='\t')
    cgenes = cgenes[cgenes['role'].isin(['Loss of function', 'Activating'])]
    cgenes = cgenes.groupby('role')['SYM'].agg(set).to_dict()

    # - Crispy bed files
    data = DataImporter()
    beds = data.get_crispy_beds()

    # - Aggregate by gene
    ratios = pd.DataFrame({s: beds[s].groupby('gene')['ratio'].median() for s in beds})

    # - Cancer genes ratios
    def annot_gene(gene):
        if (gene in cgenes['Loss of function']) and (gene in cgenes['Activating']):
            return 'Loss of function/Activating'

        elif gene in cgenes['Loss of function']:
            return 'Loss of function'

        elif gene in cgenes['Activating']:
            return 'Activating'

        else:
            return 'None'

    df = ratios.mean(1).sort_values().reset_index()
    df.columns = ['gene', 'ratio']
    df['MOA'] = df['gene'].apply(annot_gene)
    df = df[df['MOA'] != 'None']

    # - Enrichment of ratios
    dataset = cy.Utils.gkn(ratios.dropna().mean(1)).to_dict()

    for gset in cgenes:
        print(f'[INFO] {gset}')

        e_score, p_value, hits, running_hit = cy.SSGSEA.gsea(dataset, cgenes[gset], 10000)

        ax = cy.GSEAplot.plot_gsea(hits, running_hit)

        ax.set_ylabel('ssGSEA\nenrichment score')
        ax.set_title(f'{gset}\nscore={e_score:.2f}, p-value={p_value:.1e}')

        plt.gcf().set_size_inches(3, 2)
        plt.savefig(f'reports/analysis/gsea_ratio_{gset}.pdf', bbox_inches='tight', transparent=True)
        plt.close('all')

