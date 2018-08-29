#!/usr/bin/env python
# Copyright (C) 2018 Emanuel Goncalves

import numpy as np
import crispy as cy
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from crispy.qc_plot import QCplot

DIR = 'data/organoids/isogenic/'

SAMPLESHEET = 'samplesheet.csv'


if __name__ == '__main__':
    # - Imports
    # Manifest
    manifest = pd.read_csv(f'{DIR}/{SAMPLESHEET}', index_col=0)

    # Library
    library = cy.get_crispr_lib()

    # - Calculate fold-changes
    gene_fc = pd.concat([
        cy.Crispy(
            raw_counts=pd.read_csv(f'{DIR}/raw_counts/{f}', sep='\t', index_col=0).drop('gene', axis=1),
            library=library
        ).gene_fold_changes(qc_replicates_thres=None).rename(manifest.loc[f, 'name']) for f in manifest.index
    ], axis=1)

    gene_fc.to_csv(f'{DIR}/crispr_folc_changes_gene.csv')

    # - Plot gene-sets AUPR
    # Gene-sets
    essential = pd.Series(list(cy.get_essential_genes())).rename('essential')
    nessential = pd.Series(list(cy.get_non_essential_genes())).rename('non-essential')

    # AUPRs
    for gset in [essential, nessential]:
        pal = manifest.set_index('name')['colour'].to_dict()

        QCplot.plot_cumsum_auc(gene_fc, gset, palette=pal)

        plt.gcf().set_size_inches(3, 3)
        plt.savefig(f'reports/organoids/isogenic_qc_aucs_{gset.name}.png', bbox_inches='tight', dpi=600)
        plt.close('all')
