#!/usr/bin/env python
# Copyright (C) 2018 Emanuel Goncalves

import gdsc
import numpy as np
import crispy as cy
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


if __name__ == '__main__':
    # - Imports
    # Samplesheet
    samplesheet = gdsc.get_samplesheet()

    # CRISPR-Cas9 raw counts
    raw_counts = gdsc.get_raw_counts()
    raw_counts = raw_counts.dropna().astype(int)
    raw_counts = raw_counts.drop(gdsc.QC_FAILED_SGRNAS, errors='ignore')

    # Copy-number
    copy_number = gdsc.get_copy_number_segments_wgs()
    copy_number = copy_number[~copy_number['chr'].isin(['chrX', 'chrY'])]

    # Yusa library v1.1
    lib = cy.Utils.get_crispr_lib()
    lib = lib[lib['sgrna'].isin(raw_counts.index)]

    # Gene-sets
    essential, nessential = cy.Utils.get_essential_genes(), cy.Utils.get_non_essential_genes()

    # - Cell line replicates and plasmid map
    manifest = samplesheet.reset_index().groupby('name')['sample_id'].agg(list).to_dict()
    plasmid = samplesheet.groupby('name')['plasmid'].agg(lambda v: list(set(v))[0]).to_dict()

    # - Correction
    gene_original, gene_corrected = {}, {}
    for s in gdsc.BRCA_SAMPLES:
        print(f'[INFO] Sample: {s}')

        s_crispy = cy.Crispy(
            raw_counts=raw_counts[manifest[s] + [plasmid[s]]],
            copy_number=copy_number.query(f"sample == '{s}'"),
            library=lib,
            plasmid=plasmid[s]
        )

        bed_df = s_crispy.correct(qc_replicates_thres=None)

        if bed_df is not None:
            bed_df.to_csv(f'{gdsc.DIR}/bed/{s}.crispy.wgs.bed', index=False, sep='\t')

            gene_original[s] = s_crispy.gene_fold_changes(bed_df)
            gene_corrected[s] = s_crispy.gene_corrected_fold_changes(bed_df)

    # - Build gene fold-changes matrices
    gene_original = pd.DataFrame(gene_original).dropna()
    gene_original.columns.name = 'original'
    gene_original.round(5).to_csv(f'{gdsc.DIR}/gene_fold_changes_wgs.csv')

    gene_corrected = pd.DataFrame(gene_corrected).dropna()
    gene_corrected.columns.name = 'corrected'
    gene_corrected.round(5).to_csv(f'{gdsc.DIR}/gene_fold_changes_corrected_wgs.csv')

    # - Essential and non-essential AUPRs
    aucs = []
    for df in [gene_original, gene_corrected]:
        for gset in [essential, nessential]:

            ax, auc_stats = cy.QCplot.plot_cumsum_auc(df, gset, legend=False)

            mean_auc = pd.Series(auc_stats['auc']).mean()

            plt.title(f'CRISPR-Cas9 {df.columns.name} fold-changes\nmean AURC={mean_auc:.2f}')

            plt.gcf().set_size_inches(3, 3)
            plt.savefig(f'reports/gdsc/wgs/qc_aucs_{df.columns.name}_{gset.name}.png', bbox_inches='tight', dpi=600)
            plt.close('all')

            aucs.append(pd.DataFrame(dict(aurc=auc_stats['auc'], type=df.columns.name, geneset=gset.name)))

    aucs = pd.concat(aucs).reset_index().rename(columns={'index': 'sample'})
    aucs.sort_values('aurc', ascending=False).to_csv(f'{gdsc.DIR}/qc_aucs_wgs.csv', index=False)

    # - Scatter comparison
    for gset in [essential, nessential]:
        plot_df = pd.pivot_table(
            aucs.query("geneset == '{}'".format(gset.name)),
            index='sample', columns='type', values='aurc'
        )

        ax = cy.QCplot.aucs_scatter('original', 'corrected', plot_df)

        ax.set_title('AURCs {} genes'.format(gset.name))

        plt.gcf().set_size_inches(3, 3)
        plt.savefig(f'reports/gdsc/wgs/qc_aucs_{gset.name}.png', bbox_inches='tight', dpi=600)
        plt.close('all')
