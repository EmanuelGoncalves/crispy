#!/usr/bin/env python
# Copyright (C) 2018 Emanuel Goncalves

import numpy as np
import crispy as cy
import pandas as pd
import seaborn as sns
import geckov2 as gecko
import matplotlib.pyplot as plt


def replicates_correlation():
    rep_fold_changes = pd.concat([
        cy.Crispy(
            raw_counts=raw_counts[manifest[sample] + [gecko.PLASMID]],
            copy_number=copy_number.query(f"Sample == '{sample}'"),
            library=lib_guides,
            plasmid=gecko.PLASMID
        ).replicates_foldchanges() for sample in manifest
    ], axis=1, sort=False)

    corr = rep_fold_changes.corr()

    corr = corr.where(np.triu(np.ones(corr.shape), 1).astype(np.bool))
    corr = corr.unstack().dropna().reset_index()
    corr = corr.set_axis(['sample_1', 'sample_2', 'corr'], axis=1, inplace=False)
    corr['replicate'] = [int(s1.split(' Rep')[0] == s2.split(' Rep')[0]) for s1, s2 in corr[['sample_1', 'sample_2']].values]

    replicates_corr = corr.query('replicate == 1')['corr'].mean()
    print(f'[INFO] Replicates correlation: {replicates_corr:.2f}')

    return corr


if __name__ == '__main__':
    # - Imports
    # Samplesheet
    samplesheet = gecko.get_samplesheet()

    # Data-sets
    raw_counts = gecko.get_raw_counts()
    copy_number = gecko.get_copy_number_segments()

    # GeCKOv2 library
    lib = gecko.get_crispr_library()

    lib_guides = lib.groupby(['chr', 'start', 'end', 'sgrna'])['gene'].agg(lambda v: ';'.join(set(v))).rename('gene').reset_index()
    lib_guides = lib_guides[[';' not in i for i in lib_guides['gene']]]

    # Gene-sets
    essential, nessential = cy.Utils.get_essential_genes(), cy.Utils.get_non_essential_genes()

    # - Sample manifest
    manifest = {
        v: [c for c in raw_counts if c.split(' Rep')[0] == i] for i, v in samplesheet['name'].iteritems()
    }

    # - Replicates correlation threshold
    corr_df = replicates_correlation()
    corr_thres = corr_df.query('replicate == 0')['corr'].mean()
    corr_df.sort_values('corr').to_csv(f'{gecko.DIR}/replicates_foldchange_correlation.csv', index=False)

    sns.boxplot(
        'replicate', 'corr', data=corr_df, notch=True, saturation=1, showcaps=False, color=cy.QCplot.PAL_DBGD[2],
        whiskerprops=cy.QCplot.WHISKERPROPS, boxprops=cy.QCplot.BOXPROPS, medianprops=cy.QCplot.MEDIANPROPS, flierprops=cy.QCplot.FLIERPROPS
    )

    plt.xticks([0, 1], ['No', 'Yes'])

    plt.title('GeCKOv2')
    plt.xlabel('Replicate')
    plt.ylabel('Pearson correlation')

    plt.gcf().set_size_inches(1, 2)
    plt.savefig('reports/geckov2/replicates_correlation.png', bbox_inches='tight', dpi=600)
    plt.close('all')
    print(f'[INFO] Correlation fold-change thredhold = {corr_thres:.2f}')

    # - Correction
    gene_original, gene_corrected = {}, {}
    for s in manifest:
        print(f'[INFO] Sample: {s}')

        s_crispy = cy.Crispy(
            raw_counts=raw_counts[manifest[s] + [gecko.PLASMID]],
            copy_number=copy_number.query(f"Sample == '{s}'"),
            library=lib_guides,
            plasmid=gecko.PLASMID,
        )

        bed_df = s_crispy.correct(qc_replicates_thres=None)

        if bed_df is not None:
            bed_df.to_csv(f'{gecko.DIR}/bed/{s}.crispy.bed', index=False, sep='\t')

            gene_original[s] = s_crispy.gene_fold_changes(bed_df)
            gene_corrected[s] = s_crispy.gene_corrected_fold_changes(bed_df)

    # - Build gene fold-changes matrices
    gene_original = pd.DataFrame(gene_original).dropna()
    gene_original.columns.name = 'original'
    gene_original.round(5).to_csv(f'{gecko.DIR}/gene_fold_changes.csv')

    gene_corrected = pd.DataFrame(gene_corrected).dropna()
    gene_corrected.columns.name = 'corrected'
    gene_corrected.round(5).to_csv(f'{gecko.DIR}/gene_fold_changes_corrected.csv')

    # - Essential and non-essential AUPRs
    aucs = []
    for df in [gene_original, gene_corrected]:
        for gset in [essential, nessential]:

            ax, auc_stats = cy.QCplot.plot_cumsum_auc(df, gset, legend=False)

            mean_auc = pd.Series(auc_stats['auc']).mean()

            plt.title(f'CRISPR-Cas9 {df.columns.name} fold-changes\nmean AURC={mean_auc:.2f}')

            plt.gcf().set_size_inches(3, 3)
            plt.savefig(f'reports/geckov2/qc_aucs_{df.columns.name}_{gset.name}.png', bbox_inches='tight', dpi=600)
            plt.close('all')

            aucs.append(pd.DataFrame(dict(aurc=auc_stats['auc'], type=df.columns.name, geneset=gset.name)))

    aucs = pd.concat(aucs).reset_index().rename(columns={'index': 'sample'})
    aucs.sort_values('aurc', ascending=False).to_csv(f'{gecko.DIR}/qc_aucs.csv', index=False)

    # - Scatter comparison
    for gset in [essential, nessential]:
        plot_df = pd.pivot_table(
            aucs.query("geneset == '{}'".format(gset.name)),
            index='sample', columns='type', values='aurc'
        )

        ax = cy.QCplot.aucs_scatter('original', 'corrected', plot_df)

        ax.set_title('AURCs {} genes'.format(gset.name))

        plt.gcf().set_size_inches(3, 3)
        plt.savefig(f'reports/geckov2/qc_aucs_{gset.name}.png', bbox_inches='tight', dpi=600)
        plt.close('all')
