#!/usr/bin/env python
# Copyright (C) 2018 Emanuel Goncalves

import numpy as np
import crispy as cy
import pandas as pd
import seaborn as sns
import depmap18q3
import matplotlib.pyplot as plt


def replicates_correlation():
    sample_map = depmap18q3.get_replicates_map().dropna()
    sample_map = sample_map.replace({'Broad_ID': samplesheet['CCLE_name']})
    sample_map = sample_map.set_index('replicate_ID')

    rep_fold_changes = pd.concat([
        cy.Crispy(
            raw_counts=raw_counts[manifest[s] + plasmids[s]],
            copy_number=copy_number.query(f"sample == '{s}'"),
            library=lib_guides,
            plasmid=plasmids[s]
        ).replicates_foldchanges() for s in manifest
    ], axis=1, sort=False)

    corr = rep_fold_changes.corr()

    corr = corr.where(np.triu(np.ones(corr.shape), 1).astype(np.bool))
    corr = corr.unstack().dropna().reset_index()
    corr = corr.set_axis(['sample_1', 'sample_2', 'corr'], axis=1, inplace=False)

    corr = corr.assign(name_1=sample_map.loc[corr['sample_1'], 'Broad_ID'].values)
    corr = corr.assign(name_2=sample_map.loc[corr['sample_2'], 'Broad_ID'].values)

    corr = corr.assign(replicate=(corr['name_1'] == corr['name_2']).astype(int))

    replicates_corr = corr.query('replicate == 1')['corr'].mean()
    print(f'[INFO] Replicates correlation: {replicates_corr:.2f}')

    return corr


if __name__ == '__main__':
    # - Imports
    # Samplesheet
    samplesheet = depmap18q3.get_samplesheet()

    # Replicates map
    rep_map = depmap18q3.get_replicates_map()

    # CRISPR-Cas9 raw counts
    raw_counts = depmap18q3.get_raw_counts()

    # Copy-number segments
    copy_number = depmap18q3.get_copy_number_segments()
    copy_number = copy_number.assign(sample=samplesheet.loc[copy_number['sample'], 'CCLE_name'].values)

    # GeCKOv2 library
    lib = depmap18q3.get_crispr_library()

    lib_guides = lib.groupby(['chr', 'start', 'end', 'sgrna'])['gene'].agg(lambda v: ';'.join(set(v))).rename('gene').reset_index()
    lib_guides = lib_guides[[';' not in i for i in lib_guides['gene']]]

    # Gene-sets
    essential, nessential = cy.Utils.get_essential_genes(), cy.Utils.get_non_essential_genes()

    # - Sample manifest
    manifest = rep_map.dropna().groupby('Broad_ID')['replicate_ID']\
        .agg(lambda v: list(set(v)))\
        .rename(samplesheet['CCLE_name'])\
        .to_dict()

    plasmids = rep_map.groupby('Broad_ID')['pDNA_batch'].first().rename(samplesheet['CCLE_name'])
    plasmids = {
        k: list(rep_map[rep_map['Broad_ID'].isna()].query(f'pDNA_batch == {v}')['replicate_ID']) for k, v in plasmids.iteritems()
    }

    # - Replicates correlation threshold
    corr_df = replicates_correlation()
    corr_thres = corr_df.query('replicate == 0')['corr'].mean()
    corr_df.sort_values('corr').to_csv(f'{depmap18q3.DIR}/replicates_foldchange_correlation.csv', index=False)

    sns.boxplot(
        'replicate', 'corr', data=corr_df, notch=True, saturation=1, showcaps=False, color=cy.QCplot.PAL_DBGD[2],
        whiskerprops=cy.QCplot.WHISKERPROPS, boxprops=cy.QCplot.BOXPROPS, medianprops=cy.QCplot.MEDIANPROPS, flierprops=cy.QCplot.FLIERPROPS
    )

    plt.xticks([0, 1], ['No', 'Yes'])

    plt.title('Broad DepMap 18Q3')
    plt.xlabel('Replicate')
    plt.ylabel('Pearson correlation')

    plt.gcf().set_size_inches(1, 2)
    plt.savefig('reports/depmap_18q3/replicates_correlation.png', bbox_inches='tight', dpi=600)
    plt.close('all')
    print(f'[INFO] Correlation fold-change thredhold = {corr_thres:.2f}')

    # - Correction
    gene_original, gene_corrected = {}, {}
    for sample in manifest:
        print(f'[INFO] Sample: {sample}')

        s_crispy = cy.Crispy(
            raw_counts=raw_counts[manifest[sample] + plasmids[sample]],
            copy_number=copy_number.query(f"sample == '{sample}'"),
            library=lib_guides,
            plasmid=plasmids[sample],
        )

        bed_df = s_crispy.correct(qc_replicates_thres=None)

        if bed_df is not None:
            bed_df.to_csv(f'{depmap18q3.DIR}/bed/{sample}.crispy.bed', index=False, sep='\t')

            gene_original[sample] = s_crispy.gene_fold_changes(bed_df)
            gene_corrected[sample] = s_crispy.gene_corrected_fold_changes(bed_df)

    # - Build gene fold-changes matrices
    gene_original = pd.DataFrame(gene_original).dropna()
    gene_original.columns.name = 'original'
    gene_original.round(5).to_csv(f'{depmap18q3.DIR}/gene_fold_changes.csv')

    gene_corrected = pd.DataFrame(gene_corrected).dropna()
    gene_corrected.columns.name = 'corrected'
    gene_corrected.round(5).to_csv(f'{depmap18q3.DIR}/gene_fold_changes_corrected.csv')

    # - Essential and non-essential AUPRs
    aucs = []
    for df in [gene_original, gene_corrected]:
        for gset in [essential, nessential]:

            ax, auc_stats = cy.QCplot.plot_cumsum_auc(df, gset, legend=False)

            mean_auc = pd.Series(auc_stats['auc']).mean()

            plt.title(f'CRISPR-Cas9 {df.columns.name} fold-changes\nmean AURC={mean_auc:.2f}')

            plt.gcf().set_size_inches(3, 3)
            plt.savefig(f'reports/depmap_18q3/qc_aucs_{df.columns.name}_{gset.name}.png', bbox_inches='tight', dpi=600)
            plt.close('all')

            aucs.append(pd.DataFrame(dict(aurc=auc_stats['auc'], type=df.columns.name, geneset=gset.name)))

    aucs = pd.concat(aucs).reset_index().rename(columns={'index': 'sample'})
    aucs.sort_values('aurc', ascending=False).to_csv(f'{depmap18q3.DIR}/qc_aucs.csv', index=False)

    # - Scatter comparison
    for gset in [essential, nessential]:
        plot_df = pd.pivot_table(
            aucs.query("geneset == '{}'".format(gset.name)),
            index='sample', columns='type', values='aurc'
        )

        ax = cy.QCplot.aucs_scatter('original', 'corrected', plot_df)

        ax.set_title('AURCs {} genes'.format(gset.name))

        plt.gcf().set_size_inches(3, 3)
        plt.savefig(f'reports/depmap_18q3/qc_aucs_{gset.name}.png', bbox_inches='tight', dpi=600)
        plt.close('all')
