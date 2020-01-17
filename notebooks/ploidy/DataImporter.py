#!/usr/bin/env python
# Copyright (C) 2018 Emanuel Goncalves

import os
import numpy as np
import pandas as pd
import crispy as cy
import seaborn as sns
import matplotlib.pyplot as plt


class DataImporter:
    def __init__(
            self,
            dir_path=None,
            samplesheet_file=None,
            sample_replicates_file=None,
            raw_counts_file=None,
            ceres_scores_file=None,
            copynumber_file=None,
            sgrna_map_file=None,
            rnaseq_nexp_file=None,
            subset_samples=None
    ):
        self.DIR = 'data/analysis/datasets/' if dir_path is None else dir_path

        self.SAMPLE_SHEET = 'samplesheet.csv' if samplesheet_file is None else samplesheet_file
        self.SAMPLE_REPLICATES = 'replicate_map.csv' if sample_replicates_file is None else sample_replicates_file

        self.RAW_COUNTS = 'raw_readcounts.csv' if raw_counts_file is None else raw_counts_file
        self.CERES_SCORES = 'gene_effect.csv' if ceres_scores_file is None else ceres_scores_file

        self.COPY_NUMBER = 'segmentation_data_250_lines_picnic.csv' if copynumber_file is None else copynumber_file

        self.SGRNA_MAP = 'guide_gene_map.csv' if sgrna_map_file is None else sgrna_map_file

        self.RNASEQ_NEXP = 'rnaseq_rpkm_lower_1.csv' if rnaseq_nexp_file is None else rnaseq_nexp_file

        self.replicates_map = self.get_sample_replicates()
        self.samplesheet = self.get_sample_sheet()

        self.raw_counts = self.get_raw_counts()
        self.copy_number = self.get_copy_number()

        self.samples = list(set.intersection(set(self.samplesheet.index), set(self.copy_number['sample'])))

        if subset_samples is not None:
            self.samples = list(set(self.samples).intersection(subset_samples))
            assert len(self.samples) > 0, 'No samples overlaping with subset'

        print(f'[INFO] Number of cell lines: {len(self.samples)}')

    def get_sample_replicates(self):
        return pd.read_csv(f'{self.DIR}/{self.SAMPLE_REPLICATES}')

    def get_sample_sheet(self):
        return pd.read_csv(f'{self.DIR}/{self.SAMPLE_SHEET}', index_col=0)

    def get_sgrna_map(self, n_alignments=1):
        lib = pd.read_csv(f'{self.DIR}/{self.SGRNA_MAP}')

        lib['gene'] = lib['gene'].apply(lambda v: v.split(' ')[0])
        lib['chr'] = lib['genome_alignment'].apply(lambda v: v.split('_')[0])

        lib['start'] = lib['genome_alignment'].apply(lambda v: int(v.split('_')[1]))
        lib['end'] = lib['start'] + lib['sgrna'].apply(len)

        if n_alignments is not None:
            lib = lib[lib['n_alignments'] == n_alignments]

        return lib

    def get_raw_counts(self):
        return pd.read_csv(f'{self.DIR}/{self.RAW_COUNTS}', index_col=0)

    def get_copy_number(self):
        return pd.read_csv(f'{self.DIR}/{self.COPY_NUMBER}')

    def get_rnaseq_nexp(self, filter_essential=True):
        nexp = pd.read_csv(f'{self.DIR}/{self.RNASEQ_NEXP}', index_col=0)

        if filter_essential:
            nexp = nexp[~nexp.index.isin(cy.Utils.get_broad_core_essential())]

        return nexp

    def get_ceres_scores(self):
        return pd.read_csv(f'{self.DIR}/{self.CERES_SCORES}', index_col=0)

    def get_unique_sgrna_map(self):
        lib = self.get_sgrna_map(n_alignments=1)
        lib = lib.groupby(['chr', 'start', 'end', 'sgrna'])['gene'].agg(lambda v: ';'.join(set(v))).reset_index()
        lib = lib[[';' not in i for i in lib['gene']]]
        return lib

    def get_samples_replicates_dict(self):
        replicates = self.replicates_map.dropna().groupby('Cell Line Name')['replicate_ID'] \
            .agg(lambda v: list(set(v))) \
            .to_dict()

        batch_plasmid = self.replicates_map[self.replicates_map['Broad_ID'].isna()]
        batch_plasmid = batch_plasmid.groupby('pDNA_batch')['replicate_ID'].agg(set)

        plasmids = self.replicates_map.groupby('Cell Line Name')['pDNA_batch'].first()
        plasmids = {k: list(batch_plasmid[v]) for k, v in plasmids.iteritems()}

        return replicates, plasmids

    def get_crispy_beds(self, wgs=False, file_dir=None, subset=None):
        file_prefix = 'crispy.wgs.bed' if wgs else 'crispy.bed'
        file_dir = f'{self.DIR}/crispy_beds/' if file_dir is None else file_dir

        bed_files = {f.split('.')[0]: f for f in os.listdir(file_dir) if f.endswith(file_prefix)}
        print(bed_files)

        samples = bed_files.keys() if subset is None else set(subset).intersection(bed_files)
        assert len(samples) > 0, 'No samples'

        beds = {s: self.get_crispy_bed(bed_file=bed_files[s], file_dir=file_dir) for s in samples}

        return beds

    def get_crispy_bed(self, bed_file, file_dir=None):
        file_dir = f'{self.DIR}/crispy_beds/' if file_dir is None else file_dir
        return pd.read_csv(f'{file_dir}/{bed_file}', sep='\t')


class DataProcessor(object):

    def __init__(self, data=None, bed_filename_prefix=''):
        self.data = DataImporter() if data is None else data

        self.bed_filename_prefix = bed_filename_prefix

        self.sgrnalib = self.data.get_unique_sgrna_map()

        self.replicates_map, self.plasmids_map = self.data.get_samples_replicates_dict()

    def replicates_correlation(self, level='sgrna'):
        sample_map = self.data.replicates_map.dropna().set_index('replicate_ID')

        if level == 'sgrna':
            rep_fold_changes = pd.concat([
                cy.Crispy(
                    raw_counts=self.data.raw_counts[self.replicates_map[s] + self.plasmids_map[s]],
                    copy_number=self.data.copy_number.query(f"sample == '{s}'"),
                    library=self.sgrnalib,
                    plasmid=self.plasmids_map[s]
                ).replicates_foldchanges() for s in self.data.samples
            ], axis=1, sort=False)

        else:
            rep_fold_changes = pd.concat([
                cy.Crispy(
                    raw_counts=self.data.raw_counts[self.replicates_map[s] + self.plasmids_map[s]],
                    copy_number=self.data.copy_number.query(f"sample == '{s}'"),
                    library=self.sgrnalib,
                    plasmid=self.plasmids_map[s]
                ).gene_fold_changes(qc_replicates_thres=None, average_replicates=False) for s in self.data.samples
            ], axis=1, sort=False)

        corr = rep_fold_changes.corr()

        corr = corr.where(np.triu(np.ones(corr.shape), 1).astype(np.bool))
        corr = corr.unstack().dropna().reset_index()
        corr = corr.set_axis(['sample_1', 'sample_2', 'corr'], axis=1, inplace=False)

        corr = corr.assign(name_1=sample_map.loc[corr['sample_1'], 'Cell Line Name'].values)
        corr = corr.assign(name_2=sample_map.loc[corr['sample_2'], 'Cell Line Name'].values)

        corr = corr.assign(replicate=(corr['name_1'] == corr['name_2']).astype(int))

        replicates_corr = corr.query('replicate == 1')['corr'].mean()
        print(f'[INFO] Replicates correlation: {replicates_corr:.2f}')

        return corr

    def copynumber_correction(self):
        gene_fc_original, gene_fc_corrected = {}, {}

        # Copy-number effects correction per sample
        for s in self.data.samples:
            print(f'[INFO] Sample: {s}')

            s_crispy = cy.Crispy(
                raw_counts=self.data.raw_counts[self.replicates_map[s] + self.plasmids_map[s]],
                copy_number=self.data.copy_number.query(f"sample == '{s}'"),
                library=self.sgrnalib,
                plasmid=self.plasmids_map[s]
            )

            bed_df = s_crispy.correct(qc_replicates_thres=None)
            bed_df.to_csv(f'{self.data.DIR}/crispy_beds/{s}.crispy{self.bed_filename_prefix}.bed', index=False, sep='\t')

            gene_fc_original[s] = s_crispy.gene_fold_changes(bed_df)
            gene_fc_corrected[s] = s_crispy.gene_corrected_fold_changes(bed_df)

        # Build gene fold-changes matrices
        gene_fc_original = pd.DataFrame(gene_fc_original).dropna()
        gene_fc_original.columns.name = 'original'

        gene_fc_corrected = pd.DataFrame(gene_fc_corrected).dropna()
        gene_fc_corrected.columns.name = 'corrected'

        return gene_fc_original, gene_fc_corrected


if __name__ == '__main__':
    # - Imports
    data_process = DataProcessor()

    essential, nessential = cy.Utils.get_essential_genes(), cy.Utils.get_non_essential_genes()

    # - Replicates correlation
    replicates_corr_df = data_process.replicates_correlation()
    replicates_corr_df.sort_values('corr').to_csv(f'data/analysis/replicates_correlation_sgrna_fc.csv', index=False)

    replicates_corr_df = data_process.replicates_correlation(level='gene')
    replicates_corr_df.sort_values('corr').to_csv(f'data/analysis/replicates_correlation_gene_fc.csv', index=False)

    # - Correction
    fc_original, fc_corrected = data_process.copynumber_correction()

    fc_original.round(5).to_csv('data/analysis/gene_fc_original.csv')
    fc_corrected.round(5).to_csv('data/analysis/gene_fc_crispy_corrected.csv')

    # - Ploting
    # - Cancer type coverage
    plot_df = data_process.data.samplesheet['Cancer Type']\
        .value_counts(ascending=True)\
        .reset_index()\
        .rename(columns={'index': 'Cancer type', 'Cancer Type': 'Counts'})\
        .sort_values('Counts', ascending=False)
    plot_df = plot_df.assign(ypos=np.arange(plot_df.shape[0]))

    sns.barplot(plot_df['Counts'], plot_df['Cancer type'], color=cy.QCplot.PAL_DBGD[0])

    for xx, yy in plot_df[['Counts', 'ypos']].values:
        plt.text(xx - .25, yy, str(xx), color='white', ha='right', va='center', fontsize=6)

    plt.grid(color=cy.QCplot.PAL_DBGD[2], ls='-', lw=.3, axis='x')

    plt.xlabel('Number cell lines')
    plt.title('CRISPR-Cas9 screens\n({} cell lines)'.format(plot_df['Counts'].sum()))

    plt.gcf().set_size_inches(2, 6)
    plt.savefig('reports/analysis/cancer_types_histogram.pdf', bbox_inches='tight', transparent=True)
    plt.close('all')

    # - Replicates correlation boxplot
    sns.boxplot(
        'replicate', 'corr', data=replicates_corr_df, notch=True, saturation=1, showcaps=False, color=cy.QCplot.PAL_DBGD[2],
        whiskerprops=cy.QCplot.WHISKERPROPS, boxprops=cy.QCplot.BOXPROPS, medianprops=cy.QCplot.MEDIANPROPS, flierprops=cy.QCplot.FLIERPROPS
    )

    plt.xticks([0, 1], ['No', 'Yes'])

    plt.title('CRISPR-Cas9\nsgRNA fold-changes')
    plt.xlabel('Replicate')
    plt.ylabel('Pearson correlation')

    plt.gcf().set_size_inches(1, 2)
    plt.savefig('reports/analysis/replicates_correlation_boxplot.pdf', bbox_inches='tight', transparent=True)
    plt.close('all')

    # - Essential and non-essential AUPRs
    aucs = []
    for df in [fc_original, fc_corrected]:
        for gset in [essential, nessential]:

            ax, auc_stats = cy.QCplot.plot_cumsum_auc(df, gset, legend=False)

            mean_auc = pd.Series(auc_stats['auc']).mean()

            plt.title(f'CRISPR-Cas9 {df.columns.name} fold-changes\nmean AURC={mean_auc:.2f}')

            plt.gcf().set_size_inches(3, 3)
            plt.savefig(f'reports/analysis/qc_aucs_{df.columns.name}_{gset.name}.pdf', bbox_inches='tight', transparent=True)
            plt.close('all')

            aucs.append(pd.DataFrame(dict(aurc=auc_stats['auc'], type=df.columns.name, geneset=gset.name)))

    aucs = pd.concat(aucs).reset_index().rename(columns={'index': 'sample'})
    aucs.sort_values('aurc', ascending=False).to_csv(f'data/analysis/gene_ess_ness_aucs.csv', index=False)

    # - Essential and non-essential scatter comparison
    for gset in [essential, nessential]:
        plot_df = pd.pivot_table(
            aucs.query("geneset == '{}'".format(gset.name)),
            index='sample', columns='type', values='aurc'
        )

        ax = cy.QCplot.aucs_scatter('original', 'corrected', plot_df)

        ax.set_title('AURCs {} genes'.format(gset.name))

        plt.gcf().set_size_inches(3, 3)
        plt.savefig(f'reports/analysis/qc_aucs_{gset.name}.pdf', bbox_inches='tight', transparent=True)
        plt.close('all')
