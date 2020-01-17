#!/usr/bin/env python
# Copyright (C) 2018 Emanuel Goncalves

import numpy as np
import pandas as pd
import crispy as cy
import seaborn as sns
from pybedtools import BedTool
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind
from analysis.DataImporter import DataImporter, DataProcessor


class DataImporterWGS(DataImporter):
    def __init__(self, copynumber_file=None, subset_samples=None):
        super().__init__(copynumber_file=copynumber_file, subset_samples=subset_samples)

    def import_brass_bedpes(self, fdir=None, bkdist=None, splitreads=True):
        fdir = f'{self.DIR}/wgs/' if fdir is None else fdir

        brass = {
            s:
                cy.Utils.import_brass_bedpe(
                    bedpe_file=f'{fdir}/{s}.brass.annot.bedpe',
                    bkdist=bkdist,
                    splitreads=splitreads
                ) for s in self.samples
        }

        return brass

    def get_copy_number(self, as_dict=False):
        wgs_cnv = super().get_copy_number()

        if as_dict:
            wgs_cnv = {k: df for k, df in data.copy_number.groupby('sample')}

        return wgs_cnv


def svcount_barplot(dfs):
    plot_df = pd.concat([dfs[k] for k in dfs])

    plot_df = plot_df.query("assembly_score != '_'")['svclass'].value_counts().to_frame().sort_values('svclass')
    plot_df = plot_df.assign(ypos=np.arange(plot_df.shape[0]))
    plot_df.index = [i.capitalize() for i in plot_df.index]

    plt.barh(plot_df['ypos'], plot_df['svclass'], .8, color=cy.QCplot.PAL_DBGD[0], align='center')

    plt.xticks(np.arange(0, 601, 200))
    plt.yticks(plot_df['ypos'])
    plt.yticks(plot_df['ypos'], plot_df.index)

    plt.xlabel('Count')
    plt.title('Cancer cell lines\nstructural rearrangements')

    plt.grid(color=cy.QCplot.PAL_DBGD[0], ls='-', lw=.1, axis='x')


def sv_cn_aligned(brass, ascat, offset=1e5, svtypes=None):
    svtypes = ['tandem-duplication', 'deletion'] if svtypes is None else svtypes

    sv_aligned = []

    for sv_chr, sv_start, sv_end, sv in brass[['chr1', 'start1', 'end2', 'svclass']].values:
        for cn_chr, cn_start, cn_end, cn in ascat.values:
            is_in_svtypes = sv in svtypes

            is_same_chr = cn_chr == sv_chr
            is_start_close = cn_start - offset < sv_start < cn_start + offset
            is_end_close = cn_end - offset < sv_end < cn_end + offset

            if is_in_svtypes and is_same_chr and is_start_close and is_end_close:
                alignment = dict(
                    chr=cn_chr,
                    start_cn=cn_start, end_cn=cn_end, cn=cn,
                    start_sv=sv_start, end_sv=sv_end, sv=sv
                )

                sv_aligned.append(alignment)

    sv_aligned = pd.DataFrame(sv_aligned)
    return sv_aligned


def matching_svs(brass_bedpes, ascat_beds, offsets=[1e3, 1e4, 1e5]):
    sv_df, ratio_df = [], []
    for dist_offset in offsets:
        for c in brass_bedpes:
            svs = sv_cn_aligned(brass_bedpes[c], ascat_beds[c].drop(['sample'], axis=1), offset=dist_offset)

            if svs.shape[0] != 0:
                names = ['chr', 'start_sv', 'end_sv', 'start_cn', 'end_cn', 'cn', 'sv']

                svs = svs[svs['start_sv'] < svs['end_sv']]

                svs_bed = BedTool(svs[names].to_string(index=False, header=False), from_string=True).sort()

                svs_sgrnas = svs_bed.map(crispr_beds[c], c='4', o='mean,count') \
                    .to_dataframe(names=names + ['fc_mean', 'fc_count'])

                ratios_sgrnas = svs_bed.map(crispr_beds[c], c='5', o='mean,count') \
                    .to_dataframe(names=names + ['ratio_mean', 'ratio_count'])

                sv_align = pd.concat([
                    svs_sgrnas.set_index(names),
                    ratios_sgrnas.set_index(names)
                ], axis=1)

                sv_df.append(
                    sv_align\
                        .reset_index()\
                        .assign(sample=c)\
                        .assign(offset=dist_offset)
                )

    sv_df = pd.concat(sv_df).query("fc_mean != '.'").reset_index(drop=True)

    sv_df['fc_mean'] = sv_df['fc_mean'].astype(float).values
    sv_df['fc_count'] = sv_df['fc_count'].astype(int).values

    sv_df['ratio_mean'] = sv_df['ratio_mean'].astype(float).values
    sv_df['ratio_count'] = sv_df['ratio_count'].astype(int).values

    return sv_df


if __name__ == '__main__':
    # Define data-sets
    subset_samples = ['HCC1937', 'HCC1143', 'HCC1954', 'HCC1395']

    data = DataImporterWGS(subset_samples=subset_samples)
    data_process = DataProcessor(data=data, bed_filename_prefix='.wgs')

    # BRASS bedpe
    brass = data.import_brass_bedpes(splitreads=True)

    # Copy-number segments
    picnic = data.get_copy_number(as_dict=True)

    # Correction
    fc_original, fc_corrected = data_process.copynumber_correction()

    fc_original.round(5).to_csv('data/analysis/gene_fc_original_wgs.csv')
    fc_corrected.round(5).to_csv('data/analysis/gene_fc_crispy_corrected_wgs.csv')

    # Crispy beds
    beds = data.get_crispy_beds(wgs=True)

    # CRISPR beds
    crispr_beds = {}
    for c in beds:
        crispr_df = beds[c]

        ess_metric = crispr_df[crispr_df['gene'].isin(cy.Utils.get_essential_genes())]['fold_change'].median()
        ness_metric = crispr_df[crispr_df['gene'].isin(cy.Utils.get_non_essential_genes())]['fold_change'].median()
        crispr_df['scaled'] = crispr_df['fold_change'].subtract(ness_metric).divide(ness_metric - ess_metric)

        crispr_beds[c] = BedTool(
            crispr_df[['chr', 'start', 'end', 'scaled', 'ratio']].to_string(index=False, header=False),
            from_string=True
        ).sort()

    # Matching SVs with copy-number segments
    sv_df = matching_svs(brass, picnic)

    # - Plotting
    # Count
    svcount_barplot(brass)
    plt.gcf().set_size_inches(2, 1)
    plt.savefig('reports/analysis/brass_svs_counts_barplot.pdf', bbox_inches='tight', transparent=True)
    plt.close('all')

    # Overlaping SVs
    for offset, df in sv_df.groupby('offset'):
        plot_df = df.copy()

        for y_var in ['fc_mean', 'ratio_mean']:
            t_stat, pval = ttest_ind(
                plot_df.query("sv == 'deletion'")[y_var],
                plot_df.query("sv == 'tandem-duplication'")[y_var],
                equal_var=False
            )

            print(f'\nOff-set: {offset}')
            print(f'Deletion median {y_var}: ', plot_df.query("sv == 'deletion'")[y_var].mean())
            print(f'Tandem-duplication median {y_var}: ', plot_df.query("sv == 'tandem-duplication'")[y_var].mean())

            order = ['deletion', 'tandem-duplication']

            ax = cy.QCplot.bias_boxplot(plot_df, 'sv', y_var, notch=False, add_n=True, tick_base=.2)

            ax.axhline(0, ls='-', color='black', lw=.1, zorder=0, alpha=.5)

            for i in ax.get_xticklabels():
                i.set_rotation(30)
                i.set_horizontalalignment("right")

            ax.set_xlabel('')
            ax.set_ylabel('CRISPR-Cas9 fold-change' if y_var == 'fc_mean' else 'Copy-number ratio')

            ax.set_title(f"Welch's t-test\np-value={pval:.1e}")

            plt.gcf().set_size_inches(1, 2)
            plt.savefig(f'reports/analysis/svs_{y_var}_boxplot_{offset:.1e}.pdf', bbox_inches='tight', transparent=True)
            plt.close('all')

    # SVs examples
    svs = [
        ('HCC1937', 'chr2', (120e6, 210e6), -2.5, 14),
        ('HCC1143', 'chr12', (45e6, 85e6), -2., 14),
        ('HCC1395', 'chr3', (100e6, 200e6), -2., 8),
        ('HCC1954', 'chr8', (60e6, 150e6), -2., 12),
        ('HCC1954', 'chr5', None, -2., 20),
        ('HCC1395', 'chr15', None, -2., 8),
    ]

    for s, chrm, xlim, ax3_ylim, ax2_ylim in svs:
        crispy_bed, brass_bedpe, ascat_bed = beds[s], brass[s], picnic[s].drop(['sample'], axis=1)

        ax1, ax2, ax3 = cy.QCplot.plot_rearrangements(
            brass_bedpe, ascat_bed, crispy_bed, chrm, show_legend=False, xlim=xlim
        )

        if ax2_ylim is not None:
            ax2.set_ylim(0, ax2_ylim)

        y_lim = ax3.get_ylim()
        ax3.set_ylim(ax3_ylim, y_lim[1])

        plt.suptitle(f'{s}', y=1.05)

        plt.gcf().set_size_inches(3, 2)
        plt.savefig(f'reports/analysis/svs_{s}_{chrm}.pdf', bbox_inches='tight', transparent=True)
        plt.close('all')

    # SVs all
    chrms = [
        'chr1', 'chr2', 'chr3', 'chr4', 'chr5', 'chr6', 'chr7', 'chr8', 'chr9', 'chr10',
        'chr11', 'chr12', 'chr13', 'chr14', 'chr15', 'chr16', 'chr17', 'chr18', 'chr19',
        'chr20', 'chr21', 'chr22'
    ]

    for s in data.samples:
        for chrm in chrms:
            crispy_bed, brass_bedpe, ascat_bed = beds[s], brass[s], picnic[s].drop(['sample'], axis=1)

            ax1, ax2, ax3 = cy.QCplot.plot_rearrangements(
                brass_bedpe, ascat_bed, crispy_bed, chrm, show_legend=False
            )

            if ax1 is not None:
                y_lim = ax3.get_ylim()
                ax3.set_ylim(-2.5, y_lim[1])

                plt.suptitle(f'{s}', y=1.05)

                plt.gcf().set_size_inches(3, 2)
                plt.savefig(f'reports/analysis/svs/svs_{s}_{chrm}.pdf', bbox_inches='tight', transparent=True)
                plt.close('all')
