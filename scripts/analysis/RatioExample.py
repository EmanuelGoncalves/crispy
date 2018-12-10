#!/usr/bin/env python
# Copyright (C) 2018 Emanuel Goncalves

import numpy as np
import crispy as cy
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from natsort import natsorted
from scipy.stats import spearmanr
from analysis.DataImporter import DataImporter


def build_matrix(dataframe, field):
    matrix = pd.pivot_table(dataframe, index='gene', columns='sample', values=field).dropna()
    return matrix


if __name__ == '__main__':
    # - Imports
    data = DataImporter()

    # Crispy bed files
    beds = data.get_crispy_beds()

    # Non-expressed genes
    nexp = data.get_rnaseq_nexp(filter_essential=True)

    # - Add CERES gene scores to bed
    ceres = data.get_ceres_scores()
    beds = {s: beds[s].assign(ceres=ceres.loc[beds[s]['gene'], s].values) for s in beds}

    # - Aggregate by gene
    agg_fun = dict(
        fold_change=np.mean, copy_number=np.mean, ploidy=np.mean, chr_copy=np.mean, ratio=np.mean, chr='first',
        corrected=np.mean, ceres=np.mean
    )

    melted_bed = []
    for s in beds:
        df = beds[s].groupby('gene').agg(agg_fun)
        df = df.assign(scaled=cy.Crispy.scale_crispr(df['fold_change']))
        df = df.assign(corrected_scaled=cy.Crispy.scale_crispr(df['corrected']))
        df = df.assign(ceres_scaled=cy.Crispy.scale_crispr(df['ceres']))
        df = df.assign(nexp=df.index.isin(nexp[nexp[s] == 1].index).astype(int))
        df = df.assign(sample=s)

        melted_bed.append(df)

    melted_bed = pd.concat(melted_bed).reset_index()

    melted_bed = melted_bed.assign(copy_number_bin=melted_bed['copy_number'].apply(cy.Utils.bin_cnv, args=(10,)))
    melted_bed = melted_bed.assign(ratio_bin=melted_bed['ratio'].apply(cy.Utils.bin_cnv, args=(4,)))
    melted_bed = melted_bed.assign(chr_copy_bin=melted_bed['chr_copy'].apply(cy.Utils.bin_cnv, args=(6,)))

    # -
    fc_original = build_matrix(melted_bed, 'scaled')
    fc_ceres = build_matrix(melted_bed, 'ceres_scaled')
    fc_crispy = build_matrix(melted_bed, 'corrected_scaled')

    genes = set.intersection(set(fc_original.index), set(fc_ceres.index), set(fc_crispy.index))
    samples = set.intersection(set(fc_original), set(fc_ceres), set(fc_crispy))
    print(f'[INFO] Genes: {len(genes)}, Samples: {len(samples)}')

    corr_df = pd.concat([
        fc_original.loc[genes, samples].corrwith(fc_ceres.loc[genes, samples], axis=1).rename('ceres'),
        fc_original.loc[genes, samples].corrwith(fc_crispy.loc[genes, samples], axis=1).rename('crispy')
    ], axis=1)

    corr_df.loc[cy.Utils.get_essential_genes()].dropna().eval('crispy - ceres').sort_values()

    #
    gene_drive = pd.read_csv('/Users/eg14/Downloads/DriverGeneList.csv')
    gene_drive = gene_drive[gene_drive['Tumor suppressor or oncogene prediction (by 20/20+)'] == 'oncogene']

    gene_cn = pd.concat([
        melted_bed.groupby('gene')['ratio'].max(),
        melted_bed.groupby('gene')['copy_number'].max(),
        melted_bed.groupby('gene')['nexp'].min()
    ], axis=1)

    gene_cn.query('(ratio < 2.5) & (nexp == 1)').sort_values('copy_number')
    gene_cn[gene_cn.index.isin(gene_drive['Gene'])].sort_values('copy_number')

    # -
    ess_bed = melted_bed[melted_bed['gene'].isin(cy.Utils.get_essential_genes())]
    ess_bed.query("(copy_number > 7) & (ratio < 2) & (ceres > -.25)")['gene'].value_counts()

    ess_bed.loc[ess_bed.eval('scaled - ceres_scaled').sort_values().index]\
        .query('(ratio < 2.5) & (scaled > -3)')

    #
    gene = 'NEUROD2'

    plot_df = melted_bed.query(f"gene == '{gene}'")
    plot_df['cancer_type'] = data.samplesheet.loc[plot_df['sample'], 'Cancer Type'].values

    for field in ['corrected_scaled', 'ceres_scaled']:
        ax = plt.gca()

        cn_feature = 'copy_number' if field.startswith('ceres') else 'ratio'

        hue_order = natsorted(set(plot_df[f'{cn_feature}_bin']), reverse=False)
        pal = pd.Series(list(cy.QCplot.get_palette_continuous(len(hue_order))), index=hue_order)

        for cn in hue_order:
            df = plot_df.query(f"{cn_feature}_bin == '{cn}'")
            ax.scatter(df[field], df['scaled'], color=pal[cn], lw=.3, edgecolor='white', label=cn)

        sns.regplot(
            plot_df[field], plot_df['scaled'], scatter=False, lowess=True,
            line_kws=dict(lw=1., color=cy.QCplot.PAL_DBGD[1])
        )

        cor, pval = spearmanr(plot_df[field], plot_df['scaled'])
        annot_text = f'Spearman R={cor:.2g}, p={pval:.1e}'

        ax.text(.95, .05, annot_text, fontsize=5, transform=ax.transAxes, ha='right')

        lgd = plt.legend(
            frameon=False, loc=2, prop={'size': 3.5},
            title='Copy-number ratio' if cn_feature == 'ratio' else 'Copy-number',
        )
        lgd._legend_box.align = 'left'
        lgd.get_title().set_fontsize(4)

        plt.title(gene)
        plt.ylabel('Original fold-change\n(scaled)')
        plt.xlabel('Ceres corrected (scaled)' if field.startswith('ceres') else 'Crispy corrected (scaled)')

        plt.gcf().set_size_inches(2.5, 2.5)
        plt.savefig(
            f"reports/analysis/example_{'ceres' if field.startswith('ceres') else 'crispy'}.pdf",
            bbox_inches='tight', transparent=True
        )
        plt.close('all')
