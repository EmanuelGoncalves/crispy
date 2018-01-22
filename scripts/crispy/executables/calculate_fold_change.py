#!/usr/bin/env python
# Copyright (C) 2018 Emanuel Goncalves

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from crispy import bipal_dbgd
from crispy.data_matrix import DataMatrix


def foldchange_gdsc(
        lib_file='data/gdsc/crispr/KY_Library_v1.1_annotated.csv',
        man_file='data/gdsc/crispr/canapps_manifest.csv',
        counts_file='data/gdsc/crispr/gdsc_crispr_rawcounts.csv',
        cnv_file='data/crispy_gene_copy_number_snp.csv',
        ss_file='data/gdsc/samplesheet.csv'
):
    # - Imports
    # Samplesheet
    ss = pd.read_csv(ss_file, index_col=0)

    # sgRNA library
    sgrna_lib = pd.read_csv(lib_file, index_col=0).dropna(subset=['GENES'])

    # Manifest
    manifest = pd.read_csv(man_file, index_col='header').query('to_use == True')

    # Raw Counts
    counts = DataMatrix(pd.read_csv(counts_file, index_col=0))

    # Copy-number
    cnv = pd.read_csv(cnv_file, index_col=0)

    # - Processing
    # Build counts data-frame (discard sgRNAs with problems)
    qc_failed_sgrnas = [
        'DHRSX_CCDS35195.1_ex1_X:2161152-2161175:+_3-1', 'DHRSX_CCDS35195.1_ex6_Y:2368915-2368938:+_3-3', 'DHRSX_CCDS35195.1_ex4_X:2326777-2326800:+_3-2', 'sgPOLR2K_1'
    ]
    counts = counts.drop(qc_failed_sgrnas, errors='ignore')

    # Add constant
    counts = counts + 1

    # Normalise by library size
    size_factors = counts.estimate_size_factors()
    counts = counts.divide(size_factors)

    # Calculate replicates correlation (pearson)
    corr_coefs = []
    for c in set(manifest['name']):
        df = np.log2(counts[list(manifest.query("name == '%s'" % c).index)]).corr()
        corr_coefs.append(
            df.where(np.triu(np.ones(df.shape), k=1).astype(np.bool))
                .unstack()
                .dropna()
                .reset_index()
                .rename(columns={'level_0': 's1', 'level_1': 's2', 0: 'pearson'})
                .assign(cell_line=c)
        )
    corr_coefs = pd.concat(corr_coefs).sort_values('pearson')

    # Remove plasmid low count sgRNAs
    for c in set(manifest['plasmid']):
        counts.loc[counts[c] < 30, c] = np.nan

    # Calculate log2 fold-changes: log2(sample / plasmid)
    fc = pd.DataFrame({c: np.log2(counts[c] / counts[manifest.loc[c, 'plasmid']]) for c in manifest.index})

    # Remove lowly correlated replicates (< 0.7 pearson coef)
    fc = fc.loc[:, set(corr_coefs.query('pearson > .7')[['s1', 's2']].unstack())]

    # Average replicates
    fc = fc.groupby(manifest.loc[fc.columns, 'name'], axis=1).mean()

    # Average gene level
    fc = fc.groupby(sgrna_lib.reindex(fc.index)['GENES']).mean().dropna()

    # Overlap with copy-number data
    fc = fc.loc[:, fc.columns.isin(set(cnv))]

    # Export
    fc.round(5).to_csv('data/crispr_gdsc_logfc.csv')

    # - Tissue distribution
    plot_df = ss.loc[list(fc), 'Cancer Type'].value_counts(ascending=True).reset_index().rename(columns={'index': 'Cancer Type', 'Cancer Type': 'Counts'})
    plot_df = plot_df.assign(ypos=np.arange(plot_df.shape[0]))

    plt.barh(plot_df['ypos'], plot_df['Counts'], .8, color=bipal_dbgd[0], align='center')

    for x, y in plot_df[['ypos', 'Counts']].values:
        plt.text(y - .5, x, str(y), color='white', ha='right' if y != 1 else 'center', va='center', fontsize=6)

    plt.yticks(plot_df['ypos'])
    plt.yticks(plot_df['ypos'], plot_df['Cancer Type'])

    plt.xlabel('# cell lines')
    plt.title('CRISPR/Cas9 screen\n(%d cell lines)' % plot_df['Counts'].sum())

    # plt.show()
    plt.gcf().set_size_inches(2, 3)
    plt.savefig('reports/crispy/screen_samples.png', bbox_inches='tight', dpi=600)
    plt.close('all')


def foldchange_ccle(
        lib_file='data/ccle/crispr_ceres/sgrnamapping.csv',
        man_file='data/ccle/crispr_ceres/replicatemap.csv',
        counts_file='data/ccle/crispr_ceres/sgrna_raw_readcounts.csv'
):
    # sgRNA library
    sgrna_lib = pd.read_csv(lib_file).dropna(subset=['Gene'])

    # Manifest
    manifest = pd.read_csv(man_file, index_col=0)
    manifest = manifest.assign(plasmid=['pDNA_%d' % i for i in manifest['Batch']]).dropna()

    # Raw Counts
    counts = DataMatrix(pd.read_csv(counts_file, index_col=0))

    # - Process
    # Build counts data-frame (discard sgRNAs with problems)
    qc_failed_sgrnas = list(counts[counts.sum(1) < 30].index)
    counts = counts.drop(qc_failed_sgrnas, errors='ignore')

    # Add constant
    counts = counts + 1

    # Normalise by library size
    size_factors = counts.estimate_size_factors()
    counts = counts.divide(size_factors)

    # Calculate replicates correlation (pearson)
    corr_coefs = []
    for c in set(manifest['CellLine']):
        df = np.log2(counts[list(manifest.query("CellLine == '%s'" % c).index)]).corr()
        corr_coefs.append(
            df.where(np.triu(np.ones(df.shape), k=1).astype(np.bool))
                .unstack()
                .dropna()
                .reset_index()
                .rename(columns={'level_0': 's1', 'level_1': 's2', 0: 'pearson'})
                .assign(cell_line=c)
        )
    corr_coefs = pd.concat(corr_coefs).sort_values('pearson')

    # Remove plasmid low count sgRNAs
    for c in set(manifest['plasmid']):
        counts.loc[counts[c] < 30, c] = np.nan

    # Calculate log2 fold-changes: log2(sample / plasmid)
    fc = pd.DataFrame({c: np.log2(counts[c] / counts[manifest.loc[c, 'plasmid']]) for c in manifest.index})

    # Remove lowly correlated replicates (< 0.7 pearson coef)
    fc = fc.loc[:, set(corr_coefs.query('pearson > .7')[['s1', 's2']].unstack())]

    # Average replicates
    fc = fc.groupby(manifest.loc[fc.columns, 'CellLine'], axis=1).mean()

    # Average gene level
    fc = fc.loc[sgrna_lib['Guide']].groupby(sgrna_lib['Gene'].values).mean().dropna()

    # Export
    fc.round(5).to_csv('data/crispr_ccle_logfc.csv')


if __name__ == '__main__':
    # Fold-change GDSC CRISPR
    foldchange_gdsc()

    # Fold-change CCLE CRISPR
    foldchange_ccle()
