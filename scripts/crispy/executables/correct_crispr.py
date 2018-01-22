#!/usr/bin/env python
# Copyright (C) 2018 Emanuel Goncalves

import pandas as pd
from crispy.biases_correction import CRISPRCorrection


def correct_cnv(sample, crispr, cnv, lib):
    lib_genes = lib.groupby('GENES')['CHRM'].first()

    # Build data-frame
    df = pd.concat([crispr[sample].rename('fc'), cnv[sample].rename('cnv'), lib_genes.rename('chr')], axis=1).dropna()

    # Correction
    crispy = CRISPRCorrection().rename(sample).fit_by(by=df['chr'], X=df[['cnv']], y=df['fc'])
    crispy = pd.concat([v.to_dataframe() for k, v in crispy.items()])

    return crispy


def correct_cnv_gdsc(lib_file='data/gdsc/crispr/KY_Library_v1.1_annotated.csv', crispr_file='data/crispr_gdsc_logfc.csv', cnv_file='data/crispy_gene_copy_number_snp.csv'):
    # Import sgRNA library
    lib = pd.read_csv(lib_file, index_col=0).dropna(subset=['GENES'])

    # CRISPR fold-changes
    crispr = pd.read_csv(crispr_file, index_col=0)

    # Copy-number
    cnv = pd.read_csv(cnv_file, index_col=0)

    # Crispy
    for sample in set(crispr).intersection(cnv):
        print('GDSC Crispy: %s' % sample)

        # Bias correction
        crispy = correct_cnv(sample, crispr, cnv, lib)

        # Export
        crispy.to_csv('data/crispy/crispr_gdsc_crispy_%s.csv' % sample)


def correct_cnv_ccle(lib_file='data/ccle/crispr_ceres/sgrnamapping.csv', crispr_file='data/crispr_ccle_logfc.csv', cnv_file='data/ccle/crispr_ceres/data_log2CNA.txt'):
    # Import sgRNA library
    sgrna_lib = pd.read_csv(lib_file).dropna(subset=['Gene'])
    sgrna_lib = sgrna_lib.assign(chr=sgrna_lib['Locus'].apply(lambda x: x.split('_')[0]))
    sgrna_lib_genes = sgrna_lib.groupby('Gene')['chr'].first()

    # CRISPR fold-changes
    crispr = pd.read_csv(crispr_file, index_col=0)

    # Copy-number absolute counts
    cnv = pd.read_csv(cnv_file, sep='\t', index_col=0)

    # Crispy
    for sample in set(crispr).intersection(cnv):
        print('CCLE Crispy: %s' % sample)

        # Build data-frame
        df = pd.concat([crispr[sample].rename('fc'), 2 * (2 ** cnv[sample].rename('cnv')), sgrna_lib_genes.rename('chr')], axis=1).dropna()

        # Biases correction
        crispy = CRISPRCorrection().rename(sample).fit_by(by=df['chr'], X=df[['cnv']], y=df['fc'])
        crispy = pd.concat([v.to_dataframe() for k, v in crispy.items()])

        # Store
        crispy.to_csv('data/crispy/crispr_ccle_crispy_%s.csv' % sample)


if __name__ == '__main__':
    # Correct GDSC CRISPR
    correct_cnv_gdsc()

    # # Correct CCLE CRISPR
    # correct_cnv_ccle()
