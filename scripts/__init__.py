#!/usr/bin/env python
# Copyright (C) 2018 Emanuel Goncalves

import os
import numpy as np
import pandas as pd
from crispy import get_essential_genes, get_non_essential_genes


# META DATA PATHS
MANIFEST = 'data/gdsc/crispr/manifest.csv'
SAMPLESHEET = 'data/gdsc/samplesheet.csv'

# CRISPR LIBRARY
QC_FAILED_SGRNAS = [
    'DHRSX_CCDS35195.1_ex1_X:2161152-2161175:+_3-1',
    'DHRSX_CCDS35195.1_ex6_Y:2368915-2368938:+_3-3',
    'DHRSX_CCDS35195.1_ex4_X:2326777-2326800:+_3-2',
    'sgPOLR2K_1'
]

LOW_COUNT_THRES = 30

# CRISPR PATHS
CRISPR_RAW_COUNTS = 'data/gdsc/crispr/gdsc_crispr_rawcounts.csv'
CRISPR_SGRNA_FC = 'data/crispr_gdsc_sgrna_logfc.csv'
CRISPR_GENE_FC = 'data/crispr_gdsc_logfc.csv'
CRISPR_GENE_CORRECTED_FC = 'data/crispr_gdsc_logfc_corrected.csv'

# WGS ASCAT PATHS
WGS_ASCAT_BED = 'data/gdsc/wgs/ascat_bed/'

# WGS BRASS PATHS
WGS_BRASS_BEDPE = 'data/gdsc/wgs/brass_bedpe/'

# WGS + CRISPR BRCA samples
BRCA_SAMPLES = ['HCC1954', 'HCC1143', 'HCC38', 'HCC1187', 'HCC1937', 'HCC1395']

# SNP6 PICNIC PATHS
SNP_PICNIC_BED = 'data/gdsc/copynumber/snp6_bed/'

# CN PATHS
CN_GENE = 'data/crispy_copy_number_gene_{}.csv'
CN_GENE_RATIO = 'data/crispy_copy_number_gene_ratio_{}.csv'
CN_CHR = 'data/crispy_copy_number_chr_{}.csv'
CN_PLOIDY = 'data/crispy_copy_number_ploidy_{}.csv'

# CRISPY
CRISPY_OUTDIR = 'data/crispy/gdsc/'
CRISPY_WGS_OUTDIR = 'data/crispy/gdsc_brass/'

# NON-EXPRESSED GENES: RNA-SEQ
NON_EXP = 'data/non_expressed_genes.csv'
BIOMART_HUMAN_ID_TABLE = 'data/resources/biomart/biomart_human_id_table.csv'

GDSC_RNASEQ_SAMPLESHEET = 'data/gdsc/gene_expression/merged_sample_annotation.csv'
GDSC_RNASEQ_RPKM = 'data/gdsc/gene_expression/merged_rpkm.csv'

# GENE-SETS
CANCER_GENES = 'data/gene_sets/Census_allThu Dec 21 15_43_09 2017.tsv'

# GTex
GTEX_TPM = 'data/GTEx_Analysis_2016-01-15_v7_RNASeQCv1.1.8_gene_median_tpm.gct'


# - GETTERS AND SETTERS
def get_rnaseq_rpkm():
    rpkm = pd.read_csv(GDSC_RNASEQ_RPKM, index_col=0)
    return rpkm


def get_crispr(dfile, scale=True):
    crispr = pd.read_csv(dfile, index_col=0).dropna()

    if scale:
        crispr = scale_crispr(crispr)

    return crispr


def get_non_exp():
    if not os.path.exists(NON_EXP):
        write_non_expressed_matrix()

    return pd.read_csv(NON_EXP, index_col=0)


def get_copynumber(dtype='snp'):
    return pd.read_csv(CN_GENE.format(dtype), index_col=0)


def get_ploidy(dtype='snp'):
    return pd.read_csv(CN_PLOIDY.format(dtype), index_col=0, names=['sample', 'ploidy'])['ploidy']


def get_chr_copies(dtype='snp'):
    return pd.read_csv(CN_CHR.format(dtype), index_col=0)


def get_samplesheet(index_col=0):
    return pd.read_csv(SAMPLESHEET, index_col=index_col)


def get_rnaseq_samplesheet():
    ss = pd.read_csv(GDSC_RNASEQ_SAMPLESHEET)
    return ss


def get_ensembl_gene_id_table():
    ensg = pd.read_csv(BIOMART_HUMAN_ID_TABLE)
    ensg = ensg.groupby('Gene name')['Gene stable ID'].agg(lambda x: set(x)).to_dict()
    return ensg


# -- METHODS
def scale_crispr(df, ess=None, ness=None, metric=np.median):
    """
    Min/Max scaling of CRISPR-Cas9 log-FC by median (default) Essential and Non-Essential.

    :param df: Float pandas.DataFrame
    :param ess: set(String)
    :param ness: set(String)
    :param metric: np.Median (default)
    :return: Float pandas.DataFrame
    """

    if ess is None:
        ess = get_essential_genes()

    if ness is None:
        ness = get_non_essential_genes()

    assert len(ess.intersection(df.index)) != 0, 'DataFrame has no index overlapping with essential list'
    assert len(ness.intersection(df.index)) != 0, 'DataFrame has no index overlapping with non-essential list'

    ess_metric = metric(df.reindex(ess).dropna(), axis=0)
    ness_metric = metric(df.reindex(ness).dropna(), axis=0)

    df = df.subtract(ness_metric).divide(ness_metric - ess_metric)

    return df


def write_non_expressed_matrix(rpkm_thres=1):
    # Gene map
    ensg = get_ensembl_gene_id_table()

    # Samplesheet
    ss = get_rnaseq_samplesheet()
    ss = pd.Series({k: v.split('.')[0] for k, v in ss.groupby('expression_matrix_name')['sample_name'].first().to_dict().items()})

    # RNA-seq RPKMs
    rpkm = get_rnaseq_rpkm()

    # - Low-expressed genes: take the highest abundance across the different transcripts
    nexp = pd.DataFrame(
        {g: (rpkm.reindex(ensg[g]) < rpkm_thres).astype(int).min() for g in ensg if len(ensg[g].intersection(rpkm.index)) > 0}
    ).T

    # Merged cell lines replicates: take the highest abundance across the different replicates
    nexp = nexp.groupby(ss.loc[nexp.columns], axis=1).min()

    # Export
    nexp.to_csv(NON_EXP)
