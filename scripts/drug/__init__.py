#!/usr/bin/env python
# Copyright (C) 2018 Emanuel Goncalves

import pandas as pd
from scipy.stats import iqr
from scripts.drug.assemble_ppi import STRING_PICKLE, BIOGRID_PICKLE

# META DATA
SAMPLESHEET_FILE = 'data/gdsc/samplesheet.csv'
DRUGSHEET_FILE = 'data/gdsc/drug_samplesheet.csv'

# GENE LISTS
HART_ESSENTIAL = 'data/gene_sets/curated_BAGEL_essential.csv'
HART_NON_ESSENTIAL = 'data/gene_sets/curated_BAGEL_nonEssential.csv'

# GROWTH RATE
GROWTHRATE_FILE = 'data/gdsc/growth/growth_rate.csv'

# CRISPR
CRISPR_GENES_FILE = 'data/gdsc/crispr/_00_Genes_for_panCancer_assocStudies.txt'

CRISPR_GENE_FC_CORRECTED = 'data/gdsc/crispr/corrected_logFCs.tsv'

# DRUG-RESPONSE
DRUG_RESPONSE_FILE = 'data/gdsc/drug_single/drug_ic50_merged_matrix.csv'

# GENOMIC
MOBEM_FILE = 'data/gdsc/mobems/PANCAN_simple_MOBEM.rdata.annotated.csv'


def crispr_genes(file=None, samples_thres=5, type_thres=3):
    file = CRISPR_GENES_FILE if file is None else file

    c_genes = pd.read_csv(file, sep='\t', index_col=0)

    c_genes = c_genes[c_genes.drop('n. vulnerable cell lines', axis=1).sum(1) <= type_thres]

    c_genes = c_genes[c_genes['n. vulnerable cell lines'] >= samples_thres]

    return set(c_genes.index)


def filter_crispr(df, essential=None, essential_thres=90, value_thres=1.5, value_nevents=5):
    if not isinstance(essential, set):
        essential = pd.read_csv(CRISPR_GENES_FILE, sep='\t', index_col=0)['n. vulnerable cell lines']
        essential = set(essential[essential > essential_thres].index)

    assert len(essential) != 0, 'Essential genes list is empty'

    df = df.drop(essential, errors='ignore', axis=0)

    df = df[(df.abs() >= value_thres).sum(1) >= value_nevents]

    return df


def filter_drug_response(df, percentage_measurements=0.85, ic50_samples=5, iqr_thres=1):
    df = df[df.count(1) > df.shape[1] * percentage_measurements]

    df = df.loc[(df < df.mean().mean()).sum(1) >= ic50_samples]

    df = df.loc[[iqr(values, nan_policy='omit') > iqr_thres for idx, values in df.iterrows()]]

    return df


def filter_mobem(df, n_events=5):
    df = df[df.sum(1) >= n_events]
    return df


def drug_targets(file=None):
    file = DRUGSHEET_FILE if file is None else file

    d_targets = pd.read_csv(file, index_col=0)['Target Curated'].dropna().to_dict()

    d_targets = {k: {t.strip() for t in d_targets[k].split(';')} for k in d_targets}

    return d_targets


def check_in_list(gene_list, test_list=None):
    if test_list is None:
        test_list = {'TP53', 'PTEN', 'EGFR', 'ERBB2', 'IGF1R', 'MDM2', 'MDM4', 'BRAF', 'MAPK1'}

    assert len(test_list.intersection(gene_list)) != len(test_list), 'Genes missing: {}'.format(';'.join(test_list.difference(gene_list)))
