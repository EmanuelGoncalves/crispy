#!/usr/bin/env python
# Copyright (C) 2018 Emanuel Goncalves

import pandas as pd
from scipy.stats import iqr
from scripts.drug.assemble_ppi import STRING_PICKLE, BIOGRID_PICKLE

# META DATA
SAMPLESHEET_FILE = 'data/gdsc/samplesheet.csv'

DRUGSHEET_FILE = 'data/gdsc/drug_samplesheet.csv'

# GROWTH RATE
GROWTHRATE_FILE = 'data/gdsc/growth/growth_rate.csv'

# CRISPR
CRISPR_GENES_FILE = 'data/gdsc/crispr/_00_Genes_for_panCancer_assocStudies.txt'

CRISPR_GENE_FC_CORRECTED = 'data/gdsc/crispr/corrected_logFCs.tsv'

# DRUG-RESPONSE
DRUG_RESPONSE_FILE = 'data/gdsc/drug_single/drug_ic50_merged_matrix.csv'


def crispr_genes(file=None, samples_thres=5, type_thres=3):
    file = CRISPR_GENES_FILE if file is None else file

    c_genes = pd.read_csv(file, sep='\t', index_col=0)

    c_genes = c_genes[c_genes.drop('n. vulnerable cell lines', axis=1).sum(1) <= type_thres]

    c_genes = c_genes[c_genes['n. vulnerable cell lines'] >= samples_thres]

    return set(c_genes.index)


def filter_drug_response(df, percentage_measurements=0.85, ic50_samples=5, iqr_thres=1):
    df = df[df.count(1) > df.shape[1] * percentage_measurements]

    df = df.loc[(df < df.mean().mean()).sum(1) >= ic50_samples]

    df = df.loc[[iqr(values, nan_policy='omit') > iqr_thres for idx, values in df.iterrows()]]

    return df


def drug_targets(file=None):
    file = DRUGSHEET_FILE if file is None else file

    d_targets = pd.read_csv(file, index_col=0)['Target Curated'].dropna().to_dict()

    d_targets = {k: {t.strip() for t in d_targets[k].split(';')} for k in d_targets}

    return d_targets
