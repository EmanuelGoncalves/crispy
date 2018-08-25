#!/usr/bin/env python
# Copyright (C) 2018 Emanuel Goncalves

import os
import pandas as pd

DIR = 'data/depmap_18q3/'

SAMPLESHEET = 'sample_info.csv'
REPLICATES = 'replicate_map.csv'

RAW_COUNTS = 'raw_readcounts.csv'
CERES = 'gene_effect.csv'

COPYNUMBER = 'copy_number.csv'

SGRNA_MAP = 'guide_gene_map.csv'


def import_crispy_beds():
    beds = {}

    for f in os.listdir(f'{DIR}/bed/'):
        if f.endswith('crispy.bed'):
            beds[f.split('.')[0]] = pd.read_csv(f'{DIR}/bed/{f}', sep='\t')

    return beds


def get_ceres():
    samplesheet = get_samplesheet()

    ceres = pd.read_csv(f'{DIR}/{CERES}', index_col=0).T
    ceres = ceres.rename(columns=samplesheet['CCLE_name'])
    ceres.index = [i.split(' ')[0] for i in ceres.index]

    return ceres


def get_replicates_map():
    replicates_map = pd.read_csv(f'{DIR}/{REPLICATES}')
    return replicates_map


def get_samplesheet():
    samplesheet = pd.read_csv(f'{DIR}/{SAMPLESHEET}', index_col=1)
    return samplesheet


def get_raw_counts():
    raw_counts = pd.read_csv(f'{DIR}/{RAW_COUNTS}', index_col=0)
    return raw_counts


def get_copy_number_segments():
    columns = {
        'Chromosome': 'chr', 'Start': 'start', 'End': 'end', 'Segment_Mean': 'copy_number', 'Broad_ID': 'sample'
    }

    copy_number = pd.read_csv(f'{DIR}/{COPYNUMBER}').rename(columns=columns)
    copy_number = copy_number[list(columns.values())]
    copy_number = copy_number.assign(chr=copy_number['chr'].apply(lambda v: f'chr{v}'))

    return copy_number


def get_crispr_library(n_alignments=1):
    lib = pd.read_csv(f'{DIR}/{SGRNA_MAP}')

    lib['gene'] = lib['gene'].apply(lambda v: v.split(' ')[0])
    lib['chr'] = lib['genome_alignment'].apply(lambda v: v.split('_')[0])

    lib['start'] = lib['genome_alignment'].apply(lambda v: int(v.split('_')[1]))
    lib['end'] = lib['start'] + lib['sgrna'].apply(len)

    if n_alignments is not None:
        lib = lib[lib['n_alignments'] == n_alignments]

    return lib
