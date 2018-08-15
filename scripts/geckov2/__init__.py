#!/usr/bin/env python
# Copyright (C) 2018 Emanuel Goncalves

import pandas as pd

DIR = 'data/geckov2/'

SAMPLESHEET = 'Achilles_v3.3.8.samplesheet.txt'
RAW_COUNTS = 'Achilles_v3.3.8_rawreadcounts.gct'

COPYNUMBER = 'Achilles_v3.3.8_ABSOLUTE_CN_segtab.txt'
SGRNA_MAP = 'Achilles_v3.3.8_sgRNA_mappings.txt'

PLASMID = 'pDNA_pXPR003_120K_20140624'


def get_samplesheet():
    return pd.read_csv(f'{DIR}/{SAMPLESHEET}', sep='\t', index_col=1)


def get_raw_counts():
    return pd.read_csv(f'{DIR}/{RAW_COUNTS}', index_col=0, skiprows=2, sep='\t').drop('Description', axis=1)


def get_copy_number_segments():
    columns = dict(Chromosome='chr', Start='start', End='end', CN='copy_number')

    df = pd.read_csv(f'{DIR}/{COPYNUMBER}', sep='\t').rename(columns=columns)
    df['chr'] = df['chr'].apply(lambda v: f'chr{v}')

    return df


def get_crispr_library(drop_non_targeting=True):
    lib = pd.read_csv(f'{DIR}/{SGRNA_MAP}', sep='\t')

    if drop_non_targeting:
        lib = lib.dropna(subset=['Chromosome', 'Pos', 'Strand'])

    lib['sgrna'] = lib['Guide'].apply(lambda v: v.split('_')[0])
    lib['gene'] = lib['Guide'].apply(lambda v: v.split('_')[1])

    lib = lib.rename(columns={'Chromosome': 'chr', 'Pos': 'start'})

    lib['start'] = lib['start'].astype(int)
    lib['end'] = lib['start'] + lib['sgrna'].apply(len) + lib['PAM'].apply(len)

    return lib
