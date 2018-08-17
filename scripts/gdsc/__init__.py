#!/usr/bin/env python
# Copyright (C) 2018 Emanuel Goncalves

import os
import numpy as np
import pandas as pd


DIR = 'data/gdsc/'

SAMPLESHEET = 'samplesheet.csv'
RAW_COUNTS = 'raw_read_counts.csv'
COPYNUMBER = 'copynumber_PICNIC_segtab.csv'

QC_FAILED_SGRNAS = [
    'DHRSX_CCDS35195.1_ex1_X:2161152-2161175:+_3-1',
    'DHRSX_CCDS35195.1_ex6_Y:2368915-2368938:+_3-3',
    'DHRSX_CCDS35195.1_ex4_X:2326777-2326800:+_3-2',
    'sgPOLR2K_1'
]


def import_crispy_beds():
    beds = {
        f.split('.')[0]:
            pd.read_csv(f'{DIR}/bed/{f}', sep='\t') for f in os.listdir(f'{DIR}/bed/') if f.endswith('crispy.bed')
    }
    return beds


def get_samplesheet():
    return pd.read_csv(f'{DIR}/{SAMPLESHEET}', index_col=0)


def get_raw_counts():
    return pd.read_csv(f'{DIR}/{RAW_COUNTS}', index_col=0)


def get_copy_number_segments():
    return pd.read_csv(f'{DIR}/{COPYNUMBER}')
