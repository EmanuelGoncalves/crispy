#!/usr/bin/env python
# Copyright (C) 2017 Emanuel Goncalves

import pandas as pd
from crispy.data_matrix import DataMatrix

# Import
counts = pd.read_csv('data/gdsc/crispr/gdsc_crispr_rawcounts.csv', index_col=0)

# Build counts data-frame (discard sgRNAs with problems)
qc_failed_sgrnas = [
    'DHRSX_CCDS35195.1_ex1_X:2161152-2161175:+_3-1',
    'DHRSX_CCDS35195.1_ex6_Y:2368915-2368938:+_3-3',
    'DHRSX_CCDS35195.1_ex4_X:2326777-2326800:+_3-2',
    'sgPOLR2K_1'
]
c_counts = DataMatrix(counts).drop(qc_failed_sgrnas, errors='ignore')

# Filter by minimum required control counts
c_counts = c_counts[c_counts[[c for c in c_counts if c in controls][0]] >= 30]

# Build contrast matrix
design = pd.DataFrame({manifest[c_sample]: {c: -1 if c in controls else 1 / (c_counts.shape[1] - 1) for c in c_counts}})

# Estimate fold-changes
c_fc = c_counts.add_constant().counts_normalisation().log_transform().estimate_fold_change(design)

