#!/usr/bin/env python
# Copyright (C) 2017 Emanuel Goncalves

import numpy as np
import pandas as pd
import scipy.stats as st
from datetime import datetime as dt

# - Imports
# Samplesheet
ss = pd.read_csv('data/gdsc/crispr/crispr_raw_counts_files_vFB.csv', sep=',', index_col=0).query("to_use == 'Yes'")

# Manifest
manifest = pd.read_csv('data/gdsc/crispr/manifest.txt', sep='\t')
manifest['CGaP_Sample_Name'] = [i.split('_')[0] for i in manifest['CGaP_Sample_Name']]
manifest = manifest.groupby('CGaP_Sample_Name')['Cell_Line_Name'].first()

# sgRNA library
sgrna_lib = pd.read_csv('data/gdsc/crispr/KY_Library_v1.1_annotated.csv', index_col=0).dropna(subset=['STARTpos'])


# -
fold_changes = {}
# Assemble counts matrix
for c_sample in manifest.index:
    # c_sample = 'HT29'
    print('[%s] Preprocess Sample: %s' % (dt.now().strftime('%Y-%m-%d %H:%M:%S'), c_sample))

    # Sample files
    c_files = set(ss[[i.split('_')[0] == c_sample for i in ss.index]].index)

    # Read counts
    c_counts = {}
    for f in c_files:
        df = pd.read_csv('data/gdsc/crispr/raw_counts/%s' % f, sep='\t', index_col=0).drop(['gene'], axis=1).to_dict()
        c_counts.update(df)

    c_counts = pd.DataFrame(c_counts).drop('sgPOLR2K_1', errors='ignore')

    # Add dummy count for 0 counts
    c_counts += 1

    # Normalisation coefficients
    c_counts_gmean = pd.Series(st.gmean(c_counts, axis=1), index=c_counts.index)
    c_counts_gmean = c_counts.T.divide(c_counts_gmean).T
    c_counts_gmean = c_counts_gmean.median()

    c_counts = c_counts.divide(c_counts_gmean)

    # Log2 transform
    c_counts = np.log2(c_counts)

    # Calculate fold-changes
    c_control = [i for i in c_counts if i in ['ERS717283.plasmid', 'CRISPR_C6596666.sample']][0]
    c_counts = c_counts.subtract(c_counts[c_control], axis=0).drop(c_control, axis=1)

    # Average fold-changes
    c_counts = c_counts.median(1)

    # Store
    fold_changes[manifest[c_sample]] = c_counts

fold_changes = pd.DataFrame(fold_changes)
print(fold_changes.shape)


# - Export
fold_changes.to_csv('data/gdsc/crispr/crispy_fold_change.csv')
