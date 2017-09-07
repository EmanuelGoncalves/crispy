#!/usr/bin/env python
# Copyright (C) 2017 Emanuel Goncalves

import numpy as np
import pandas as pd
from datetime import datetime as dt
from crispy.data_matrix import DataMatrix

# - Imports
# Cell lines samplesheet
css = pd.read_csv('data/gdsc/samplesheet.csv', index_col=0)

# Manifest
manifest = pd.read_csv('data/gdsc/crispr/manifest.txt', sep='\t')
manifest['CGaP_Sample_Name'] = [i.split('_')[0] for i in manifest['CGaP_Sample_Name']]
manifest = manifest.groupby('CGaP_Sample_Name')['Cell_Line_Name'].first()

# Samplesheet
ss = pd.read_csv('data/gdsc/crispr/crispr_raw_counts_files_vFB.csv', sep=',', index_col=0).query("to_use == 'Yes'")
ss = ss.assign(name=list(manifest.loc[[i.split('_')[0] for i in ss.index]]))
ss = ss.assign(tissue=list(css.loc[ss['name'], 'GDSC1'].replace(np.nan, 'other')))

# sgRNA library
sgrna_lib = pd.read_csv('data/gdsc/crispr/KY_Library_v1.1_annotated.csv', index_col=0).dropna(subset=['GENES'])

# Controls
controls = set(ss['control'])


# - Calculate fold-changes
fold_changes = []
for c_sample in manifest.index:
    # c_sample = 'HT29'
    print('[%s] Preprocess Sample: %s' % (dt.now().strftime('%Y-%m-%d %H:%M:%S'), c_sample))

    # Read counts
    c_counts = {}
    for f in set(ss[ss['name'] == manifest[c_sample]].index):
        df = pd.read_csv('data/gdsc/crispr/raw_counts/%s' % f, sep='\t', index_col=0).drop(['gene'], axis=1).to_dict()
        c_counts.update(df)

    # Build counts data-frame (discard sgRNAs with problems)
    c_counts = DataMatrix(c_counts).drop('sgPOLR2K_1', errors='ignore')

    # Build contrast matrix
    design = pd.DataFrame({manifest[c_sample]: {c: -1 if c in controls else 1 / (c_counts.shape[1] - 1) for c in c_counts}})

    # Estimate fold-changes
    c_fc = c_counts.add_constant().counts_normalisation().log_transform().zscore().estimate_fold_change(design)

    # Store
    fold_changes.append(c_fc)

fold_changes = pd.concat(fold_changes, axis=1)
print(fold_changes.shape)


# - Export
# sgRNA level
fold_changes.round(5).to_csv('data/gdsc/crispr/crispy_fold_change_sgrna.csv')

# Gene level (mean)
fold_changes.loc[sgrna_lib.index].groupby(sgrna_lib['GENES']).mean().dropna().round(5).to_csv('data/gdsc/crispr/crispy_fold_change_gene.csv')

