#!/usr/bin/env python
# Copyright (C) 2017 Emanuel Goncalves

import sys
import numpy as np
import pandas as pd
from crispy.data_correction import DataCorrection


# - Imports
# sgRNA library
sgrna_lib = pd.read_csv('data/gdsc/crispr/KY_Library_v1.1_annotated.csv', index_col=0).dropna(subset=['STARTpos'])

# CRISPR fold-changes
fc_sgrna = pd.read_csv('data/gdsc/crispr/crispy_gdsc_fold_change_sgrna.csv', index_col=0)


# - Biases correction method
sample = 'AU565'
# sample = sys.argv[1:]

df = pd.concat([fc_sgrna[sample].rename('logfc'), sgrna_lib], axis=1).dropna(subset=['logfc', 'GENES'])
df = df.groupby('GENES').agg({
    'CODE': 'first',
    'CHRM': 'first',
    'STARTpos': np.mean,
    'ENDpos': np.mean,
    'logfc': np.mean
}).sort_values('CHRM')

# Correction
df_corrected = DataCorrection(df[['STARTpos']], df[['logfc']]).scale_positions().correct_position(df['CHRM'])


# - Export
df_corrected.round(5).to_csv('data/crispy/crispy_gdsc_fold_change_gene_unsupervised_%s.csv' % sample)
