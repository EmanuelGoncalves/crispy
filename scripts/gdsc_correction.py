#!/usr/bin/env python
# Copyright (C) 2017 Emanuel Goncalves

import sys
import numpy as np
import pandas as pd
from crispy.data_correction import DataCorrectionPosition


# - Imports
# sgRNA library
sgrna_lib = pd.read_csv('data/gdsc/crispr/KY_Library_v1.1_annotated.csv', index_col=0).dropna(subset=['STARTpos'])

# CRISPR fold-changes
fc_sgrna = pd.read_csv('data/gdsc/crispr/crispy_gdsc_fold_change_sgrna.csv', index_col=0)

# Copy-number absolute counts
cnv = pd.read_csv('data/gdsc/copynumber/Gene_level_CN.txt', sep='\t', index_col=0)
cnv = cnv.loc[:, cnv.columns.isin(list(fc_sgrna))]
cnv_abs = cnv.applymap(lambda v: int(v.split(',')[0]))


# - Biases correction method
# sample = 'ES-2'
sample = sys.argv[1]

df = pd.concat([fc_sgrna[sample].rename('logfc'), sgrna_lib], axis=1).dropna(subset=['logfc', 'GENES'])
df = df.groupby('GENES').agg({
    'CODE': 'first',
    'CHRM': 'first',
    'STARTpos': np.median,
    'ENDpos': np.mean,
    'logfc': np.mean
}).sort_values('CHRM').dropna()
df = pd.concat([df, cnv_abs[sample].rename('cnv')], axis=1).dropna()

# Correction
df_corrected = DataCorrectionPosition(df[['cnv']], df[['logfc']])
df_corrected = df_corrected.correct(df['CHRM'], scale_pos=1, length_scale=(1e-5, 10), alpha=(1e-5, 10))

# - Export
df_corrected.to_csv('data/crispy/crispy_gdsc_fold_change_gene_unsupervised_%s.csv' % sample)
