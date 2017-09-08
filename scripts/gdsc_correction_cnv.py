#!/usr/bin/env python
# Copyright (C) 2017 Emanuel Goncalves

import sys
import numpy as np
import pandas as pd
from crispy.data_correction import DataCorrectionCopyNumber


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
# sample = 'AU565'
sample = sys.argv[1]

df = pd.concat([fc_sgrna[sample].rename('logfc'), sgrna_lib], axis=1).dropna(subset=['logfc', 'GENES'])
df = df.groupby('GENES').agg({
    'CODE': 'first',
    'CHRM': 'first',
    'STARTpos': np.mean,
    'ENDpos': np.mean,
    'logfc': np.mean
}).sort_values('CHRM')
df = pd.concat([df, cnv_abs[sample].rename('cnv')], axis=1).dropna()

# Correction
for metric in ['rbf', 'linear']:
    df_corrected = DataCorrectionCopyNumber(df[['cnv']], df[['logfc']]).correct(df['CHRM'], metric=metric)
    df_corrected.to_csv('data/crispy/crispy_gdsc_fold_change_gene_cnv_%s_%s.csv' % (metric, sample))
