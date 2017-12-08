#!/usr/bin/env python
# Copyright (C) 2017 Emanuel Goncalves

import sys
import pandas as pd
from crispy.biases_correction import CRISPRCorrection

# - Imports
# sgRNA library
sgrna_lib = pd.read_csv('data/ccle/crispr_ceres/sgrnamapping.csv').dropna(subset=['Gene'])
sgrna_lib = sgrna_lib.assign(chr=sgrna_lib['Locus'].apply(lambda x: x.split('_')[0]))
sgrna_lib_genes = sgrna_lib.groupby('Gene')['chr'].first()

# CRISPR fold-changes
fc = pd.read_csv('data/crispr_ccle_logfc.csv', index_col=0)

# Copy-number absolute counts
cnv = pd.read_csv('data/ccle/crispr_ceres/data_log2CNA.txt', sep='\t', index_col=0)


# - Biases correction
# sample = 'AU565_BREAST'
sample = sys.argv[1]
overlap = set(fc).intersection(cnv)

if sample in overlap:
    # Build data-frame
    df = pd.concat([
        fc[sample].rename('fc'),
        2 * (2 ** cnv[sample].rename('cnv')),
        sgrna_lib_genes.rename('chr')
    ], axis=1).dropna()

    # Correction
    crispy = CRISPRCorrection().rename(sample).fit_by(by=df['chr'], X=df[['cnv']], y=df['fc'])
    crispy = pd.concat([v.to_dataframe() for k, v in crispy.items()])

    # Store
    crispy.to_csv('data/crispy/crispr_ccle_crispy_%s.csv' % sample)
