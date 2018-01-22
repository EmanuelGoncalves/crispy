#!/usr/bin/env python
# Copyright (C) 2017 Emanuel Goncalves

import sys
import pandas as pd
from crispy.biases_correction import CRISPRCorrection

# - Imports
# sgRNA library
sgrna_lib = pd.read_csv('data/gdsc/crispr/KY_Library_v1.1_annotated.csv', index_col=0).dropna(subset=['GENES'])
sgrna_lib_genes = sgrna_lib.groupby('GENES')['CHRM'].first()

# CRISPR fold-changes
fc = pd.read_csv('data/crispr_gdsc_logfc.csv', index_col=0)

# Copy-number absolute counts
cnv = pd.read_csv('data/gdsc/copynumber/Gene_level_CN.txt', sep='\t', index_col=0)
cnv = cnv.loc[:, cnv.columns.isin(list(fc))]
cnv_abs = cnv.applymap(lambda v: int(v.split(',')[1]))


# - Biases correction
# sample = 'AU565'
sample = sys.argv[1]
overlap = set(fc).intersection(cnv_abs)

if sample in overlap:
    # Build data-frame
    df = pd.concat([
        fc[sample].rename('fc'),
        cnv_abs[sample].rename('cnv'),
        sgrna_lib_genes.rename('chr')
    ], axis=1).dropna().query('cnv != -1')

    # Correction
    crispy = CRISPRCorrection().rename(sample).fit_by(by=df['chr'], X=df[['cnv']], y=df['fc'])
    crispy = pd.concat([v.to_dataframe() for k, v in crispy.items()])

    # Store
    crispy.to_csv('data/crispy/crispr_gdsc_crispy_%s.csv' % sample)
