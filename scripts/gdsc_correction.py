#!/usr/bin/env python
# Copyright (C) 2017 Emanuel Goncalves

import sys
import pickle
import numpy as np
import pandas as pd
from crispy.biases_correction import CRISPRCorrection


# - Imports
# CRISPR fold-changes
fc = pd.read_csv('data/crispr_gdsc_logfc.csv', index_col=0)

# Copy-number absolute counts
cnv = pd.read_csv('data/gdsc/copynumber/Gene_level_CN.txt', sep='\t', index_col=0)
cnv = cnv.loc[:, cnv.columns.isin(list(fc_sgrna))]
cnv_abs = cnv.applymap(lambda v: int(v.split(',')[1]))


# - Biases correction method
# sample = 'AU565'
sample = sys.argv[1]
overlap = set(fc).intersection(cnv_abs)

if sample in overlap:
    # Correction
    df = pd.concat([fc[sample].rename('fc'), cnv_abs[sample].rename('cnv')], axis=1).dropna().query('cnv != -1')

    df_gene = df.groupby('GENES').agg({'logfc': np.mean, 'cnv': np.mean, 'CHRM': 'first'})

    crispy = CRISPRCorrection().rename(sample).fit_by(by=df_gene['CHRM'], X=df_gene[['cnv']], y=df_gene['logfc'])
    df_corrected = pd.concat([
        v.to_dataframe(df.query("CHRM == '%s'" % k)[['cnv']], df.query("CHRM == '%s'" % k)['logfc'], return_var_exp=True) for k, v in crispy.items()
    ])
    df_corrected = df_corrected.assign(GENES=sgrna_lib.loc[df_corrected.index, 'GENES']).assign(sample=sample)
    df_corrected.to_csv('data/crispy/crispy_gdsc_%s_sgrna.csv' % sample)

    # Gene-level fold-changes
    df_corrected_gene = CRISPRCorrection.logratio_by(by=df_corrected['fit_by'], x=df_corrected['GENES'], Y=df_corrected[['logfc', 'regressed_out']]).assign(sample=sample)
    df_corrected_gene.to_csv('data/crispy/crispy_gdsc_%s_gene.csv' % sample)
