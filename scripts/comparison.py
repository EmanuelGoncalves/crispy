#!/usr/bin/env python
# Copyright (C) 2017 Emanuel Goncalves

import os
import pandas as pd
import numpy as np
from datetime import datetime as dt
from crispy.data_matrix import DataMatrix
from crispy.biases_correction import CRISPRCorrection


# - Imports
# Samplesheet
ss = pd.read_csv('data/gdsc/crispr/manifest.txt', sep='\t').groupby('Truncated_Name')['Cell_Line_Name'].first().set_value('A2058', 'A2058')
controls = {'CRISPR_C6596666.sample', 'ERS717283.plasmid'}

# CRISPR files
cfiles = pd.DataFrame([{'file': i, 'sample': ss.loc[i.split('_')[0]]} for i in os.listdir('data/gdsc/crispr_comparison/gdsc/') if i.endswith('read_count.tsv')])
cfiles = cfiles.groupby('sample')['file'].agg(lambda x: set(x)).to_dict()

# sgRNA library
sgrna_lib = pd.read_csv('data/gdsc/crispr/KY_Library_v1.1_annotated.csv', index_col=0).dropna(subset=['STARTpos', 'GENES'])

# Copy-number absolute counts
cnv = pd.read_csv('data/gdsc/copynumber/Gene_level_CN.txt', sep='\t', index_col=0)
cnv = cnv.loc[:, cnv.columns.isin(list(cfiles))]
cnv_abs = cnv.applymap(lambda v: int(v.split(',')[0]))


# - Calculate fold-changes
fold_changes = []
# sample = 'HT-29'
for sample in cfiles:
    print('[%s] Preprocess Sample: %s' % (dt.now().strftime('%Y-%m-%d %H:%M:%S'), sample))

    # Read counts
    c_counts = {}
    for f in cfiles[sample]:
        df = pd.read_csv('data/gdsc/crispr_comparison/gdsc/%s' % f, sep='\t', index_col=0).drop(['gene'], axis=1).to_dict()
        c_counts.update(df)

    # Build counts data-frame (discard sgRNAs with problems)
    qc_failed_sgrnas = [
        'DHRSX_CCDS35195.1_ex1_X:2161152-2161175:+_3-1',
        'DHRSX_CCDS35195.1_ex6_Y:2368915-2368938:+_3-3',
        'DHRSX_CCDS35195.1_ex4_X:2326777-2326800:+_3-2',
        'sgPOLR2K_1'
    ]
    c_counts = DataMatrix(c_counts).drop(qc_failed_sgrnas, errors='ignore')

    # Filter by minimum required control counts
    c_counts = c_counts[c_counts[[c for c in c_counts if c in controls][0]] >= 30]

    # Build contrast matrix
    design = pd.DataFrame({sample: {c: -1 if c in controls else 1 / (c_counts.shape[1] - 1) for c in c_counts}})

    # Estimate fold-changes
    c_fc = c_counts.add_constant().counts_normalisation().log_transform().estimate_fold_change(design)

    # Store
    fold_changes.append(c_fc)

fold_changes = pd.concat(fold_changes, axis=1)
fold_changes.to_csv('data/gdsc/crispr_comparison/crispy_gdsc_foldchanges.csv')
print(fold_changes.shape)


# - CNV correction
crispy_sgrna = {}

for sample in cfiles:
    print('[%s] Correct Sample: %s' % (dt.now().strftime('%Y-%m-%d %H:%M:%S'), sample))

    # Build data-frame
    df = pd.concat([fold_changes[sample].rename('logfc'), sgrna_lib], axis=1).dropna(subset=['logfc', 'GENES'])
    df = df[df['GENES'].isin(cnv_abs.index)]
    df = df.assign(cnv=list(cnv_abs.loc[df['GENES'], sample]))

    # Correction
    df_gene = df.groupby('GENES').agg({'logfc': np.mean, 'cnv': np.mean, 'CHRM': 'first'})

    crispy = CRISPRCorrection().rename(sample).fit_by(by=df_gene['CHRM'], X=df_gene[['cnv']], y=df_gene['logfc'])
    df_corrected = pd.concat([
        v.to_dataframe(df.query("CHRM == '%s'" % k)[['cnv']], df.query("CHRM == '%s'" % k)['logfc'], return_var_exp=True) for k, v in crispy.items()
    ])
    crispy_sgrna[sample] = df_corrected.assign(GENES=sgrna_lib.loc[df_corrected.index, 'GENES']).assign(sample=sample)

for sample in cfiles:
    crispy_sgrna[sample].to_csv('data/gdsc/crispr_comparison/crispy_gdsc_corrected_sgrna_%s.csv' % sample)

crispy_sgrna_df = pd.DataFrame({s: crispy_sgrna[s]['regressed_out'] for s in cfiles})
crispy_sgrna_df.to_csv('data/gdsc/crispr_comparison/crispy_gdsc_corrected_sgrna.csv')
print(crispy_sgrna_df.shape)
