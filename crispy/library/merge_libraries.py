#!/usr/bin/env python
# Copyright (C) 2017 Emanuel Goncalves

import numpy as np
import pandas as pd

# - Import
# Kosuke's sgRNA library
sgrna_lib_k = pd.read_csv('data/gdsc/crispr/KY_Library_v1.0.csv')

sgrna_lib_k_exp = pd.read_csv('data/gdsc/crispr/KY_Library_v1.1.csv', sep='\t', index_col=0)

# Sabatini's sgRNA library
sgrna_lib_s = pd.read_csv('data/gdsc/crispr/sabatini_library_targets.csv', index_col=0)


# Overlap in guides
guides = set(sgrna_lib_k_exp.index).intersection(sgrna_lib_s.index)

# Shared headers
headers = ['CODE', 'GENES', 'EXONE', 'CHRM', 'STRAND', 'STARTpos', 'ENDpos']

# Expand sgRNA annotation
extra_guides = sgrna_lib_s.loc[guides].rename(columns={'Symbol': 'GENES', 'Chromosome': 'CHRM', 'Genomic strand targeted': 'STRAND', 'sgRNA location': 'STARTpos'})
extra_guides['ENDpos'] = [s + len(g) for s, g in extra_guides[['STARTpos', 'sgRNA sequence']].values]
extra_guides['CHRM'] = [i if str(i).lower() == 'nan' else i.replace('chr', '') for i in extra_guides['CHRM']]
extra_guides['EXONE'] = np.nan
extra_guides['CODE'] = np.nan

sgrna_lib_k_complete = pd.concat([sgrna_lib_k[headers], extra_guides[headers]])

# Export
sgrna_lib_k_complete.to_csv('data/gdsc/crispr/KY_Library_v1.1_annotated.csv')
