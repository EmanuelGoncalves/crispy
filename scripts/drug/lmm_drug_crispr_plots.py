#!/usr/bin/env python
# Copyright (C) 2018 Emanuel Goncalves

import numpy as np
import pandas as pd

# - Imports
ss = pd.read_csv('data/gdsc/samplesheet.csv', index_col=0).dropna(subset=['Cancer Type'])
ds = pd.read_csv('data/gdsc/drug_samplesheet.csv', index_col=0)


# - Drug targets manual curated list
d_targets = ds['Target Curated'].dropna().to_dict()
d_targets = {k: {t.strip() for t in d_targets[k].split(';')} for k in d_targets}


# - Drug-Crispr associations
lmm_df = pd.read_csv('data/drug/lmm_drug_crispr.csv')
lmm_df = lmm_df.assign(target=[int(g in d_targets[d]) if d in d_targets else np.nan for d, g in lmm_df[['drug_id', 'crispr']].values])

