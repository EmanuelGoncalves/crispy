#!/usr/bin/env python
# Copyright (C) 2017 Emanuel Goncalves

import pandas as pd
from bsub import bsub
from datetime import datetime as dt

# Define variables
samples = list(pd.read_csv('data/gdsc/crispr/crispy_gdsc_fold_change_sgrna.csv', index_col=0))

memory, queque, cores = 32000, 'normal', 16

# Submit a single job per sample
for sample in samples:
# for sample in ['C2BBe1', 'LoVo', 'MIA-PaCa-2', 'NCI-H3122']:
    jname = 'bsub_crispy_%s' % sample

    # Define command
    j_cmd = '/software/bin/python3.6.1 scripts/gdsc_correction.py %s' % sample

    # Create bsub
    j = bsub(
        jname, R='select[mem>%d] rusage[mem=%d] span[hosts=1]' % (memory, memory), M=memory,
        verbose=True, q=queque, J=jname, o=jname, e=jname, n=cores,
    )

    # Submit
    j(j_cmd)
    print('[%s] bsub %s' % (dt.now().strftime('%Y-%m-%d %H:%M:%S'), sample))