#!/usr/bin/env python
# Copyright (C) 2017 Emanuel Goncalves

import pandas as pd
from bsub import bsub
from datetime import datetime as dt

# Define variables
samples = list(pd.read_csv('data/gdsc/crispr/crispy_gdsc_fold_change_sgrna.csv', index_col=0))

memory, queque, cores = 32000, 'normal', 16

# Submit a single job per sample
for sample in ['HT-29', 'AU565', 'SW48']:
    jname = 'crispy_%s' % sample

    # Define command
    j_cmd = '/software/bin/python3.6.1 sgrnas_pairwise_dist.py %s' % sample

    # Create bsub
    j = bsub(
        jname, R='select[mem>%d] rusage[mem=%d] span[hosts=1]' % (memory, memory), M=memory,
        verbose=True, q=queque, J=jname, o=jname, e=jname, n=cores,
    )

    # Submit
    j(j_cmd)
    print('[%s] bsub %s' % (dt.now().strftime('%Y-%m-%d %H:%M:%S'), sample))
