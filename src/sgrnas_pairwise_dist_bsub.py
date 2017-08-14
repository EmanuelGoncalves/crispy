#!/usr/bin/env python
# Copyright (C) 2017 Emanuel Goncalves

import pandas as pd
from bsub import bsub
from datetime import datetime as dt

# sgRNA library
sgrna_lib = pd.read_csv('data/gdsc/crispr/KY_Library_v1.0.csv')

# Define variables
memory, queque, cores = 32000, 'normal', 12

# Submit a single job per chromossome per sample
# sample, sample_chr = 'AU565', '17'
for sample in ['AU565']:
	for sample_chr in set(sgrna_lib['CHRM']):
		jname = 'crispy_%s_%s' % (sample, sample_chr)

		# Define command
		j_cmd = '/software/bin/python3.6.1 sgrnas_pairwise_dist.py %s %s' % (sample, sample_chr)

		# Create bsub
		j = bsub(
			jname, R='select[mem>%d] rusage[mem=%d] span[hosts=1]' % (memory, memory), M=memory,
			verbose=True, q=queque, J=jname, o=jname, e=jname, n=cores,
		)

		# Submit
		j(j_cmd)
		print('[%s] bsub %s %s' % (dt.now().strftime('%Y-%m-%d %H:%M:%S'), sample, sample_chr))
