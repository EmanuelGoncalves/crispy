#!/usr/bin/env python
# Copyright (C) 2018 Emanuel Goncalves

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

snp_ploidy = pd.read_csv('data/crispy_sample_ploidy.csv', index_col=0)
wgs_ploidy = pd.read_csv('data/gdsc/wgs/wgs_samplestatistics.csv', index_col='sample')

samples = set(snp_ploidy.index).intersection(wgs_ploidy.index)
print('Samples: %d' % len(samples))

plot_df = pd.concat([
    snp_ploidy.loc[samples, 'ploidy'].rename('snp'), wgs_ploidy.loc[samples, 'Ploidy'].rename('wgs')
], axis=1)

sns.jointplot('snp', 'wgs', data=plot_df)
plt.show()
