#!/usr/bin/env python
# Copyright (C) 2017 Emanuel Goncalves

import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Ridge


# - Imports
# Copy-number ratio
cnv_ratio = pd.read_csv('data/copy_number_ratio.csv')
cnv_ratio_m = pd.pivot_table(cnv_ratio, index='gene', columns='sample', values='ratio')

# CRISPR bias
c_gdsc = pd.DataFrame({
    os.path.splitext(f)[0].replace('crispr_gdsc_crispy_', ''):
        pd.read_csv('data/crispy/' + f, index_col=0)['k_mean']
    for f in os.listdir('data/crispy/') if f.startswith('crispr_gdsc_crispy_')
}).dropna()

# MOBEMs
mobems = pd.read_csv('data/gdsc/mobems/PANCAN_simple_MOBEM.rdata.annotated.csv', index_col=0)


# -
samples = list(set(mobems).intersection(c_gdsc))
print(len(samples))


# -
xs = mobems[samples].T
xs = xs.loc[:, xs.sum() >= 3]

ys = cnv_ratio_m[xs.index].dropna().T

lm = Ridge().fit(ys, xs)

lm_coef = pd.DataFrame(lm.coef_, index=xs.columns, columns=ys.columns)
lm_coef_df = lm_coef.unstack().sort_values()
print(lm_coef_df)


# -
plot_df = pd.concat([
    gexp.loc[g, samples].dropna().rename('gexp'),
    xs['TP53_mut']
], axis=1).dropna()

sns.boxplot('Microsatellite', 'gexp', data=plot_df, palette=pal, notch=False, linewidth=.3, order=order, sym='')
sns.stripplot('Microsatellite', 'gexp', data=plot_df, linewidth=.3, edgecolor='white', jitter=.2, size=1.5, palette=pal, order=order)
