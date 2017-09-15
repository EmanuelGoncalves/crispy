#!/usr/bin/env python
# Copyright (C) 2017 Emanuel Goncalves

import sys
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from crispy.benchmark_plot import plot_chromosome
from crispy.biases_correction import CRISPRCorrection


# - Imports
# sgRNA library
sgrna_lib = pd.read_csv('data/gdsc/crispr/KY_Library_v1.1_annotated.csv', index_col=0).dropna(subset=['STARTpos', 'GENES'])

# CRISPR fold-changes
fc_sgrna = pd.read_csv('data/gdsc/crispr/crispy_gdsc_fold_change_sgrna.csv', index_col=0)

# Copy-number absolute counts
cnv = pd.read_csv('data/gdsc/copynumber/Gene_level_CN.txt', sep='\t', index_col=0)
cnv = cnv.loc[:, cnv.columns.isin(list(fc_sgrna))]
cnv_abs = cnv.applymap(lambda v: int(v.split(',')[0]))


# - Biases correction method
sample = 'AU565'
# sample = sys.argv[1]

# Build data-frame
df = pd.concat([fc_sgrna[sample].rename('logfc'), sgrna_lib], axis=1).dropna(subset=['logfc', 'GENES'])
df = df[df['GENES'].isin(cnv_abs.index)]
df = df.assign(cnv=list(cnv_abs.loc[df['GENES'], sample]))

# Correction
df_gene = df.groupby('GENES').agg({'logfc': np.mean, 'cnv': np.mean, 'CHRM': 'first'})
crispy = CRISPRCorrection().rename(sample).fit_by(by=df_gene['CHRM'], X=df_gene[['cnv']], y=df_gene['logfc'])

df_corrected = pd.concat([
    v.to_dataframe(df.query("CHRM == '%s'" % k)[['cnv']], df.query("CHRM == '%s'" % k)['logfc']) for k, v in crispy.items()
])

# Plot
c = '8'

ghighlight = None

plot_df = df_corrected.query("fit_by == '%s'" % c)
plot_df = plot_df.assign(STARTpos=sgrna_lib.groupby('GENES')['STARTpos'].mean())

plot_chromosome(
    plot_df['STARTpos'], (plot_df['regressed_out'] + plot_df['mean']), (plot_df['k_mean'] - plot_df['mean']),
    highlight=ghighlight, legend=False
)

plt.ylabel('')
plt.title('Sample: %s; Chr: %s' % (sample, c))
plt.gcf().set_size_inches(3, 1.5)
plt.savefig('reports/test_chromosome_plot.png', bbox_inches='tight', dpi=600)
plt.close('all')

# # - Export
# df_corrected.to_csv('data/crispy/crispy_gdsc_fold_change_gene_unsupervised_%s.csv' % sample)
