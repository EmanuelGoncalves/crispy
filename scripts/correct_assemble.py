#!/usr/bin/env python
# Copyright (C) 2017 Emanuel Goncalves

import os
import pandas as pd
import matplotlib.pyplot as plt
from crispy.benchmark_plot import plot_cumsum_auc

# - Import
# Essential genes
essential = pd.read_csv('data/resources/curated_BAGEL_essential.csv', sep='\t')['gene'].rename('essential')

# Non-essential genes
nessential = pd.read_csv('data/resources/curated_BAGEL_nonEssential.csv', sep='\t')['gene'].rename('non-essential')


# - Assemble
# GDSC
c_gdsc = pd.DataFrame({
    os.path.splitext(f)[0].replace('crispr_gdsc_crispy_', ''):
        pd.read_csv('data/crispy/' + f, index_col=0)['regressed_out']
    for f in os.listdir('data/crispy/') if f.startswith('crispr_gdsc_crispy_')
}).dropna()
c_gdsc.round(5).to_csv('data/crispr_gdsc_crispy.csv')

# CCLE
c_ccle = pd.DataFrame({
    os.path.splitext(f)[0].replace('crispr_ccle_crispy_', ''):
        pd.read_csv('data/crispy/' + f, index_col=0)['regressed_out']
    for f in os.listdir('data/crispy/') if f.startswith('crispr_ccle_crispy_')
}).dropna()
c_ccle.round(5).to_csv('data/crispr_ccle_crispy.csv')


# - Benchmark
# Essential/Non-essential genes AUC
for n, df in [('gdsc', c_gdsc), ('ccle', c_ccle)]:
    ax, stats_ess = plot_cumsum_auc(df, essential, plot_mean=True, legend=False)
    plt.gcf().set_size_inches(3, 3)
    plt.savefig('reports/processing_%s_crispy_essential_auc.png' % n, bbox_inches='tight', dpi=600)
    plt.close('all')

for n, df in [('gdsc', c_gdsc), ('ccle', c_ccle)]:
    ax, stats_ness = plot_cumsum_auc(df, nessential, plot_mean=True, legend=False)
    plt.gcf().set_size_inches(3, 3)
    plt.savefig('reports/processing_%s_crispy_nonessential_auc.png' % n, bbox_inches='tight', dpi=600)
    plt.close('all')
