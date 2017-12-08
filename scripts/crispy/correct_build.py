#!/usr/bin/env python
# Copyright (C) 2017 Emanuel Goncalves

import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from crispy import bipal_gray
from crispy.benchmark_plot import plot_cumsum_auc

# - Import
# Essential genes
essential = pd.read_csv('data/resources/curated_BAGEL_essential.csv', sep='\t')['gene'].rename('essential')

# Non-essential genes
nessential = pd.read_csv('data/resources/curated_BAGEL_nonEssential.csv', sep='\t')['gene'].rename('non-essential')

# Copy-number absolute counts
gdsc_cnv = pd.read_csv('data/gdsc/copynumber/Gene_level_CN.txt', sep='\t', index_col=0).dropna()


# - Assemble
# GDSC
c_gdsc_fc = pd.read_csv('data/crispr_gdsc_logfc.csv', index_col=0)

c_gdsc = pd.DataFrame({
    os.path.splitext(f)[0].replace('crispr_gdsc_crispy_', ''):
        pd.read_csv('data/crispy/' + f, index_col=0)['regressed_out']
    for f in os.listdir('data/crispy/') if f.startswith('crispr_gdsc_crispy_')
}).dropna()
c_gdsc.round(5).to_csv('data/crispr_gdsc_crispy.csv')

# CCLE
c_ccle_fc = pd.read_csv('data/crispr_ccle_logfc.csv', index_col=0)

c_ccle = pd.DataFrame({
    os.path.splitext(f)[0].replace('crispr_ccle_crispy_', ''):
        pd.read_csv('data/crispy/' + f, index_col=0)['regressed_out']
    for f in os.listdir('data/crispy/') if f.startswith('crispr_ccle_crispy_')
}).dropna()
c_ccle.round(5).to_csv('data/crispr_ccle_crispy.csv')


# - Benchmark
# Essential/Non-essential genes AUC
for n, df, df_ in [('gdsc', c_gdsc, c_gdsc_fc)]:
    # Essential AUCs corrected fold-changes
    ax, stats_ess = plot_cumsum_auc(df, essential, plot_mean=True, legend=False)
    plt.gcf().set_size_inches(3, 3)
    plt.savefig('reports/processing_%s_crispy_essential_auc.png' % n, bbox_inches='tight', dpi=600)
    plt.close('all')

    # Essential AUCs original fold-changes
    ax, stats_ess_fc = plot_cumsum_auc(df_, essential, plot_mean=True, legend=False)
    plt.gcf().set_size_inches(3, 3)
    plt.savefig('reports/processing_%s_crispy_essential_auc_original_fc.png' % n, bbox_inches='tight', dpi=600)
    plt.close('all')

    # Scatter plot of AUCs
    plot_df = pd.concat([pd.Series(stats_ess['auc'], name='Corrected'), pd.Series(stats_ess_fc['auc'], name='Original')], axis=1)

    ax = plt.gca()

    ax.scatter(plot_df['Original'], plot_df['Corrected'], c=bipal_gray[0], s=3, marker='x', lw=0.5)
    sns.rugplot(plot_df['Original'], height=.02, axis='x', c=bipal_gray[0], lw=.3, ax=ax)
    sns.rugplot(plot_df['Corrected'], height=.02, axis='y', c=bipal_gray[0], lw=.3, ax=ax)

    x_lim, y_lim = ax.get_xlim(), ax.get_ylim()
    ax.plot(x_lim, y_lim, 'k--', lw=.3)

    ax.set_xlim(x_lim)
    ax.set_ylim(y_lim)

    ax.set_xlabel('Original fold-changes AUCs')
    ax.set_ylabel('Corrected fold-changes AUCs')
    ax.set_title('Essential genes')

    plt.gcf().set_size_inches(2, 2)
    plt.savefig('reports/processing_%s_crispy_essential_auc_scatter.png' % n, bbox_inches='tight', dpi=600)
    plt.close('all')

for n, df, df_ in [('gdsc', c_gdsc, c_gdsc_fc)]:
    # Non-essential AUCs corrected fold-changes
    ax, stats_ness = plot_cumsum_auc(df, nessential, plot_mean=True, legend=False)
    plt.gcf().set_size_inches(3, 3)
    plt.savefig('reports/processing_%s_crispy_nonessential_auc.png' % n, bbox_inches='tight', dpi=600)
    plt.close('all')

    # Non-essential AUCs original fold-changes
    ax, stats_ness_fc = plot_cumsum_auc(df_, nessential, plot_mean=True, legend=False)
    plt.gcf().set_size_inches(3, 3)
    plt.savefig('reports/processing_%s_crispy_nonessential_auc_original_fc.png' % n, bbox_inches='tight', dpi=600)
    plt.close('all')

    # Scatter plot of AUCs
    plot_df = pd.concat([pd.Series(stats_ness['auc'], name='Corrected'), pd.Series(stats_ness_fc['auc'], name='Original')], axis=1)

    ax = plt.gca()

    ax.scatter(plot_df['Original'], plot_df['Corrected'], c=bipal_gray[0], s=3, marker='x', lw=0.5)
    sns.rugplot(plot_df['Original'], height=.02, axis='x', c=bipal_gray[0], lw=.3, ax=ax)
    sns.rugplot(plot_df['Corrected'], height=.02, axis='y', c=bipal_gray[0], lw=.3, ax=ax)

    x_lim, y_lim = ax.get_xlim(), ax.get_ylim()
    ax.plot(x_lim, y_lim, 'k--', lw=.3)

    ax.set_xlim(x_lim)
    ax.set_ylim(y_lim)

    ax.set_xlabel('Original fold-changes AUCs')
    ax.set_ylabel('Corrected fold-changes AUCs')
    ax.set_title('Non-essential genes')

    plt.gcf().set_size_inches(2, 2)
    plt.savefig('reports/processing_%s_crispy_nonessential_auc_scatter.png' % n, bbox_inches='tight', dpi=600)
    plt.close('all')


# Copy-number bias
samples = set(gdsc_cnv).intersection(c_gdsc_fc)
gdsc_cnv = gdsc_cnv[list(samples)]

