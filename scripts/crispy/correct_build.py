#!/usr/bin/env python
# Copyright (C) 2017 Emanuel Goncalves

import os
import pickle
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from natsort import natsorted
from crispy import bipal_gray
from sklearn.metrics import roc_curve, auc
from crispy.benchmark_plot import plot_cumsum_auc

# - Import
# Non-expressed genes
nexp = pickle.load(open('data/gdsc/nexp_pickle.pickle', 'rb'))

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
plot_df = pd.DataFrame({c: c_gdsc_fc.loc[nexp[c], c] for c in c_gdsc_fc if c in nexp})
plot_df = pd.concat([
    plot_df.unstack().rename('fc'),
    gdsc_cnv.loc[plot_df.index, plot_df.columns].unstack().rename('cnv')
], axis=1).dropna()
plot_df = plot_df.assign(cnv=plot_df['cnv'].apply(lambda v: int(v.split(',')[1])).values).query('cnv >= 2')
plot_df = plot_df.assign(cnv=[str(i) if i < 10 else '10+' for i in plot_df['cnv']])

thresholds = natsorted(set(plot_df['cnv']))
thresholds_color = sns.light_palette(bipal_gray[0], n_colors=len(thresholds)).as_hex()

ax = plt.gca()

for c, thres in zip(thresholds_color, thresholds):
    fpr, tpr, _ = roc_curve((plot_df['cnv'] == thres).astype(int), -plot_df['fc'])
    ax.plot(fpr, tpr, label='%s: AUC=%.2f' % (thres, auc(fpr, tpr)), lw=1., c=c)

ax.plot((0, 1), (0, 1), 'k--', lw=.3, alpha=.5)
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.set_xlabel('False positive rate')
ax.set_ylabel('True positive rate')
ax.set_title('CRISPR/Cas9 copy-number amplification bias\nnon-expressed genes')
legend = ax.legend(loc=4, title='Copy-number', prop={'size': 6})
legend.get_title().set_fontsize('7')

# plt.show()
plt.gcf().set_size_inches(3, 3)
plt.savefig('reports/processing_gdsc_copy_number_bias.png', bbox_inches='tight', dpi=600)
plt.close('all')
