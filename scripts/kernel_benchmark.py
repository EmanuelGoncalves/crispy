#!/usr/bin/env python
# Copyright (C) 2017 Emanuel Goncalves

import pickle
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from crispy import bipal_gray
from matplotlib.gridspec import GridSpec
from crispy.biases_correction import CRISPRCorrection
from crispy.benchmark_plot import plot_cumsum_auc, plot_cnv_rank, plot_chromosome
from sklearn.gaussian_process.kernels import WhiteKernel, ConstantKernel, RationalQuadratic, RBF


# - Imports
sample = 'AU565'

# Non-expressed genes
nexp = pickle.load(open('data/gdsc/nexp_pickle.pickle', 'rb'))[sample]

# Essential genes
essential = pd.read_csv('data/resources/curated_BAGEL_essential.csv', sep='\t')['gene'].rename('essential')

# sgRNA library
sgrna_lib = pd.read_csv('data/gdsc/crispr/KY_Library_v1.1_annotated.csv', index_col=0).dropna(subset=['STARTpos', 'GENES'])

# CRISPR fold-changes
fc_sgrna = pd.read_csv('data/gdsc/crispr/crispy_gdsc_fold_change_sgrna.csv', index_col=0)[sample]

# Copy-number absolute counts
cnv = pd.read_csv('data/gdsc/copynumber/Gene_level_CN.txt', sep='\t', index_col=0)[sample]
cnv_abs = cnv.apply(lambda v: int(v.split(',')[0]))


# - Biases correction method
# Build data-frame
df = pd.concat([
    fc_sgrna.groupby(sgrna_lib.loc[fc_sgrna.index, 'GENES']).mean().rename('logfc'),
    cnv_abs.rename('cnv'),
    sgrna_lib.groupby('GENES')['CHRM'].first().rename('chr'),
], axis=1).dropna().query('cnv != -1')

# Correction
crispy_rquad = CRISPRCorrection(
    kernel=ConstantKernel() * RationalQuadratic(length_scale_bounds=(1e-5, 10), alpha_bounds=(1e-5, 10)) + WhiteKernel()
).rename(sample).fit_by(by=df['chr'], X=df[['cnv']], y=df['logfc'])
crispy_rquad_df = pd.concat([v.to_dataframe() for k, v in crispy_rquad.items()])
crispy_rquad_df.to_csv('data/kernel_benchmark_rquad.csv')

crispy_rbf = CRISPRCorrection(
    kernel=ConstantKernel() * RBF(length_scale_bounds=(1e-5, 10)) + WhiteKernel()
).rename(sample).fit_by(by=df['chr'], X=df[['cnv']], y=df['logfc'])
crispy_rbf_df = pd.concat([v.to_dataframe() for k, v in crispy_rbf.items()])
crispy_rbf_df.to_csv('data/kernel_benchmark_rbf.csv')


# - Plots
plot_df = pd.concat([
    df,
    (crispy_rquad_df['logfc'] - crispy_rquad_df['k_mean']).rename('RQuad'), (crispy_rbf_df['logfc'] - crispy_rbf_df['k_mean']).rename('RBF'),
    (crispy_rquad_df['k_mean']).rename('RQuad_kmean'), (crispy_rbf_df['k_mean']).rename('RBF_kmean'),
    sgrna_lib.groupby('GENES')['STARTpos'].min().rename('pos')
], axis=1).dropna()

# Correlation
g = sns.jointplot(
    'RQuad', 'RBF', data=plot_df, space=0, color=bipal_gray[0], joint_kws={'s': 2, 'lw': .1, 'edgecolor': 'white', 'alpha': .4}
)
g.set_axis_labels('Rational quadratic', 'Exponentiated quadratic')
xlim, ylim = g.ax_joint.get_xlim(), g.ax_joint.get_ylim()
g.ax_joint.plot(xlim, ylim, 'k--', lw=.3)
g.ax_joint.set_xlim(xlim)
g.ax_joint.set_ylim(ylim)
plt.gcf().set_size_inches(2, 2)
plt.savefig('reports/kernel_benchmark_correlation.png', bbox_inches='tight', dpi=600)
plt.close('all')

# Essential genes AUC
plot_cumsum_auc(plot_df[['RQuad', 'RBF']], essential)
plt.gcf().set_size_inches(2, 2)
plt.savefig('reports/kernel_benchmark_essential_auc.png', bbox_inches='tight', dpi=600)
plt.close('all')

# Copy-number boxplot effects
gs = GridSpec(2, 1, hspace=.3, wspace=.3)
for i, l in enumerate(['RQuad', 'RBF']):
    ax = plt.subplot(gs[i])
    plot_cnv_rank(plot_df['cnv'].astype(int), plot_df[l], ax=ax)
plt.gcf().set_size_inches(4, 4)
plt.savefig('reports/kernel_benchmark_cnv.png', bbox_inches='tight', dpi=600)
plt.close('all')

# Plot chromossome
gs = GridSpec(2, 1, hspace=.3, wspace=.3)
for i, l in enumerate(['RQuad_kmean', 'RBF_kmean']):
    ax = plt.subplot(gs[i])
    plot_df_ = plot_df.query("chr == '8'").sort_values('pos')
    ax = plot_chromosome(plot_df_['pos'], plot_df_['logfc'], plot_df_[l], ax=ax)
    ax.set_ylabel('%s (log2 FC)' % l)

plt.gcf().set_size_inches(4, 4)
plt.savefig('reports/kernel_benchmark_chromossome_plot.png', bbox_inches='tight', dpi=600)
plt.close('all')
