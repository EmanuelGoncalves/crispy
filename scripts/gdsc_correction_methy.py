#!/usr/bin/env python
# Copyright (C) 2017 Emanuel Goncalves

import sys
import pickle
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from crispy.benchmark_plot import plot_chromosome
from crispy.data_correction import DataCorrectionCopyNumber


# - Imports
# sgRNA library
sgrna_lib = pd.read_csv('data/gdsc/crispr/KY_Library_v1.1_annotated.csv', index_col=0).dropna(subset=['STARTpos'])

# Non-expressed genes
nexp = pickle.load(open('data/gdsc/gene_expression/non_expressed_genes.pickle', 'rb'))

# CRISPR fold-changes
fc_sgrna = pd.read_csv('data/gdsc/crispr/crispy_gdsc_fold_change_sgrna_ccrispy.csv', index_col=0)

# Gene-expression
gexp = pd.read_csv('data/gdsc/gene_expression/merged_voom_preprocessed.csv', index_col=0)
gexp = gexp.loc[:, gexp.columns.isin(fc_sgrna)]

# Methylation
methy_ss = pd.read_csv('data/gdsc/methylation/samplesheet.tab', sep='\t')
methy_ss = methy_ss[methy_ss['Sample_Name'].isin(list(fc_sgrna))]
methy_ss = methy_ss.assign(id=methy_ss['Sentrix_ID'].astype(str) + '_' + methy_ss['Sentrix_Position'].astype(str))
methy_ss = methy_ss.set_index('id')

methy = pd.read_csv('data/gdsc/methylation/sanger_methy_minfi_beta.tab', sep='\t', index_col=0)
methy = methy.loc[:, methy.columns.isin(methy_ss.index)]
methy = methy.rename(columns=methy_ss['Sample_Name'])

# Samplesheet
ss = pd.read_csv('data/gdsc/samplesheet.csv', index_col=1)

mobems = pd.read_csv('data/gdsc/mobems/PANCAN_simple_MOBEM.rdata.tsv', sep='\t', index_col=0)
mobems.columns = [int(i) for i in mobems.columns]
mobems = mobems.rename(columns=ss['sample'])
mobems = mobems.loc[[i.startswith('chr') for i in mobems.index], mobems.columns.isin(fc_sgrna)]


# - cg sites agg
cg = pd.read_csv('data/gdsc/methylation/HumanMethylation450_15017482_v1-2.csv', index_col=0).dropna(subset=['UCSC_RefGene_Name'])
cg['map'] = [{(g, p) for g, p in zip(*(cg_gene.split(';'), cg_pos.split(';')))} for cg_gene, cg_pos in cg[['UCSC_RefGene_Name', 'UCSC_RefGene_Group']].values]

mpos = {'TSS1500'}  # ['TSS1500', 'TSS200', "5'UTR", '1stExon', 'Body', "3'UTR"]

cgp = pd.DataFrame([{'cg': c, 'gene': g, 'pos': p} for c, v in cg['map'].items() for g, p in v])
cgp = cgp[cgp['pos'].isin(mpos)]
cgp = cgp[cgp['cg'].isin(methy.index)]

methy_cgp = methy.loc[cgp['cg']].groupby(cgp['gene'].values).mean()


# -
sample = 'SW48'
# sample = sys.argv[1]

df = pd.concat([
    fc_sgrna[sample].rename('logfc'),
    sgrna_lib.groupby('GENES').agg({'CHRM': 'first', 'STARTpos': np.mean})
], axis=1)
df = pd.concat([df, methy_cgp[sample].rename('methy')], axis=1).dropna()

# # Correction
# metric = 'rbf'
# df_corrected = DataCorrectionCopyNumber(df[['methy']], df[['logfc']]).correct(df['CHRM'], metric=metric, normalize_y=True)
# print(df_corrected.groupby('chrm')['var_rq'].mean().sort_values())

#
plot_df = pd.concat([df, df['STARTpos']], axis=1).dropna()

g = sns.jointplot(
    'logfc', 'methy', data=plot_df, kind='scatter', color='#34495e', space=0, stat_func=None,
    marginal_kws={'hist': True, 'rug': False, 'kde': False, 'bins': 20}, annot_kws={'template': 'R={val:.2g}, pvalue={p:.1e}', 'loc': 4},
    joint_kws={'s': 2, 'edgecolor': 'w', 'linewidth': .1, 'alpha': .5}
)

g.ax_joint.axvline(0, ls='-', lw=0.3, c='black', alpha=.2)
g.ax_joint.axhline(0, ls='-', lw=0.3, c='black', alpha=.2)
g.set_axis_labels('Crispy FC', 'Methylation')

plt.suptitle(';'.join(mpos))

plt.gcf().set_size_inches(1.5, 1.5)
plt.savefig('reports/chromosome_plot.png', bbox_inches='tight', dpi=600)
plt.close('all')


# -
sample = 'SW48'
# sample = sys.argv[1]

df = pd.concat([
    fc_sgrna[sample].rename('logfc'),
    gexp[sample].rename('gexp'),
    sgrna_lib.groupby('GENES').agg({'CHRM': 'first', 'STARTpos': np.mean})
], axis=1)
df = pd.concat([df, methy_cgp[sample].rename('methy')], axis=1).dropna()

# # Correction
# metric = 'rbf'
# df_corrected = DataCorrectionCopyNumber(df[['methy']], df[['logfc']]).correct(df['CHRM'], metric=metric, normalize_y=True)
# print(df_corrected.groupby('chrm')['var_rq'].mean().sort_values())

#
plot_df = pd.concat([df, df['STARTpos']], axis=1).dropna()

g = sns.jointplot(
    'gexp', 'methy', data=plot_df, kind='scatter', color='#34495e', space=0, stat_func=None,
    marginal_kws={'hist': True, 'rug': False, 'kde': False, 'bins': 20}, annot_kws={'template': 'R={val:.2g}, pvalue={p:.1e}', 'loc': 4},
    joint_kws={'s': 2, 'edgecolor': 'w', 'linewidth': .1, 'alpha': .5}
)

g.ax_joint.axvline(0, ls='-', lw=0.3, c='black', alpha=.2)
g.ax_joint.axhline(0, ls='-', lw=0.3, c='black', alpha=.2)
g.set_axis_labels('Gene-expression', 'Methylation')

plt.suptitle(';'.join(mpos))

plt.gcf().set_size_inches(1.5, 1.5)
plt.savefig('reports/chromosome_plot.png', bbox_inches='tight', dpi=600)
plt.close('all')
