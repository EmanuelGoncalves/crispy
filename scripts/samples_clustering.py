#!/usr/bin/env python
# Copyright (C) 2017 Emanuel Goncalves

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# CCLE clinical data
ccli = pd.read_csv('data/ccle/data_clinical.txt', sep='\t', index_col=1)

# PAM50 annotation
pam50 = pd.read_csv('data/gdsc/breast_pam50.csv').dropna()
pam50 = pam50.assign(sample=ccli.loc[pam50['cell_line'], 'SAMPLE_ID'].values).set_index('sample')

pam50_pal = pd.Series(dict(zip(*(set(pam50['PAM50']), sns.color_palette('tab20c', n_colors=4).as_hex()))))

# CRISPR: CERES
ceres = pd.read_csv('data/ccle/crispr_ceres/ceresgeneeffects.csv', index_col=0)

# CRISPR: logFC
sgrna_lib = pd.read_csv('data/ccle/crispr_ceres/sgrnamapping.csv').dropna(subset=['Gene'])

logfc = pd.read_csv('data/ccle/crispr_ceres/sgrnareplicateslogfcnorm.csv', index_col=0).dropna()

samplesheet = pd.read_csv('data/ccle/crispr_ceres/replicatemap.csv', index_col=0).dropna()
samples = set(logfc).intersection(samplesheet.index)

logfc = logfc.groupby(samplesheet.loc[samples, 'CellLine'], axis=1).mean()
logfc = logfc.loc[sgrna_lib['Guide']].groupby(sgrna_lib['Gene'].values).mean()

# Overlap
samples, genes = list(set(pam50.index).intersection(ceres)), list(set(ceres.index).intersection(logfc.index))
print(len(samples), len(genes))

# Most variable genes
var_genes = set(ceres[ceres.std(1) > .5].index).intersection(genes)
print(len(var_genes))

# - Clustermaps
# CERES
plot_df = ceres.loc[var_genes, samples].corr()

sns.clustermap(
    plot_df, cmap='YlGn',
    col_colors=pam50_pal[pam50.loc[plot_df.columns, 'PAM50']].values,
    row_colors=pam50_pal[pam50.loc[plot_df.index, 'PAM50']].values
)
plt.gcf().set_size_inches(6, 6)
plt.savefig('reports/breast_pam50_clustermap_ceres.png', bbox_inches='tight', dpi=600)
plt.close('all')

# logFC
plot_df = logfc.loc[var_genes, samples].corr()

sns.clustermap(
    plot_df, cmap='YlGn',
    col_colors=pam50_pal[pam50.loc[plot_df.columns, 'PAM50']].values,
    row_colors=pam50_pal[pam50.loc[plot_df.index, 'PAM50']].values
)
plt.gcf().set_size_inches(6, 6)
plt.savefig('reports/breast_pam50_clustermap_logfc.png', bbox_inches='tight', dpi=600)
plt.close('all')
