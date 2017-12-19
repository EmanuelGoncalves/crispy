#!/usr/bin/env python
# Copyright (C) 2017 Emanuel Goncalves

import crispy
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.stats import ttest_ind
from matplotlib.colors import rgb2hex
from sklearn.linear_model import LinearRegression
from statsmodels.stats.multitest import multipletests


# - Imports
# Essential genes
essential = pd.read_csv('data/resources/curated_BAGEL_essential.csv', sep='\t')['gene'].rename('essential')

# Samplesheet
ss = pd.read_csv('data/gdsc/samplesheet.csv', index_col=0)

# Pathway annot
path = pd.read_csv('data/resources/mmr_pathways.csv')

# Gene-expression
gexp = pd.read_csv('data/gdsc/gene_expression/merged_voom_preprocessed.csv', index_col=0)

# Copy-number absolute counts
cnv = pd.read_csv('data/gdsc/copynumber/Gene_level_CN.txt', sep='\t', index_col=0).dropna()

# CRISPR
crispr = pd.read_csv('data/gdsc/crispr/crisprcleanr_gene.csv', index_col=0)

# shRNA
shrna = pd.read_csv('data/ccle/DRIVE_RSA_data.txt', index_col=0, sep='\t')
shrna.columns = [i.upper() for i in shrna.columns]
shrna = shrna.rename(columns=ss.reset_index().dropna(subset=['CCLE ID']).groupby('CCLE ID')['Cell Line Name'].first().to_dict())
shrna = shrna.dropna(axis=1)

# BROAD crispr
crispr_ccle = pd.read_csv('data/ccle/crispr_ceres/ceresgeneeffects.csv', index_col=0)
crispr_ccle = crispr_ccle.rename(columns=ss.reset_index().dropna(subset=['CCLE ID']).groupby('CCLE ID')['Cell Line Name'].first().to_dict())
crispr_ccle = crispr_ccle.dropna(axis=1)

# Methylation
methy_ss = pd.read_csv('data/gdsc/methylation/samplesheet.tab', sep='\t')
methy_ss = methy_ss.assign(id=methy_ss['Sentrix_ID'].astype(str) + '_' + methy_ss['Sentrix_Position'].astype(str))
methy_ss = methy_ss.set_index('id')

methy = pd.read_csv('data/gdsc/methylation/sanger_methy_minfi_beta.tab', sep='\t', index_col=0)
methy = methy.loc[:, methy.columns.isin(methy_ss.index)]
methy = methy.rename(columns=methy_ss['Sample_Name'])

cg = pd.read_csv('data/gdsc/methylation/HumanMethylation450_15017482_v1-2.csv', index_col=0).dropna(subset=['UCSC_RefGene_Name'])
cg['map'] = [{(g, p) for g, p in zip(*(cg_gene.split(';'), cg_pos.split(';')))} for cg_gene, cg_pos in cg[['UCSC_RefGene_Name', 'UCSC_RefGene_Group']].values]

mpos = {'TSS1500', 'TSS200', "5'UTR", '1stExon'}  # [, 'Body', "3'UTR"]

cgp = pd.DataFrame([{'cg': c, 'gene': g, 'pos': p} for c, v in cg['map'].items() for g, p in v])
cgp = cgp[cgp['pos'].isin(mpos)]
cgp = cgp[cgp['cg'].isin(methy.index)]

methy_cgp = methy.loc[cgp['cg']].groupby(cgp['gene'].values).mean()


# - Lists
genes = set(path['gene'])

samples = set(ss.dropna(subset=['Microsatellite']).index).intersection(set(crispr).union(set(crispr_ccle)).union(set(shrna)))
print(len(samples), len(genes))


# - Exports
gexp.loc[genes, [i for i in samples if i in gexp]].dropna().to_csv('/Users/eg14/Google Drive/mmr/tables/mmr_genes_gexp.csv')
cnv.loc[genes, [i for i in samples if i in cnv]].dropna().applymap(lambda v: int(v.split(',')[0])).to_csv('/Users/eg14/Google Drive/mmr/tables/mmr_genes_cnv.csv')
crispr.loc[genes, [i for i in samples if i in crispr]].dropna().to_csv('/Users/eg14/Google Drive/mmr/tables/mmr_genes_crispr.csv')
shrna.loc[genes, [i for i in samples if i in shrna]].dropna().to_csv('/Users/eg14/Google Drive/mmr/tables/mmr_genes_shrna.csv')
methy_cgp.loc[genes, [i for i in samples if i in methy_cgp]].dropna().to_csv('/Users/eg14/Google Drive/mmr/tables/mmr_genes_methy.csv')
crispr_ccle.loc[genes, [i for i in samples if i in crispr_ccle]].dropna().to_csv('/Users/eg14/Google Drive/mmr/tables/mmr_genes_crispr_broad.csv')
ss.loc[samples].to_csv('/Users/eg14/Google Drive/mmr/tables/samplesheet.csv')


# - Differential gene-expression
ttest_msi = {}
for g in genes:
    if g in gexp.index:
        df = pd.concat([
            gexp.loc[g, samples].dropna().rename('gexp'),
            ss.loc[samples, 'Microsatellite']
        ], axis=1).dropna()

        t, p = ttest_ind(df.query("Microsatellite == 'MSI-H'")['gexp'], df.query("Microsatellite == 'MSI-S'")['gexp'], equal_var=False)

        ttest_msi[g] = {'t_stat': t, 'p_value': p}

ttest_msi = pd.DataFrame(ttest_msi).T.sort_values('p_value')
ttest_msi = ttest_msi.assign(fdr=multipletests(ttest_msi['p_value'], method='fdr_bh')[1])

ttest_msi.to_csv('/Users/eg14/Google Drive/mmr/gexp_msi_associations.csv')


# - Plot
for g in ttest_msi.head(10).index:
    pal = pd.Series(sns.color_palette('tab20c', n_colors=len(order)).as_hex(), index=order)

    plot_df = pd.concat([
        gexp.loc[g, samples].dropna().rename('gexp'),
        ss.loc[samples, 'Microsatellite']
    ], axis=1).dropna()

    order = ['MSI-S', 'MSI-L', 'MSI-H']
    sns.boxplot('Microsatellite', 'gexp', data=plot_df, palette=pal, notch=False, linewidth=.3, order=order, sym='')
    sns.stripplot('Microsatellite', 'gexp', data=plot_df, linewidth=.3, edgecolor='white', jitter=.2, size=1.5, palette=pal, order=order)
    plt.axhline(0, ls='--', lw=.3, alpha=.5, c='black')
    plt.xlabel('Microsatellite')
    plt.ylabel('Gene-expression (RNA-seq voom)')
    plt.title('%s - FDR=%.2e' % (g, ttest_msi.loc[g, 'fdr']))

    plt.gcf().set_size_inches(2, 3)
    plt.savefig('/Users/eg14/Google Drive/mmr/reports/gexp_msi_%s.png' % g, bbox_inches='tight', dpi=600)
    plt.close('all')


# -
plot_df = gexp.loc[ttest_msi.query('fdr < 0.05').index, samples].dropna(axis=1)

pal = sns.color_palette('tab20c', n_colors=4).as_hex()
pal = pd.Series([pal[3], pal[0]], index=[0, 1])

col_colors = pd.concat([
    (ss.loc[plot_df.columns, 'Microsatellite'] == 'MSI-H').astype(int).replace(pal).rename('Microsatellite = MSI-H'),
    pd.Series({c: pal[int(crispr.loc['WRN', c] < -1)] if c in crispr.columns else '#FFFFFF' for c in plot_df}).rename('GDSC CRISPR WRN < -1'),
    pd.Series({c: pal[int(crispr_ccle.loc['WRN', c] < -0.5)] if c in crispr_ccle.columns else '#FFFFFF' for c in plot_df}).rename('CCLE CRISPR WRN < -0.5'),
    pd.Series({c: pal[int(shrna.loc['WRN', c] < -3)] if c in shrna.columns else '#FFFFFF' for c in plot_df}).rename('shRNA RSA WRN < -3'),
], axis=1)

sns.set(style='white', font_scale=.5)
g = sns.clustermap(plot_df, col_colors=col_colors, metric='correlation', cmap='GnBu_r', row_cluster=False, figsize=(10, 4), side_colors_ratio=.15)
# plt.gcf().set_size_inches(10, 4)
plt.savefig('/Users/eg14/Google Drive/mmr/reports/heatmap_gexp_msi.png', bbox_inches='tight', dpi=600)
plt.close('all')

