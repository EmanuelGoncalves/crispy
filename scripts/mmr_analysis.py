#!/usr/bin/env python
# Copyright (C) 2017 Emanuel Goncalves

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# - Imports
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
samples = list(ss.query("Microsatellite == 'MSI-H'").index) + ['SW620', 'OVCAR-4', 'PEO1', 'SW837', 'MFE-280', 'ESS-1']
genes = set(path['gene'])


# - Exports
gexp.loc[genes, [i for i in samples if i in gexp]].dropna().to_csv('/Users/eg14/Google Drive/mmr/tables/mmr_genes_gexp.csv')
cnv.loc[genes, [i for i in samples if i in cnv]].dropna().applymap(lambda v: int(v.split(',')[0])).to_csv('/Users/eg14/Google Drive/mmr/tables/mmr_genes_cnv.csv')
crispr.loc[genes, [i for i in samples if i in crispr]].dropna().to_csv('/Users/eg14/Google Drive/mmr/tables/mmr_genes_crispr.csv')
shrna.loc[genes, [i for i in samples if i in shrna]].dropna().to_csv('/Users/eg14/Google Drive/mmr/tables/mmr_genes_shrna.csv')
methy_cgp.loc[genes, [i for i in samples if i in methy_cgp]].dropna().to_csv('/Users/eg14/Google Drive/mmr/tables/mmr_genes_methy.csv')
ss.loc[samples].to_csv('/Users/eg14/Google Drive/mmr/tables/samplesheet.csv')


# -
ys = gexp.loc[genes, [i for i in samples if i in gexp]].dropna().T
xs = (ss.loc[ys.index, 'Microsatellite'] == 'MSI-H').astype(int).to_frame()

lm = LinearRegression().fit(xs, ys)

lm_coefs = pd.Series(lm.coef_[:, 0], index=ys.columns).sort_values()

lm_coefs.to_csv('/Users/eg14/Google Drive/mmr/gexp_msi_associations.csv')


# - Plot
g = 'CDA'

plot_df = pd.concat([
    ys[g].rename('gexp'), xs
], axis=1)

sns.boxplot('Microsatellite', 'gexp', data=plot_df, fliersize=2, palette='Reds_r', notch=False)
plt.axhline(0, ls='--', lw=.3, alpha=.5, c='black')
plt.xlabel('Microsatellite')
plt.ylabel('%s gene-expression (RNA-seq voom)' % g)

plt.show()

# plt.gcf().set_size_inches(7, 3)
# plt.savefig('reports/shrna_crispr_drive_agreement.png', bbox_inches='tight', dpi=600)
# plt.close('all')
