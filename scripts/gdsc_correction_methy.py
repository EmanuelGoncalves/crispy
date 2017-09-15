#!/usr/bin/env python
# Copyright (C) 2017 Emanuel Goncalves

import re
import os
import sys
import pickle
import numpy as np
import pandas as pd
import seaborn as sns
import itertools as it
from limix.stats import qvalues
import matplotlib.pyplot as plt
from limix.qtl import qtl_test_lmm, qtl_test_lm
from crispy.biases_correction import CRISPRCorrection
from crispy.benchmark_plot import plot_chromosome
from sklearn.gaussian_process import GaussianProcessRegressor
from limix_core.util.preprocess import covar_rescaling_factor_efficient
from sklearn.gaussian_process.kernels import WhiteKernel, ConstantKernel, RationalQuadratic, PairwiseKernel, RBF


# - Imports
# HGNC
hgnc = pd.read_csv('data/resources/hgnc/non_alt_loci_set.txt', sep='\t', index_col=1).dropna(subset=['location'])

# sgRNA library
sgrna_lib = pd.read_csv('data/gdsc/crispr/KY_Library_v1.1_annotated.csv', index_col=0).dropna(subset=['STARTpos'])

# Chromosome cytobands
cytobands = pd.read_csv('data/resources/cytoBand.txt', sep='\t')

# Non-expressed genes
nexp = pickle.load(open('data/gdsc/gene_expression/non_expressed_genes.pickle', 'rb'))

# Crispy corrected fold-changes
crispy_files = [f for f in os.listdir('data/crispy/') if f.startswith('crispy_gdsc_fold_change_gene_unsupervised_')]
fc_crispy = {os.path.splitext(f)[0].split('_')[6]: pd.read_csv('data/crispy/%s' % f, index_col=0) for f in crispy_files}

crispy = pd.DataFrame({i: fc_crispy[i]['corrected'] for i in fc_crispy})
crispy_original = pd.DataFrame({i: fc_crispy[i]['original'] for i in fc_crispy})

# Gene-expression
gexp = pd.read_csv('data/gdsc/gene_expression/merged_voom_preprocessed.csv', index_col=0)
gexp = gexp.loc[:, gexp.columns.isin(crispy)]

# Methylation
methy_ss = pd.read_csv('data/gdsc/methylation/samplesheet.tab', sep='\t')
methy_ss = methy_ss[methy_ss['Sample_Name'].isin(list(crispy))]
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
mobems = mobems.loc[[i.startswith('chr') for i in mobems.index], mobems.columns.isin(crispy)]


# - cg sites agg
cg = pd.read_csv('data/gdsc/methylation/HumanMethylation450_15017482_v1-2.csv', index_col=0).dropna(subset=['UCSC_RefGene_Name'])
cg['map'] = [{(g, p) for g, p in zip(*(cg_gene.split(';'), cg_pos.split(';')))} for cg_gene, cg_pos in cg[['UCSC_RefGene_Name', 'UCSC_RefGene_Group']].values]

mpos = {'TSS200'}  # ['TSS1500', 'TSS200', "5'UTR", '1stExon', 'Body', "3'UTR"]

cgp = pd.DataFrame([{'cg': c, 'gene': g, 'pos': p} for c, v in cg['map'].items() for g, p in v])
cgp = cgp[cgp['pos'].isin(mpos)]
cgp = cgp[cgp['cg'].isin(methy.index)]

methy_cgp = methy.loc[cgp['cg']].groupby(cgp['gene'].values).mean()


# -
samples = set(methy_cgp).intersection(crispy).intersection(ss.query("TCGA == 'BRCA'")['sample'])

# chrm = '17'
lm_res = []
for chrm in set(sgrna_lib['CHRM']):
    genes = set(hgnc[[chrm == re.split('q|p', i)[0] for i in hgnc['location']]].index)
    for g in genes:
        if g in crispy.index and g in methy_cgp.index:
            y, x = crispy.loc[g, samples].dropna().T, methy_cgp.loc[g, samples].dropna().T

            if len(y) > 5 and len(x) > 5:
                print(chrm, g)
                # Linear regression
                lm = qtl_test_lm(pheno=y.values, snps=x.values)

                # Result
                lm_res.append({
                    'gene': g, 'chrm': chrm,
                    'beta': lm.getBetaSNP().ravel()[0],
                    'pval': lm.getPv().ravel()[0]
                })

lm_res = pd.DataFrame(lm_res)
lm_res = lm_res.assign(qval=qvalues(lm_res['pval']))
print(lm_res.sort_values('qval'))


plt.scatter(crispy.loc['RPS14', samples], methy_cgp.loc['RPS14', samples])


# -
sample = 'AU565'

df = pd.concat([
    gexp[sample].rename('gexp'),
    sgrna_lib.groupby('GENES')['CHRM'].first(),
    sgrna_lib.groupby('GENES')['STARTpos'].mean()
], axis=1).dropna()

# Correction
chromosomes = {'17'}

df_corrected = CRISPRCorrection(df[['STARTpos']], df[['gexp']]).correct(df['CHRM'], chromosomes=chromosomes, scale_pos=1e7)
print(df_corrected)

# plot
c = '17'

ghighlight = {'ERBB2'}

plot_df = df_corrected.query("chrm == '%s'" % c)
plot_df = plot_df.assign(STARTpos=sgrna_lib.groupby('GENES')['STARTpos'].mean())

plot_chromosome(
    plot_df['STARTpos'], plot_df['original'].rename('Original FC'), plot_df['mean'].rename('Gexp'),
    highlight=ghighlight, legend=False
)

plt.gcf().set_size_inches(6, 2)
plt.savefig('reports/methy_chromossome_plot.png', bbox_inches='tight', dpi=600)
plt.close('all')

# #
# kg = {}
# # g = 'DUSP22'
# for g in set(crispy.index).intersection(methy_cgp.index):
#     # Build data-frame
#     df = pd.concat([
#         crispy.loc[g].rename('crispy'),
#         methy_cgp.loc[g].rename('methy'),
#     ], axis=1).dropna()
#
#     # Fit kernel
#     K = ConstantKernel() * PairwiseKernel(metric='linear') + WhiteKernel()
#     gp = GaussianProcessRegressor(K, n_restarts_optimizer=3, normalize_y=True).fit(df[['crispy']], df['methy'])
#
#     # Predict
#     k_mean, k_se = gp.predict(df[['crispy']], return_std=True)
#
#     # Calculate variance explained
#     kernels = [('RQ', gp.kernel_.k1), ('Noise', gp.kernel_.k2)]
#     cov_var = pd.Series({n: (1 / covar_rescaling_factor_efficient(k.__call__(df[['crispy']]))) for n, k in kernels}, name='Variance')
#     cov_var /= cov_var.sum()
#
#     kg[g] = cov_var.to_dict()
#
#     print(g, cov_var['RQ'])
#
# kg = pd.DataFrame(kg).T
# print(kg.sort_values('RQ'))
#
# # Plot
# g = 'ACO2'
#
# df = pd.concat([
#     crispy.loc[g].rename('crispy'),
#     methy_cgp.loc[g].rename('methy'),
# ], axis=1).dropna()
#
# plot_df = pd.DataFrame({
#     'crispy': df['crispy'], 'methy': df['methy'], 'mean': k_mean, 'se': k_se
# })
#
# g = sns.jointplot(
#     'crispy', 'methy', data=plot_df, kind='scatter', color='#34495e', space=0, stat_func=None,
#     marginal_kws={'hist': True, 'rug': False, 'kde': False, 'bins': 20}, annot_kws={'template': 'R={val:.2g}, pvalue={p:.1e}', 'loc': 4},
#     joint_kws={'s': 2, 'edgecolor': 'w', 'linewidth': .1, 'alpha': .5}
# )
# # g.ax_joint.scatter(plot_df['crispy'], plot_df['mean'], s=2, c='#F2C500')
# plt.gcf().set_size_inches(2, 2)
# plt.savefig('reports/methy_chromossome_plot.png', bbox_inches='tight', dpi=600)
# plt.close('all')
#
#
# # #
# # plot_df = pd.concat([df, df['STARTpos']], axis=1).dropna()
# #
# # g = sns.jointplot(
# #     'logfc', 'methy', data=plot_df, kind='scatter', color='#34495e', space=0, stat_func=None,
# #     marginal_kws={'hist': True, 'rug': False, 'kde': False, 'bins': 20}, annot_kws={'template': 'R={val:.2g}, pvalue={p:.1e}', 'loc': 4},
# #     joint_kws={'s': 2, 'edgecolor': 'w', 'linewidth': .1, 'alpha': .5}
# # )
# #
# # g.ax_joint.axvline(0, ls='-', lw=0.3, c='black', alpha=.2)
# # g.ax_joint.axhline(0, ls='-', lw=0.3, c='black', alpha=.2)
# # g.set_axis_labels('Crispy FC', 'Methylation')
# #
# # plt.suptitle(';'.join(mpos))
# #
# # plt.gcf().set_size_inches(1.5, 1.5)
# # plt.savefig('reports/chromosome_plot.png', bbox_inches='tight', dpi=600)
# # plt.close('all')
# #
# #
# # # -
# # sample = 'SW48'
# # # sample = sys.argv[1]
# #
# # df = pd.concat([
# #     fc_sgrna[sample].rename('logfc'),
# #     gexp[sample].rename('gexp'),
# #     sgrna_lib.groupby('GENES').agg({'CHRM': 'first', 'STARTpos': np.mean})
# # ], axis=1)
# # df = pd.concat([df, methy_cgp[sample].rename('methy')], axis=1).dropna()
# #
# # # # Correction
# # # metric = 'rbf'
# # # df_corrected = DataCorrectionCopyNumber(df[['methy']], df[['logfc']]).correct(df['CHRM'], metric=metric, normalize_y=True)
# # # print(df_corrected.groupby('chrm')['var_rq'].mean().sort_values())
# #
# # #
# # plot_df = pd.concat([df, df['STARTpos']], axis=1).dropna()
# #
# # g = sns.jointplot(
# #     'gexp', 'methy', data=plot_df, kind='scatter', color='#34495e', space=0, stat_func=None,
# #     marginal_kws={'hist': True, 'rug': False, 'kde': False, 'bins': 20}, annot_kws={'template': 'R={val:.2g}, pvalue={p:.1e}', 'loc': 4},
# #     joint_kws={'s': 2, 'edgecolor': 'w', 'linewidth': .1, 'alpha': .5}
# # )
# #
# # g.ax_joint.axvline(0, ls='-', lw=0.3, c='black', alpha=.2)
# # g.ax_joint.axhline(0, ls='-', lw=0.3, c='black', alpha=.2)
# # g.set_axis_labels('Gene-expression', 'Methylation')
# #
# # plt.suptitle(';'.join(mpos))
# #
# # plt.gcf().set_size_inches(1.5, 1.5)
# # plt.savefig('reports/chromosome_plot.png', bbox_inches='tight', dpi=600)
# # plt.close('all')
