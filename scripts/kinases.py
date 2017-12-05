#!/usr/bin/env python
# Copyright (C) 2017 Emanuel Goncalves

import pickle
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from limix.qtl import qtl_test_lmm
from statsmodels.stats.weightstats import ztest
from sklearn.linear_model import LinearRegression
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve, precision_recall_curve, auc

# - Imports
# Essential genes
essential = pd.read_csv('data/resources/curated_BAGEL_essential.csv', sep='\t')['gene'].rename('essential')

# Cancer genes
cgenes = pd.read_csv('data/resources/cancer_genes.txt', sep='\t')
cgenes = cgenes[cgenes['role'].isin(['Loss of function', 'Activating'])]
cgenes = cgenes.groupby('SYM')['role'].first()

# Uniprot
uniprot = pd.Series(pickle.load(open('data/resources/uniprot_to_genename.pickle', 'rb')))

# Omnipath
omnipath = pd.read_csv('data/resources/omnipath_ptms_directed.txt', sep='\t')
omnipath = omnipath[omnipath['modification'].isin(['phosphorylation', 'dephosphorylation'])]
# omnipath = omnipath[omnipath['is_stimulation'] != omnipath['is_inhibition']]
omnipath = omnipath[omnipath[['is_stimulation', 'is_inhibition']].sum(1) < 2]
omnipath = omnipath.assign(enzyme_g=uniprot[omnipath['enzyme']].values)
omnipath = omnipath.assign(substrate_g=uniprot[omnipath['substrate']].values)
omnipath = omnipath[[len(set(i.split(';')).intersection({'Signor', 'PhosphoSite'})) >= 1 for i in omnipath['sources']]]

#
ccrispy = pd.read_csv('data/crispr_gdsc_crispy.csv', index_col=0)

#
phospho = pd.read_csv('data/gdsc/proteomics/phosphoproteomics_coread_processed.csv', index_col=0)
phospho.columns = [c.split('_')[1] for c in phospho.columns]

proteomics = pd.read_csv('data/gdsc/proteomics/proteomics_coread_processed.csv', index_col=0)
proteomics.columns = [c.split('_')[1] for c in proteomics.columns]

# Drug response single-agent
d_response = pd.read_csv('data/gdsc/drug_single/drug_ic50_merged_matrix.csv', index_col=[0, 1, 2], header=[0, 1])
d_response.columns = d_response.columns.droplevel(0)
d_response.index = ['_'.join([str(d_id), name, sc]) for d_id, name, sc in d_response.index]

#
samplesheet = pd.read_csv('data/gdsc/samplesheet.csv', index_col=0).loc[ccrispy.columns]

#
kfam = pd.read_csv('data/resources/pkinfam_parsed.txt')
kfam = kfam[kfam['kinase'].isin(ccrispy.index)]

# Kphospho
knet = pd.read_csv('data/resources/Kinase_Substrate_Dataset.txt', sep='\t')
knet = knet.query("KIN_ORGANISM == 'human' & SUB_ORGANISM == 'human'")
knet = knet[knet['GENE'].isin(ccrispy.index) & knet['SUB_GENE'].isin(ccrispy.index)]
knet = knet[knet['GENE'] != knet['SUB_GENE']]

# -
genes, samples = set(ccrispy.index), set(ccrispy)


# -
std_genes = ccrispy[(ccrispy.abs() >= 1).sum(1) >= 3].copy()
std_genes = std_genes[(std_genes < -1).sum(1) < (std_genes.shape[1] * .9)]
std_genes = std_genes.drop(essential, axis=0, errors='ignore')
std_genes = set(std_genes.index)

omnipath = omnipath.groupby(['enzyme_g', 'substrate_g'])['is_stimulation'].first().reset_index()
omnipath = omnipath[omnipath['enzyme_g'].isin(std_genes) & omnipath['substrate_g'].isin(std_genes)]
omnipath = omnipath.assign(corr=[pearsonr(ccrispy.loc[s], ccrispy.loc[t])[0] for s, t in omnipath[['enzyme_g', 'substrate_g']].values])

fpr, tpr, _ = roc_curve(omnipath['is_stimulation'], omnipath['corr'])
plt.plot(fpr, tpr, label='%.2f (AUC)' % auc(fpr, tpr), lw=1.)
plt.plot((0, 1), (0, 1), 'k--', lw=.3, alpha=.5)
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.legend()
plt.show()


sns.boxplot('is_stimulation', 'corr', data=omnipath, notch=True)
plt.show()

plot_df = pd.concat([ccrispy.loc['MTOR'], ccrispy.loc['DNMT1'], samplesheet['Cancer Type']], axis=1).dropna()
sns.jointplot('MTOR', 'DNMT1', data=plot_df)
plt.show()

# ['P26358_S143', 'P26358_S143_S152', 'P26358_S154', 'P26358_S714']
plot_df = pd.concat([
    proteomics.loc['P26358'],
    phospho.loc['P26358_S714'],
    ccrispy.loc['MTOR'],
    ccrispy.loc['DNMT1'],
], axis=1).dropna()
sns.pairplot(plot_df)
plt.show()


# -
g = 'PLK4'
sns.clustermap(ccrispy.loc[set(knet.query("GENE == '%s'" % g)['SUB_GENE'])], cmap='RdGy', center=0)
plt.show()

kns = ['PLK1', 'PLK2', 'PLK3', 'PLK4']
kns_ov = pd.DataFrame({
    k1: {
        k2:
            len(set(knet.query("GENE == '%s'" % k1)['SUB_GENE']).intersection(knet.query("GENE == '%s'" % k2)['SUB_GENE']))
        for k2 in kns
    } for k1 in kns
})


# -
y = ccrispy.loc[kfam['kinase']].T
x = pd.get_dummies(samplesheet.loc[y.index, 'Cancer Type'])

x = x.loc[:, x.sum() > 3]

lm = LinearRegression().fit(x, y)
lm_coef = pd.DataFrame(lm.coef_, index=y.columns, columns=x.columns)


# -
sns.clustermap(lm_coef.T, cmap='RdGy', center=0)
plt.gcf().set_size_inches(15, 4)
plt.savefig('reports/kinases_essentiality_clustermap.png', bbox_inches='tight', dpi=600)
plt.close('all')


# -
plot_df = ccrispy.loc[kfam['kinase']].T.corr()
sns.clustermap(plot_df, cmap='RdGy', center=0)
plt.gcf().set_size_inches(15, 15)
plt.savefig('reports/kinases_essentiality_clustermap_corr.png', bbox_inches='tight', dpi=600)
plt.close('all')


# -
k_targets = knet.groupby('GENE')['SUB_GENE'].agg(lambda t: set(t)).to_dict()

kess = pd.DataFrame(
    {c:
         {k:
              ztest(ccrispy[c].loc[k_targets[k]], ccrispy[c].loc[genes.difference(k_targets[k])])[0]
          for k in k_targets}
     for c in ccrispy
     }
)

k_corr = kess[list(samples)].T.corrwith(ccrispy.loc[kess.index, samples].T)

k_corr.hist()
plt.show()


plot_df = pd.concat([k_corr.rename('corr'), cgenes.rename('role')], axis=1).dropna()
sns.boxplot('role', 'corr', data=plot_df, notch=True)
plt.show()


# -
samples = list(set(d_response).intersection(ccrispy))
print(len(samples))

d_response[samples].dropna(how='all').corrwith(kess[samples])


# -
kt_corr = pd.DataFrame([{
    'kinase': k, 'target': t,
    'corr': pearsonr(ccrispy.loc[k, samples], ccrispy.loc[t, samples])[0]
} for k in k_targets for t in k_targets[k]])
kt_corr = kt_corr.sort_values('corr')

sns.distplot(kt_corr['corr'])
plt.show()

sns.jointplot(ccrispy.loc['ZAP70'], ccrispy.loc['SH2B3'])
plt.show()
