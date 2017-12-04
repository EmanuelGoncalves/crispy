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

# - Imports
# Cancer genes
cgenes = pd.read_csv('data/resources/cancer_genes.txt', sep='\t')
cgenes = cgenes[cgenes['role'].isin(['Loss of function', 'Activating'])]
cgenes = cgenes.groupby('SYM')['role'].first()

# Uniprot
uniprot = pd.Series(pickle.load(open('data/resources/uniprot_to_genename.pickle', 'rb')))

# Omnipath ptms
ptms = pd.read_csv('data/resources/omnipath_ptms.txt', sep='\t')
ptms = ptms[ptms['modification'].isin(['dephosphorylation', 'phosphorylation'])]
ptms_set = {(s, t) for s, t in ptms[['enzyme', 'substrate']].values}

# Omnipath
omnipath = pd.read_csv('data/resources/omnipath_interactions.txt', sep='\t')
omnipath = omnipath.query('is_directed == 1')
omnipath = omnipath[omnipath['is_stimulation'] != omnipath['is_inhibition']]
omnipath = omnipath[[(s, t) in ptms_set for s, t in omnipath[['source', 'target']].values]]

omnipath = omnipath.assign(source_g=uniprot[omnipath['source']].values)
omnipath = omnipath.assign(target_g=uniprot[omnipath['target']].values)

#
ccrispy = pd.read_csv('data/crispr_gdsc_crispy.csv', index_col=0)

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
omnipath = omnipath[[s in ccrispy.index and t in ccrispy.index for s, t in omnipath[['source_g', 'target_g']].values]]
omnipath = omnipath.assign(corr=[pearsonr(ccrispy.loc[s], ccrispy.loc[t])[0] for s, t in omnipath[['source_g', 'target_g']].values])


sns.boxplot('is_inhibition', 'corr', data=omnipath)
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

sns.jointplot(ccrispy.loc[g], ccrispy.loc['ACTR2'])

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

sns.jointplot(ccrispy.loc['MTOR'], ccrispy.loc['RPTOR'])
plt.show()
