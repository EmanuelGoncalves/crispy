#!/usr/bin/env python
# Copyright (C) 2018 Emanuel Goncalves

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from crispy import bipal_dbgd
from scipy.stats import pearsonr
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression, Ridge

# - Imports
# Samplesheet
ss = pd.read_csv('data/gdsc/samplesheet.csv', index_col=0).dropna(subset=['Cancer Type', 'Microsatellite'])

# CRISPR
crispr = pd.read_csv('data/gdsc/crispr/crisprcleanr_gene.csv', index_col=0).dropna()
crispr = crispr.subtract(crispr.mean()).divide(crispr.std())

# Drug response
d_response = pd.read_csv('data/gdsc/drug_single/drug_ic50_merged_matrix.csv', index_col=[0, 1, 2], header=[0, 1])
d_response.columns = d_response.columns.droplevel(0)

# Growth rate
growth = pd.read_csv('data/gdsc/growth_rate.csv', index_col=0)


# - Overlap
samples = list(set(d_response).intersection(ss.index).intersection(crispr).intersection(growth.index))
print(len(samples))


# - Filter
d_response = d_response[samples]
d_response = d_response[d_response.count(1) > d_response.shape[1] * .85]
d_response = d_response.fillna(d_response.mean())


# - PCA
# Drug
d_pca = PCA(n_components=10).fit(d_response[samples].T)
d_pcs = pd.DataFrame(d_pca.transform(d_response[samples].T), index=samples, columns=map(str, range(10)))
print(d_pca.explained_variance_ratio_)

# CRISPR
c_pca = PCA(n_components=10).fit(crispr[samples].T)
c_pcs = pd.DataFrame(c_pca.transform(crispr[samples].T), index=samples, columns=map(str, range(10)))
print(c_pca.explained_variance_ratio_)

# Plot variance explained
plot_df = pd.DataFrame({'drug': d_pca.explained_variance_ratio_, 'crispr': c_pca.explained_variance_ratio_})
plot_df = plot_df.assign(pc=list(map(lambda x: 'PC%d' % (x + 1), plot_df.index)))

for n, t in [('drug', 'Drug response'), ('crispr', 'CRISPR')]:
    plot_df = plot_df.sort_values(n)
    ypos = list(map(lambda x: x[0], enumerate(plot_df.index)))

    plt.barh(ypos, plot_df[n], .8, color=bipal_dbgd[0], align='center')

    plt.yticks(ypos)
    plt.yticks(ypos, plot_df['pc'])

    plt.xlabel('Explained variance')
    plt.ylabel('Principal component')
    plt.title('%s PCA' % t)

    plt.gcf().set_size_inches(2, 3)
    plt.savefig('reports/drug/%s_pca_varexp.png' % n, bbox_inches='tight', dpi=600)
    plt.close('all')


# - Growth
plot_df = pd.DataFrame({
    'pc': d_pcs.loc[samples, '0'], 'growth': growth.loc[samples, 'NC1_ratio_mean']
})

g = sns.jointplot('pc', 'growth', data=plot_df, space=0, color=bipal_dbgd[0], joint_kws={'edgecolor': 'white', 'alpha': .8})

g.set_axis_labels('Drug response PC 1 (%.1f%%)' % (d_pca.explained_variance_ratio_[0] * 100), 'Growth rate')

plt.gcf().set_size_inches(3, 3)
plt.savefig('reports/drug/drug_pca_growth.png', bbox_inches='tight', dpi=600)
plt.close('all')


# - Drug response correlation with growth
dg_cor = d_response[samples].T.corrwith(growth.loc[samples, 'NC1_ratio_mean']).sort_values().rename('cor').reset_index()

# Histogram
sns.distplot(dg_cor.query("VERSION == 'RS'")['cor'], label='RS', color=bipal_dbgd[0], bins=10, kde=False, kde_kws={'lw': 1.})
sns.distplot(dg_cor.query("VERSION == 'v17'")['cor'], label='v17', color=bipal_dbgd[1], bins=10, kde=False, kde_kws={'lw': 1.})

plt.title('Drug-response correlation with growth rate')
plt.xlabel("Pearson's r")
plt.ylabel('Count')
plt.legend(loc='upper left')

plt.gcf().set_size_inches(3, 2)
plt.savefig('reports/drug/drug_growth_corr_hist.png', bbox_inches='tight', dpi=600)
plt.close('all')

# Plot correlation
drug_id, drug_name, drug_screen = dg_cor.loc[1, ['DRUG_ID', 'DRUG_NAME', 'VERSION']]

plot_df = pd.DataFrame({
    'drug': d_response.loc[(drug_id, drug_name, drug_screen), samples], 'growth': growth.loc[samples, 'NC1_ratio_mean']
})

g = sns.jointplot('drug', 'growth', data=plot_df, space=0, color=bipal_dbgd[0], joint_kws={'edgecolor': 'white', 'alpha': .8})
g.set_axis_labels('Drug response %s (%s)' % (drug_name, drug_screen), 'Growth rate')
plt.gcf().set_size_inches(3, 3)
plt.savefig('reports/drug/drug_growth_jointplot.png', bbox_inches='tight', dpi=600)
plt.close('all')
