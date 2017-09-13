#!/usr/bin/env python
# Copyright (C) 2017 Emanuel Goncalves

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from matplotlib.gridspec import GridSpec
from crispy.data_correction import DataCorrectionPosition
from sklearn.model_selection import StratifiedShuffleSplit


# - Imports
# sgRNA library
sgrna_lib = pd.read_csv('data/gdsc/crispr/KY_Library_v1.1_annotated.csv', index_col=0).dropna(subset=['STARTpos'])

# CRISPR fold-changes
fc_sgrna = pd.read_csv('data/gdsc/crispr/crispy_gdsc_fold_change_sgrna.csv', index_col=0)

# - Biases correction method
sample = 'EBC-1'

df = pd.concat([fc_sgrna[sample].rename('logfc'), sgrna_lib], axis=1).dropna(subset=['logfc', 'GENES'])
df = df.groupby('GENES').agg({
    'CODE': 'first',
    'CHRM': 'first',
    'STARTpos': np.mean,
    'ENDpos': np.mean,
    'logfc': np.mean
}).sort_values('CHRM').dropna()

# Sampling
parms = []
chromosomes = {'7', '8', '16'}
for alpha_lb in [1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10]:

    for length_scale_lb in [1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1]:

        # Cross-validation
        cv = StratifiedShuffleSplit(n_splits=10, test_size=.3)

        for idx_train, idx_test in cv.split(df, df['CHRM']):
            print()

            # Correction
            gp = DataCorrectionPosition(
                X=df.iloc[idx_train][['STARTpos']], Y=df.iloc[idx_train][['logfc']],
                X_test=df.iloc[idx_test][['STARTpos']], Y_test=df.iloc[idx_test][['logfc']], chrm_test=df.iloc[idx_test]['CHRM']
            )

            gp = gp.correct(
                df.iloc[idx_train]['CHRM'],
                chromosomes=chromosomes,
                alpha=(alpha_lb, alpha_lb),
                length_scale=(length_scale_lb, length_scale_lb),
                return_hypers=True
            )

            # Evalute
            res = {}
            for c in chromosomes:
                y_true, y_pred = zip(*(gp.query("(chrm == '%s') & (osp)" % c)[['original', 'mean']].values))
                res[c] = r2_score(y_true, y_pred)

            res['a_lb'] = alpha_lb
            res['ls_lb'] = length_scale_lb
            print('; '.join(['%s: %.2e' % (i, res[i]) for i in res]))

            # Store
            parms.append(res)

parms = pd.DataFrame(parms).assign(sample=sample)
parms.to_csv('data/out_of_sample_hyper_%s.csv' % sample, index=False)
# parms = pd.read_csv('data/out_of_sample_hyper_%s.csv' % sample)

# -
gs, pos = GridSpec(1, 3, hspace=.6, wspace=.3), 0
for c in chromosomes:
    ax = plt.subplot(gs[pos])

    plot_df = pd.pivot_table(parms, index='a_lb', columns='ls_lb', values=c, aggfunc=np.mean)

    g = sns.heatmap(plot_df, cmap='viridis', annot=True, fmt='.2f', ax=ax)

    ax.set_title('Out-of-sample prediction R2\n(Sample: %s; Chrm: %s)' % (sample, c))
    plt.setp(g.get_yticklabels(), rotation=0)

    pos += 1

plt.gcf().set_size_inches(16, 4)
plt.savefig('reports/gp_subsampling_heatmap.png', bbox_inches='tight', dpi=600)
plt.close('all')
