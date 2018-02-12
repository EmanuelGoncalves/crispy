#!/usr/bin/env python
# Copyright (C) 2017 Emanuel Goncalves

import pandas as pd
import seaborn as sns
import scripts.drug as dc
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from crispy.regression.linear import lr
from matplotlib.gridspec import GridSpec
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.multitest import multipletests
from sklearn.metrics.regression import explained_variance_score


def fun(x, a, b, c):
    return a + b * x + c * (x ** 2)


def plot_nfit(lmm_df, n=10):
    sns.set(style='ticks', context='paper', font_scale=0.75, rc={'axes.linewidth': .3, 'xtick.major.width': .3, 'ytick.major.width': .3, 'xtick.major.size': 2.5, 'ytick.major.size': 2.5, 'xtick.direction': 'in', 'ytick.direction': 'in'})
    gs, pos = GridSpec(10, 5, hspace=.5, wspace=.5), 0
    # gx, gy = 'FAM50B', 'FAM50A'
    for beta, gx, gy, pval, qval in lmm_df.head(n).values:
        ax = plt.subplot(gs[pos])

        # Get measurements
        x, y = gexp.loc[gx, samples], crispr.loc[gy, samples]

        # Fit curve
        popt, pcov = curve_fit(fun, x, y, method='trf')
        y_ = fun(x, *popt)

        # Variance explained
        vexp = explained_variance_score(y, y_)

        # Plot
        plot_df = pd.DataFrame({
            'x': x, 'y': y, 'y_': y_
        }).sort_values('x')

        ax.scatter(plot_df['x'], plot_df['y'], c='#37454B', alpha=.8, edgecolors='white', linewidths=.3, s=20, label='measurements')
        ax.plot(plot_df['x'], plot_df['y_'], ls='--', c='#F2C500', label='fit')

        ax.axhline(0, lw=.3, ls='-', color='gray', alpha=.5)

        ax.set_xlabel('%s gene-expression' % gx)
        ax.set_ylabel('%s CRISPR' % gy)
        # ax.set_title('y = {:.2f} + {:.2f} * x + {:.2f} * x**2'.format(*popt))
        ax.set_title('B=%.2f; FDR=%.2e' % (beta, qval))

        pos += 1


def lm_crispr_gexp(xs, ys, ws):
    print('CRISPR genes: %d, Drug: %d' % (len(set(xs.columns)), len(set(ys.columns))))

    # Standardize xs
    xs = pd.DataFrame(StandardScaler().fit_transform(xs), index=xs.index, columns=xs.columns)

    # Regression
    res = lr(xs, ys, ws)

    # - Export results
    res_df = pd.concat([res[i].unstack().rename(i) for i in res], axis=1).reset_index()

    res_df = res_df.assign(f_fdr=multipletests(res_df['f_pval'], method='fdr_bh')[1])
    res_df = res_df.assign(lr_fdr=multipletests(res_df['lr_pval'], method='fdr_bh')[1])

    return res_df


if __name__ == '__main__':
    # - Imports
    # CRISPR gene-level corrected fold-changes
    crispr = pd.read_csv('data/gdsc/crispr/corrected_logFCs.tsv', index_col=0, sep='\t').dropna()
    crispr_scaled = dc.scale_crispr(crispr)

    # Gene-expression
    gexp = pd.read_csv('data/gdsc/gene_expression/merged_voom_preprocessed.csv', index_col=0)
    gexp = gexp.subtract(gexp.mean()).divide(gexp.std())

    # Growth rate
    growth = pd.read_csv(dc.GROWTHRATE_FILE, index_col=0)

    # Samplesheet
    ss = pd.read_csv('data/gdsc/samplesheet.csv', index_col=0).dropna(subset=['Cancer Type'])

    # - Overlap + Filter
    # Filter
    samples = list(set(crispr_scaled).intersection(gexp).intersection(ss.index))
    crispr_scaled, gexp = crispr_scaled[samples], gexp[samples]

    # CRISPR
    crispr_scaled = dc.filter_crispr(crispr_scaled, value_thres=.75, value_nevents=5)

    # Gexp
    gexp = gexp[(gexp < -1).sum(1) < (gexp.shape[1] * 0.5)]
    gexp = gexp[(gexp.abs() >= 1.5).sum(1) >= 5]
    print(crispr_scaled.shape, gexp.shape)

    # - Covariates
    covariates = pd.concat([
        pd.get_dummies(ss[['Cancer Type']]),
        growth['growth_rate_median']
    ], axis=1).loc[samples]
    covariates = covariates.loc[:, covariates.sum() != 0]

    # - Linear regression: gexp ~ crispr + tissue
    lm_res_df = lm_crispr_gexp(crispr_scaled[samples].T, gexp[samples].T, covariates.loc[samples])
    print(lm_res_df.query('lr_fdr < 0.05'))

    # - Export
    lm_res_df.query('lr_fdr < 0.05').to_csv('data/sl/gexp_crispr_lr.csv')

    # - Plot
    plot_nfit(lm_res_df, n=50)
    plt.gcf().set_size_inches(2.5 * 5, 2.5 * 10)
    plt.savefig('reports/sl/nlfit_scatter.png', bbox_inches='tight', dpi=600)
    plt.close('all')

