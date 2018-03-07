#!/usr/bin/env python
# Copyright (C) 2018 Emanuel Goncalves

import textwrap
import numpy as np
import pandas as pd
import seaborn as sns
import scripts.drug as dc
import matplotlib.pyplot as plt
from crispy import bipal_dbgd
from scipy.stats import uniform
from matplotlib.colors import rgb2hex
from crispy.regression.linear import lr
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.multitest import multipletests
from scripts.drug.assemble_ppi import build_biogrid_ppi
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score


def lm_drug_crispr(xs, ys, ws):
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


def dist_drugtarget_genes(drug_targets, genes, ppi):
    genes = genes.intersection(set(ppi.vs['name']))
    assert len(genes) != 0, 'No genes overlapping with PPI provided'

    dmatrix = {}

    for drug in drug_targets:
        drug_genes = drug_targets[drug].intersection(genes)

        if len(drug_genes) != 0:
            dmatrix[drug] = dict(zip(*(genes, np.min(ppi.shortest_paths(source=drug_genes, target=genes), axis=0))))

    return dmatrix


def plot_volcano(lm_res_df, fdr_thres, beta_thres):
    plot_df = lm_res_df.copy()
    plot_df['signif'] = ['*' if f < fdr_thres and abs(b) > beta_thres else '-' for f, b in plot_df[['lr_fdr', 'beta']].values]
    plot_df['log10_qvalue'] = -np.log10(plot_df['lr_pval'])

    pal = dict(zip(*(['*', '-'], sns.light_palette(bipal_dbgd[0], 3, reverse=True).as_hex()[:-1])))

    # Scatter
    for i in ['*', '-']:
        plot_df_ = plot_df.query("(signif == '%s')" % i)

        plt.scatter(
            plot_df_['beta'], plot_df_['log10_qvalue'], edgecolor='white', lw=.1 if i == '*' else 0.05, s=3, c=pal[i],
            alpha=.5 if i == '*' else 0.3
        )

    # Add FDR threshold lines
    yy = plot_df.loc[(plot_df['lr_fdr'] - 0.05).abs().sort_values().index[0], 'lr_pval']

    plt.text(plt.xlim()[0] * .98, -np.log10(yy) * 1.01, 'FDR 5%', ha='left', color=pal['*'], alpha=0.65, fontsize=5)
    plt.axhline(-np.log10(yy), c=pal['*'], ls='--', lw=.1, alpha=.3)

    # Add beta lines
    plt.axvline(-beta_thres, c=pal['*'], ls='--', lw=.1, alpha=.3)
    plt.axvline(beta_thres, c=pal['*'], ls='--', lw=.1, alpha=.3)

    # Add axis lines
    plt.axvline(0, c=bipal_dbgd[0], lw=.1, ls='-', alpha=.3)

    # Add axis labels and title
    plt.title('Linear regression: Drug ~ CRISPR', fontsize=8, fontname='sans-serif')
    plt.xlabel('Model coefficient (beta)', fontsize=8, fontname='sans-serif')
    plt.ylabel('Log-ratio test p-value (-log10)', fontsize=8, fontname='sans-serif')


def plot_barplot_signif_assoc(lm_res_df, fdr_thres, beta_thres):
    plot_df = pd.DataFrame({
        'ypos': [0, 1],
        'type': ['positive', 'negative'],
        'count': [
            sum(lm_res_df.query('lr_fdr < {}'.format(fdr_thres))['beta'] > beta_thres),
            sum(lm_res_df.query('lr_fdr < {}'.format(fdr_thres))['beta'] < -beta_thres)
        ]
    })

    plt.barh(plot_df['ypos'], plot_df['count'], .8, color=bipal_dbgd[0], align='center')

    for x, y in plot_df[['ypos', 'count']].values:
        plt.text(y - .5, x, str(y), color='white', ha='right' if y != 1 else 'center', va='center', fontsize=10)

    plt.yticks(plot_df['ypos'])
    plt.yticks(plot_df['ypos'], plot_df['type'])

    plt.xlabel('# significant associations (FDR < 5%)')
    plt.title('Significant Drug ~ CRISPR associations (%d)' % plot_df['count'].sum())


def plot_target_cumsum(lm_res_df):
    ax = plt.gca()

    _, _, patches = ax.hist(
        lm_res_df.query('target != 0')['lr_pval'], lm_res_df.query('target != 0').shape[0],
        cumulative=True, normed=1, histtype='step', label='All', color=bipal_dbgd[0], lw=1.
    )
    patches[0].set_xy(patches[0].get_xy()[:-1])

    _, _, patches = ax.hist(
        lm_res_df.query('target == 0')['lr_pval'], lm_res_df.query('target == 0').shape[0],
        cumulative=True, normed=1, histtype='step', label='Target', color=bipal_dbgd[1], lw=1.
    )
    patches[0].set_xy(patches[0].get_xy()[:-1])

    _, _, patches = ax.hist(
        uniform.rvs(size=lm_res_df.shape[0]), lm_res_df.shape[0],
        cumulative=True, normed=1, histtype='step', label='Random', color='gray', lw=.3, ls='--'
    )
    patches[0].set_xy(patches[0].get_xy()[:-1])

    plt.xlim(0, 1)
    plt.ylim(0, 1)

    ax.set_xlabel('Drug ~ CRISPR p-value')
    ax.set_ylabel('Cumulative distribution')

    ax.legend(loc=4)

    return ax


def plot_target_boxplot(lm_res_df):
    # - PPI boxplot
    plot_df = lm_res_df.dropna()
    plot_df = plot_df.assign(thres=['Target' if i == 0 else ('%d' % i if i < 4 else '>=4') for i in plot_df['target']])

    order = ['Target', '1', '2', '3', '>=4']
    palette = sns.light_palette(bipal_dbgd[0], len(order), reverse=True)

    sns.boxplot('lr_pval', 'thres', data=plot_df, orient='h', linewidth=.3, notch=True, fliersize=1, palette=palette, order=order)

    plt.ylabel('PPI distance')
    plt.xlabel('Drug ~ Drug-target CRISPR p-value')


def plot_ppi_arocs(lm_res_df, fdr_thres, beta_thres):
    plot_df = lm_res_df[(lm_res_df['beta'].abs() > beta_thres) & (lm_res_df['lr_fdr'] < fdr_thres)].copy()
    plot_df = plot_df.assign(thres=['Target' if i == 0 else ('%d' % i if i < 4 else '>=4') for i in plot_df['target']])

    order = ['Target', '1', '2', '3', '>=4']
    order_color = [bipal_dbgd[1]] + list(map(rgb2hex, sns.light_palette(bipal_dbgd[0], len(order) - 1, reverse=True)))

    ax = plt.gca()
    for t, c in zip(*(order, order_color)):
        fpr, tpr, _ = roc_curve((plot_df['thres'] == t).astype(int), 1 - plot_df['lr_fdr'])
        ax.plot(fpr, tpr, label='%s=%.2f (AUC)' % (t, auc(fpr, tpr)), lw=1., c=c)

    ax.plot((0, 1), (0, 1), 'k--', lw=.3, alpha=.5)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel('False positive rate')
    ax.set_ylabel('True positive rate')
    ax.set_title('Protein-protein interactions\nDrug ~ CRISPR (FDR<{}%, |b|>{})'.format(fdr_thres * 100, beta_thres))

    ax.legend(loc=4, prop={'size': 8})


def plot_ppi_prc(lm_res_df, fdr_thres, beta_thres):
    plot_df = lm_res_df[(lm_res_df['beta'].abs() > beta_thres) & (lm_res_df['lr_fdr'] < fdr_thres)].copy()
    plot_df = plot_df.assign(thres=['Target' if i == 0 else ('%d' % i if i < 4 else '>=4') for i in plot_df['target']])

    order = ['Target', '1', '2', '3', '>=4']
    order_color = [bipal_dbgd[1]] + list(map(rgb2hex, sns.light_palette(bipal_dbgd[0], len(order) - 1, reverse=True)))

    ax = plt.gca()
    for t, c in zip(*(order, order_color)):
        y_true, y_pred = (plot_df['thres'] == t).astype(int), 1 - plot_df['lr_fdr']

        precision, recall, _ = precision_recall_curve(y_true, y_pred)
        ax.plot(precision, recall, label='%s=%.2f (AP)' % (t, average_precision_score(y_true, y_pred)), lw=1., c=c)

    ax.plot((0, 1), (0, 1), 'k--', lw=.3, alpha=.5)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title('Protein-protein interactions\nDrug ~ CRISPR (FDR<{}%, |b|>{})'.format(fdr_thres * 100, beta_thres))

    ax.legend(loc=1, prop={'size': 6})


def plot_corrplot(x, y):
    plot_df = pd.concat([x, y], axis=1)

    g = sns.jointplot(
        x.name, y.name, data=plot_df, kind='reg', space=0, color=bipal_dbgd[0], annot_kws=dict(stat='r'),
        marginal_kws={'kde': False}, joint_kws={'scatter_kws': {'edgecolor': 'w', 'lw': .3, 's': 12}, 'line_kws': {'lw': 1.}}
    )

    g.ax_joint.axhline(0, ls='-', lw=0.1, c=bipal_dbgd[0])
    g.ax_joint.axvline(0, ls='-', lw=0.1, c=bipal_dbgd[0])

    g.set_axis_labels('{} (log10 FC)'.format(x.name), '{} (ln IC50)'.format(y.name))


def plot_drug_associations_barplot(plot_df, order, ppi_text_offset=0.075, drug_name_offset=1., ylim_offset=1.1, fdr_line=0.05):
    # Group Drug ~ Gene associations
    df = plot_df.groupby(['DRUG_NAME', 'Gene']).first().reset_index().sort_values('lr_fdr')

    # Pick top 10 associations for each drug
    df = df.groupby('DRUG_NAME').head(10).set_index('DRUG_NAME')

    df__, xpos = [], 0
    for drug_name in order:
        df_ = df.loc[[drug_name]]
        df_ = df_.assign(y=-np.log10(df_['lr_fdr']))

        df_ = df_.assign(xpos=np.arange(xpos, xpos + df_.shape[0]))
        xpos += (df_.shape[0] + 2)

        df__.append(df_)

    df = pd.concat(df__).reset_index()

    # Significant line
    if fdr_line is not None:
        plt.axhline(-np.log10(0.05), ls='--', lw=.1, c=bipal_dbgd[0], alpha=.3, zorder=0)

    # Barplot
    plt.bar(df.query('target != 0')['xpos'], df.query('target != 0')['y'], .8, color=bipal_dbgd[0], align='center', zorder=5)
    plt.bar(df.query('target == 0')['xpos'], df.query('target == 0')['y'], .8, color=bipal_dbgd[1], align='center', zorder=5)

    # Distance to target text
    for x, y, t in df[['xpos', 'y', 'target']].values:
        l = '-' if np.isnan(t) or np.isposinf(t) else ('T' if t == 0 else str(int(t)))
        plt.text(x, y - (df['y'].max() * ppi_text_offset), l, color='white', ha='center', fontsize=6, zorder=10)

    plt.ylim(ymax=df['y'].max() * ylim_offset)

    # Name drugs
    for k, v in df.groupby('DRUG_NAME')['xpos'].mean().sort_values().to_dict().items():
        plt.text(v, df['y'].max() * drug_name_offset, textwrap.fill(k, 15), ha='center', fontsize=6, zorder=10)

    plt.xticks(df['xpos'], df['Gene'], rotation=90, fontsize=5)
    plt.ylabel('Log-ratio FDR (-log10)')
    plt.title('Top significant Drug ~ CRISPR associations')


def drug_interaction_signed_ppi(ppi, drug_targets):
    d_interactions = {}

    for drug in drug_targets:
        d_interactions[drug] = {}

        for target in d_targets[drug].intersection(ppi.vs['name']):

            for edge in ppi.es[ppi.incident(target, mode='all')]:

                for edge_id, edge_name in zip(*([edge.source, edge.target], ppi.vs[[edge.source, edge.target]]['name'])):

                    if edge_name != target:
                        d_interactions[drug][edge_name] = ppi.es[edge_id]['interaction']

    return d_interactions


def plot_count_associations(lm_res_df, fdr_thres, beta_thres, min_nevents=5):
    df = lm_res_df[(lm_res_df['beta'].abs() > beta_thres) & (lm_res_df['lr_fdr'] < fdr_thres)].copy()

    df = df.groupby('DRUG_NAME')['lr_fdr'].count().sort_values(ascending=False).reset_index()

    df.columns = ['drug', 'counts']

    if min_nevents is not None:
        df = df[df['counts'] >= min_nevents]

    ax = sns.barplot('drug', 'counts', data=df, color=bipal_dbgd[0], linewidth=.8)

    for item in ax.get_xticklabels():
        item.set_rotation(90)

    plt.xlabel('')
    plt.ylabel('# associations')
    plt.title('Drug number of significant associations (FDR<{}%, |b|>{})'.format(fdr_thres * 100, beta_thres))


if __name__ == '__main__':
    # - Imports
    # Drug target
    d_targets = dc.drug_targets()

    # Samplesheet
    ss = pd.read_csv(dc.SAMPLESHEET_FILE, index_col=0).dropna(subset=['Cancer Type'])

    # Growth rate
    growth = pd.read_csv(dc.GROWTHRATE_FILE, index_col=0)

    # CRISPR gene-level corrected fold-changes
    crispr = pd.read_csv(dc.CRISPR_GENE_FC_CORRECTED, index_col=0, sep='\t').dropna()
    crispr_scaled = dc.scale_crispr(crispr)

    # Drug response
    d_response = pd.read_csv(dc.DRUG_RESPONSE_FILE, index_col=[0, 1, 2], header=[0, 1])
    d_response.columns = d_response.columns.droplevel(0)

    # - Overlap
    samples = list(set(d_response).intersection(crispr).intersection(ss.index).intersection(growth.index))
    d_response, crispr = d_response[samples], crispr[samples]
    print('Samples: %d' % len(samples))

    # - Transform and filter
    d_response = dc.filter_drug_response(d_response)
    crispr_scaled = dc.filter_crispr(crispr_scaled, value_thres=.75, value_nevents=5)

    # - Check list
    dc.check_in_list(crispr_scaled.index)

    # - Covariates
    covariates = pd.concat([
        pd.get_dummies(ss[['Cancer Type']]),
        growth['growth_rate_median']
    ], axis=1).loc[samples]
    covariates = covariates.loc[:, covariates.sum() != 0]

    # - Linear regression: drug ~ crispr + tissue
    lm_res_df = lm_drug_crispr(crispr_scaled[samples].T, d_response[samples].T, covariates.loc[samples])

    # - PPI annotation
    ppi = build_biogrid_ppi(int_type={'physical'})

    # Calculate distance between drugs and CRISPR genes in PPI
    d_drug_genes = dist_drugtarget_genes(d_targets, set(lm_res_df['Gene']), ppi)

    # d, g = 1047, 'TP53'
    lm_res_df = lm_res_df.assign(
        target=[
            d_drug_genes[d][g] if d in d_drug_genes and g in d_drug_genes[d] else np.nan for d, g in lm_res_df[['DRUG_ID', 'Gene']].values
        ]
    )

    lm_res_df.sort_values('lr_fdr').to_csv('data/drug/lm_drug_crispr.csv', index=False)
    print(lm_res_df[(lm_res_df['beta'].abs() > .5) & (lm_res_df['lr_fdr'] < 0.05)].sort_values('lr_fdr'))
    # lm_res_df = pd.read_csv('data/drug/lm_drug_crispr.csv')

    # - Plot
    fdr_thres, beta_thres = 0.05, 0.5

    # Volcano
    plot_volcano(lm_res_df, fdr_thres, beta_thres)

    plt.gcf().set_size_inches(2., 4.)
    plt.savefig('reports/drug/lm_crispr_volcano.png', bbox_inches='tight', dpi=1200)
    plt.close('all')

    # Barplot count number of significant associations
    plot_barplot_signif_assoc(lm_res_df, fdr_thres, beta_thres)

    plt.gcf().set_size_inches(3, 1.5)
    plt.savefig('reports/drug/signif_assoc_count.png', bbox_inches='tight', dpi=600)
    plt.close('all')

    # Targets cumulative distribution
    plot_target_cumsum(lm_res_df)

    plt.gcf().set_size_inches(2.5, 2)
    plt.savefig('reports/drug/drug_target_cum_dist.png', bbox_inches='tight', dpi=600)
    plt.close('all')

    plot_target_boxplot(lm_res_df)

    plt.title('Drug-targets in protein-protein networks')
    plt.gcf().set_size_inches(3, 1.5)
    plt.savefig('reports/drug/ppi_boxplot.png', bbox_inches='tight', dpi=600)
    plt.close('all')

    # PPI AROCs significant associations
    plot_ppi_arocs(lm_res_df, fdr_thres, beta_thres)

    plt.gcf().set_size_inches(3, 3)
    plt.savefig('reports/drug/ppi_signif_roc.png', bbox_inches='tight', dpi=600)
    plt.close('all')

    # PPI AUPRs significant associations
    plot_ppi_prc(lm_res_df, fdr_thres, beta_thres)

    plt.gcf().set_size_inches(3, 3)
    plt.savefig('reports/drug/ppi_signif_prc.png', bbox_inches='tight', dpi=600)
    plt.close('all')

    # Drug associations barplot
    plot_df = lm_res_df[(lm_res_df['beta'].abs() > beta_thres) & (lm_res_df['lr_fdr'] < fdr_thres)]

    drug_list = list(plot_df.groupby('DRUG_NAME')['lr_fdr'].min().sort_values().index)[:10]

    plot_drug_associations_barplot(plot_df, drug_list)

    plt.gcf().set_size_inches(8, 1.5)
    plt.savefig('reports/drug/drug_associations_barplot.png', bbox_inches='tight', dpi=600)
    plt.close('all')

    # Number of significant associations per drug
    plot_count_associations(lm_res_df, fdr_thres, beta_thres)

    plt.gcf().set_size_inches(7, 3)
    plt.savefig('reports/drug/drug_associations_count.png', bbox_inches='tight', dpi=600)
    plt.close('all')

    # Plot Drug ~ CRISPR corrplot
    idx = 1857
    d_id, d_name, d_screen, gene = lm_res_df.loc[idx, ['DRUG_ID', 'DRUG_NAME', 'VERSION', 'Gene']].values
    # d_id, d_name, d_screen, gene = 1510, 'Linsitinib', 'RS', 'FURIN'

    plot_corrplot(
        crispr_scaled.loc[gene].rename('{} CRISPR'.format(gene)), d_response.loc[(d_id, d_name, d_screen)].rename('{} {} Drug'.format(d_name, d_screen))
    )

    plt.gcf().set_size_inches(2., 2.)
    plt.savefig('reports/drug/crispr_drug_corrplot.png', bbox_inches='tight', dpi=600)
    plt.close('all')
