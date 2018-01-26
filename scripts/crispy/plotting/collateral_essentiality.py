#!/usr/bin/env python
# Copyright (C) 2018 Emanuel Goncalves

import pickle
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from crispy import bipal_dbgd
from natsort import natsorted
from crispy.regression.linear import lr
from statsmodels.stats.multitest import multipletests


def main():
    # - Import
    # (non)essential genes
    essential = pd.read_csv('data/gene_sets/curated_BAGEL_essential.csv', sep='\t')['gene'].rename('essential')
    nessential = pd.read_csv('data/gene_sets/curated_BAGEL_nonEssential.csv', sep='\t')['gene'].rename('non-essential')

    # CRISPR library
    lib = pd.read_csv('data/crispr_libs/KY_Library_v1.1_updated.csv', index_col=0).groupby('gene')['chr'].first()

    # Samplesheet
    ss = pd.read_csv('data/gdsc/samplesheet.csv', index_col=0)

    # Non-expressed genes
    nexp = pickle.load(open('data/gdsc/nexp_pickle.pickle', 'rb'))
    nexp = pd.DataFrame({c: {g: 1 for g in nexp[c]} for c in nexp}).fillna(0)

    # Copy-number
    cnv = pd.read_csv('data/crispy_copy_number_gene_snp.csv', index_col=0)

    # Ploidy
    ploidy = pd.read_csv('data/crispy_copy_number_ploidy_snp.csv', index_col=0, names=['sample', 'ploidy'])['ploidy']

    # Chromosome copies
    chrm = pd.read_csv('data/crispy_copy_number_chr_snp.csv', index_col=0)

    # Copy-number ratios
    cnv_ratios = pd.read_csv('data/crispy_copy_number_gene_ratio_snp.csv', index_col=0)

    # CRISPR
    c_gdsc_fc = pd.read_csv('data/crispr_gdsc_logfc.csv', index_col=0)
    c_gdsc_crispy = pd.read_csv('data/crispr_gdsc_logfc_corrected.csv', index_col=0)

    # Cancer gene census list
    cgenes = pd.read_csv('data/gene_sets/Census_allThu Dec 21 15_43_09 2017.tsv', sep='\t')

    # - Overlap samples
    samples = list(set(cnv).intersection(c_gdsc_fc).intersection(c_gdsc_crispy).intersection(nexp))
    print('[INFO] Samples overlap: {}'.format(len(samples)))

    # - Recurrent amplified genes
    g_amp = pd.DataFrame({s: (cnv[s] > 8 if ploidy[s] > 2.7 else cnv[s] > 4).astype(int) for s in samples})
    g_amp = g_amp.loc[g_amp.index.isin(cnv_ratios.index) & g_amp.index.isin(cgenes['Gene Symbol'])]
    g_amp = g_amp[g_amp.sum(1) >= 3]

    # - Recurrent non-expressed genes
    g_nexp = nexp.loc[c_gdsc_fc.index, samples].dropna()
    g_nexp = g_nexp[g_nexp.sum(1) >= (g_nexp.shape[1] * .5)]

    # - Linear regressions
    # Build matrices
    xs = cnv_ratios.loc[g_amp.index, samples].T
    xs = xs.loc[:, (xs > 2).sum() >= 3]
    xs = xs.replace(np.nan, 1)

    ys = c_gdsc_fc.loc[g_nexp.index, xs.index]
    ys = ys.subtract(ys.mean()).divide(ys.std())
    ys = ys[(ys.abs() >= 1.5).sum(1) >= 5].T

    # Run fit
    lm = lr(xs, ys)

    # Export lm results
    lm_res = pd.concat([lm[i].unstack().rename(i) for i in lm], axis=1)
    lm_res = lm_res.reset_index().rename(columns={'gene': 'gene_crispr', 'level_1': 'gene_ratios'})

    lm_res = lm_res[lm_res['gene_crispr'] != lm_res['gene_ratios']]

    ratio_nexp_ov = (cnv_ratios.loc[xs.columns, samples] > 1.5).astype(int).dot((nexp.loc[ys.columns, samples] == 0).astype(int).T).unstack()
    lm_res = lm_res.loc[(ratio_nexp_ov.loc[[(y, x) for y, x in lm_res[['gene_crispr', 'gene_ratios']].values]] == 0).values]

    lm_res = lm_res.assign(f_fdr=multipletests(lm_res['f_pval'], method='fdr_bh')[1]).sort_values('f_fdr')

    lm_res.query('f_fdr < 0.05').to_csv('data/crispy_df_collateral_essentialities.csv', index=False)
    print(lm_res)

    # - Plot: Collateral essentiality
    idx = 24518
    y_feat, x_feat = lm_res.loc[idx, 'gene_crispr'], lm_res.loc[idx, 'gene_ratios']

    plot_df = pd.concat([
        c_gdsc_fc.loc[y_feat, samples].rename('crispy'),
        cnv.loc[x_feat, samples].rename('cnv'),
        cnv_ratios.loc[x_feat, samples].rename('ratio'),
        g_amp.loc[x_feat].rename('amp'),
        nexp.loc[y_feat].replace({1.0: 'No', 0.0: 'Yes'}).rename('expressed'),
        ss['Cancer Type'],
    ], axis=1).dropna().sort_values('crispy')
    plot_df = plot_df.assign(ratio_bin=[str(i) if i < 4 else '4+' for i in plot_df['ratio'].round(0).astype(int)])

    order = natsorted(set(plot_df['ratio_bin']))
    pal = {'No': bipal_dbgd[1], 'Yes': bipal_dbgd[0]}

    # Boxplot
    sns.boxplot('ratio_bin', 'crispy', data=plot_df, order=order, color=bipal_dbgd[0], sym='', linewidth=.3, saturation=.1)
    sns.stripplot('ratio_bin', 'crispy', 'expressed', data=plot_df, order=order, palette=pal, edgecolor='white', linewidth=.1, size=2, jitter=.1, alpha=.8)

    plt.axhline(0, ls='-', lw=.1, c=bipal_dbgd[0])

    plt.ylabel('%s\nCRISPR/Cas9 cop-number bias' % y_feat)
    plt.xlabel('%s\nCopy-number ratio' % x_feat)
    plt.title('Collateral essentiality\nbeta={0:.1f}; q-value={0:1.2e}'.format(lm_res.loc[idx, 'beta'], lm_res.loc[idx, 'f_fdr']))

    legend = plt.legend(title='Expressed', prop={'size': 8}, loc=3)
    legend.get_title().set_fontsize('8')

    plt.gcf().set_size_inches(2, 3)
    plt.savefig('reports/crispy/collateral_essentiality_boxplot.png', bbox_inches='tight', dpi=600)
    plt.close('all')


if __name__ == '__main__':
    main()
