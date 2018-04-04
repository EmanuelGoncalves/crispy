#!/usr/bin/env python
# Copyright (C) 2018 Emanuel Goncalves

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

if __name__ == '__main__':

    # - Import ploidy and purity
    ascat_p_p = pd.read_csv('data/cosmic/ascat_acf_ploidy.tsv', sep='\t').dropna()
    ascat_p_p = ascat_p_p[ascat_p_p['Sample'].apply(lambda v: v.startswith('TCGA'))]
    ascat_p_p = ascat_p_p.assign(Sample=[i.replace('.', '-')[:15] for i in ascat_p_p['Sample']])
    ascat_p_p = ascat_p_p.drop_duplicates(['Sample'], keep=False).set_index('Sample')

    # - Import gene-expression
    gexp = pd.read_csv('data/tcga/GSM1536837_06_01_15_TCGA_24.tumor_Rsubread_FPKM.txt', sep='\t', index_col=0)
    gexp.columns = [v[:15] for v in gexp]
    gexp = gexp.loc[:, ~gexp.columns.duplicated(False)]
    gexp = gexp.loc[:, [i in ascat_p_p.index for i in gexp]]
    gexp = (gexp < 1).astype(int)

    # - Import copy-number
    cn = pd.read_csv('data/cosmic/CosmicCompleteCNA.tsv', sep='\t').dropna(subset=['TOTAL_CN'])
    cn = cn[cn['SAMPLE_NAME'].isin(gexp.columns)]
    cn = cn[cn['gene_name'].isin(gexp.index)]
    cn = cn.assign(ploidy=ascat_p_p.loc[cn['SAMPLE_NAME'], 'Ploidy'].values)
    cn = cn.assign(ratio=cn.eval('TOTAL_CN / ploidy')).sort_values('ratio')
    cn = cn.assign(td=cn['ratio'].apply(lambda v: int(v > 3)).values)
    cn = cn.assign(nexp=[gexp.loc[g, s] for g, s in cn[['gene_name', 'SAMPLE_NAME']].values])
    cn = cn.assign(cess=(cn['td'] & cn['nexp']).values)

    # - Collateral Essentialities
    coless = pd.read_csv('data/crispy_df_collateral_essentialities.csv')

    # -
    df = (cn.groupby(['SAMPLE_NAME', 'Primary site'])['cess'].sum() > 0).astype(int).reset_index()
    (df.groupby('Primary site')['cess'].sum() / df.groupby('Primary site')['cess'].count()).sort_values()

    cn.groupby(['SAMPLE_NAME', 'Primary site'])['cess'].sum().reset_index().groupby('Primary site')['cess'].median().sort_values()

    sum(cn.groupby(['SAMPLE_NAME', 'Primary site'])['cess'].sum() > 0) / len(set(cn['SAMPLE_NAME']))

    # -
    cn[cn['gene_name'] == 'GSDMA'][['Primary site', 'TOTAL_CN', 'ploidy', 'ratio', 'td', 'nexp', 'cess']].sort_values(['cess', 'TOTAL_CN'], ascending=False).head(60)
