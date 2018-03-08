#!/usr/bin/env python
# Copyright (C) 2018 Emanuel Goncalves

import pickle
import numpy as np
import pandas as pd
import itertools as it
from crispy.biases_correction import CRISPRCorrection
from scripts.crispy.plotting.annotate_bedpe import annotate_brass_bedpe


SVTYPES = ['inversion', 'tandem-duplication', 'deletion']


def count_svs(svs):
    counts = dict(zip(*(SVTYPES, np.repeat(0, len(SVTYPES)))))
    counts.update(dict(zip(*(np.unique(svs.split(','), return_counts=True)))))
    return counts


def present_svs(svs):
    svs_set = set(svs.split(','))
    return {sv.replace('-', ''): int(sv in svs_set) for sv in SVTYPES}


def combinations_x(df, n, gate):
    combs = [' {} '.format(gate).join(i) for i in it.permutations(list(df), n)]
    return pd.DataFrame({i: df.eval(i) for i in combs})


if __name__ == '__main__':
    bedpe_dir = 'data/gdsc/wgs/brass_bedpe'
    brca_samples = ['HCC1954', 'HCC1143', 'HCC38', 'HCC1187', 'HCC1937', 'HCC1395']

    # Annotate BRASS bedpes
    bed_dfs = annotate_brass_bedpe(bedpe_dir, bkdist=-1, splitreads=False, samples=brca_samples)

    # - CRISPR
    c_gdsc_fc = pd.read_csv('data/crispr_gdsc_logfc.csv', index_col=0).dropna()

    # - CRISPR lib
    sgrna_lib = pd.read_csv('data/crispr_libs/KY_Library_v1.1_updated.csv', index_col=0)

    # - Copy-number
    cnv = pd.read_csv('data/crispy_copy_number_gene_wgs.csv', index_col=0)

    # - Non-expressed genes
    nexp = pickle.load(open('data/gdsc/nexp_pickle.pickle', 'rb'))

    # - Overlap
    genes = set(c_gdsc_fc.index).intersection(cnv.index)

    # -
    # sample = 'HCC1143'
    crispy_samples = {}
    for sample in brca_samples:
        print('[INFO] {}'.format(sample))

        svmatrix = pd.DataFrame(bed_dfs[bed_dfs['sample'] == sample].set_index('feature')['collapse'].apply(present_svs).to_dict()).T

        svmatrix = svmatrix.reindex(genes).replace(np.nan, 0).astype(int)

        svmatrix = pd.concat([svmatrix, combinations_x(svmatrix, 3, 'and')], axis=1)

        svmatrix = svmatrix.assign(chr=sgrna_lib.groupby('gene')['chr'].first().reindex(svmatrix.index).values)

        svmatrix = svmatrix.assign(crispr=c_gdsc_fc[sample].reindex(svmatrix.index).values)

        svmatrix = svmatrix.assign(cnv=cnv[sample].reindex(svmatrix.index).values)

        svmatrix = svmatrix.dropna()

        # Correction only with copy-number
        crispy_cnv_only = CRISPRCorrection().rename(sample).fit_by(
            by=svmatrix['chr'], X=svmatrix[['cnv']], y=svmatrix['crispr']
        )
        crispy_cnv_only = pd.concat([v.to_dataframe() for k, v in crispy_cnv_only.items()])
        crispy_cnv_only.to_csv('data/crispy/gdsc_brass/{}.copynumber.csv'.format(sample))

        # Correction only with SV features
        crispy_svs_only = CRISPRCorrection().rename(sample).fit_by(
            by=svmatrix['chr'], X=svmatrix.drop(['chr', 'crispr', 'cnv'], axis=1), y=svmatrix['crispr']
        )
        crispy_svs_only = pd.concat([v.to_dataframe() for k, v in crispy_svs_only.items()])
        crispy_svs_only.to_csv('data/crispy/gdsc_brass/{}.svs.csv'.format(sample))

        # Correction with all the features
        crispy_all = CRISPRCorrection().rename(sample).fit_by(
            by=svmatrix['chr'], X=svmatrix.drop(['chr', 'crispr'], axis=1), y=svmatrix['crispr']
        )
        crispy_all = pd.concat([v.to_dataframe() for k, v in crispy_all.items()])
        crispy_all.to_csv('data/crispy/gdsc_brass/{}.all.csv'.format(sample))

        # Correction only with SV features (no comibnations)
        crispy_svs_no_comb = CRISPRCorrection().rename(sample).fit_by(
            by=svmatrix['chr'], X=svmatrix[['inversion', 'tandem-duplication', 'deletion']], y=svmatrix['crispr']
        )
        crispy_svs_no_comb = pd.concat([v.to_dataframe() for k, v in crispy_svs_no_comb.items()])
        crispy_svs_no_comb.to_csv('data/crispy/gdsc_brass/{}.svs_no_comb.csv'.format(sample))
