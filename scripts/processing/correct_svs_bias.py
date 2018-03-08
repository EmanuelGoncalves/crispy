#!/usr/bin/env python
# Copyright (C) 2018 Emanuel Goncalves

import pickle
import numpy as np
import pandas as pd
import scripts as mp
from crispy.biases_correction import CRISPRCorrection
from scripts.plotting.annotate_bedpe import annotate_brass_bedpe


SVTYPES = ['inversion', 'tandem-duplication', 'deletion']


def count_svs(svs):
    counts = dict(zip(*(SVTYPES, np.repeat(0, len(SVTYPES)))))
    counts.update(dict(zip(*(np.unique(svs.split(','), return_counts=True)))))
    return counts


def present_svs(svs):
    svs_set = set(svs.split(','))
    return {sv.replace('-', ''): int(sv in svs_set) for sv in SVTYPES}


if __name__ == '__main__':
    # Annotate BRASS bedpes
    bed_dfs = annotate_brass_bedpe(mp.WGS_BRASS_BEDPE, bkdist=-1, splitreads=False, samples=mp.BRCA_SAMPLES)

    # - CRISPR
    c_gdsc_fc = pd.read_csv(mp.CRISPR_GENE_FC, index_col=0).dropna()

    # - CRISPR lib
    sgrna_lib = pd.read_csv(mp.LIBRARY, index_col=0)

    # - Copy-number
    cnv = pd.read_csv(mp.CN_GENE.format('wgs'), index_col=0)

    # - Non-expressed genes
    nexp = pickle.load(open(mp.NON_EXP_PICKLE, 'rb'))

    # - Overlap
    genes = set(c_gdsc_fc.index).intersection(cnv.index)

    # -
    # sample = 'HCC1143'
    crispy_samples = {}
    for sample in mp.BRCA_SAMPLES:
        print('[INFO] {}'.format(sample))

        svmatrix = pd.DataFrame(bed_dfs[bed_dfs['sample'] == sample].set_index('feature')['collapse'].apply(present_svs).to_dict()).T

        svmatrix = svmatrix.reindex(genes).replace(np.nan, 0).astype(int)

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
