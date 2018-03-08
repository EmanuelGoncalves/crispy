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

    # - Correction
    # sample = 'HCC1143'
    crispy_samples = {}
    for sample in mp.BRCA_SAMPLES:
        # Assemble matrix
        svmatrix = pd.DataFrame(bed_dfs[bed_dfs['sample'] == sample].set_index('feature')['collapse'].apply(present_svs).to_dict()).T
        svmatrix = svmatrix.reindex(genes).replace(np.nan, 0).astype(int)
        svmatrix = svmatrix.assign(chr=sgrna_lib.groupby('gene')['chr'].first().reindex(svmatrix.index).values)
        svmatrix = svmatrix.assign(crispr=c_gdsc_fc[sample].reindex(svmatrix.index).values)
        svmatrix = svmatrix.assign(cnv=cnv[sample].reindex(svmatrix.index).values)
        svmatrix = svmatrix.dropna()

        # Correction
        cmodes = {
            'cnv': {
                'x': svmatrix[['cnv']], 'y': svmatrix['crispr'], 'by': svmatrix['chr']
            },
            'svs': {
                'x': svmatrix[['deletion', 'inversion', 'tandemduplication']], 'y': svmatrix['crispr'], 'by': svmatrix['chr']
            },
            'all': {
                'x': svmatrix[['cnv', 'deletion', 'inversion', 'tandemduplication']], 'y': svmatrix['crispr'], 'by': svmatrix['chr']
            }
        }

        for n in cmodes:
            print('[INFO] {}: {}'.format(sample, n))

            df = CRISPRCorrection().rename(sample).fit_by(by=cmodes[n]['by'], X=cmodes[n]['x'], y=cmodes[n]['y'])

            df = pd.concat([v.to_dataframe() for k, v in df.items()])

            df.to_csv('{}/{}.{}.csv'.format(mp.CRISPY_WGS_OUTDIR, sample, n))
