#!/usr/bin/env python
# Copyright (C) 2018 Emanuel Goncalves

import numpy as np
import pandas as pd
import pkg_resources
import scipy.stats as st


class Utils(object):
    DPATH = pkg_resources.resource_filename('crispy', 'data/')

    CHR_SIZES_HG19 = {
        'chr1': 249250621, 'chr2': 243199373, 'chr3': 198022430, 'chr4': 191154276, 'chr5': 180915260, 'chr6': 171115067, 'chr7': 159138663, 'chr8': 146364022,
        'chr9': 141213431, 'chr10': 135534747, 'chr11': 135006516, 'chr12': 133851895, 'chr13': 115169878, 'chr14': 107349540, 'chr15': 102531392, 'chr16': 90354753,
        'chr17': 81195210, 'chr18': 78077248, 'chr19': 59128983, 'chr20': 63025520, 'chr21': 48129895, 'chr22': 51304566, 'chrX': 155270560, 'chrY': 59373566
    }

    @staticmethod
    def bin_bkdist(distance):
        """
        Discretise genomic distances

        :param distance: float

        :return:

        """
        if distance == -1:
            bin_distance = 'diff. chr.'

        elif distance == 0:
            bin_distance = '0'

        elif 1 < distance < 10e3:
            bin_distance = '1-10 kb'

        elif 10e3 < distance < 100e3:
            bin_distance = '10-100 kb'

        elif 100e3 < distance < 1e6:
            bin_distance = '0.1-1 Mb'

        elif 1e6 < distance < 10e6:
            bin_distance = '1-10 Mb'

        else:
            bin_distance = '>10 Mb'

        return bin_distance

    @staticmethod
    def svtype(strand1, strand2, svclass, unfold_inversions):
        if svclass == 'translocation':
            svtype = 'translocation'

        elif strand1 == '+' and strand2 == '+':
            svtype = 'deletion'

        elif strand1 == '-' and strand2 == '-':
            svtype = 'tandem-duplication'

        elif strand1 == '+' and strand2 == '-':
            svtype = 'inversion_h_h' if unfold_inversions else 'inversion'

        elif strand1 == '-' and strand2 == '+':
            svtype = 'inversion_t_t' if unfold_inversions else 'inversion'

        else:
            assert False, 'SV class not recognised: strand1 == {}; strand2 == {}; svclass == {}'.format(strand1, strand2, svclass)

        return svtype

    @staticmethod
    def bin_cnv(value, thresold):
        if not np.isfinite(value):
            return np.nan

        value = int(round(value, 0))
        value = f'{value}' if value < thresold else f'{thresold}+'

        return value

    @staticmethod
    def qnorm(x):
        y = st.rankdata(x)
        y = -st.norm.isf(y / (len(x) + 1))
        return y

    @classmethod
    def get_example_data(cls, dfile='association_example_data.csv'):
        return pd.read_csv('{}/{}'.format(cls.DPATH, dfile), index_col=0)

    @classmethod
    def get_essential_genes(cls, dfile='gene_sets/curated_BAGEL_essential.csv', return_series=True):
        geneset = set(pd.read_csv('{}/{}'.format(cls.DPATH, dfile), sep='\t')['gene'])

        if return_series:
            geneset = pd.Series(list(geneset)).rename('essential')

        return geneset

    @classmethod
    def get_non_essential_genes(cls, dfile='gene_sets/curated_BAGEL_nonEssential.csv', return_series=True):
        geneset = set(pd.read_csv('{}/{}'.format(cls.DPATH, dfile), sep='\t')['gene'])

        if return_series:
            geneset = pd.Series(list(geneset)).rename('non-essential')

        return geneset

    @classmethod
    def get_crispr_lib(cls, dfile='crispr_libs/KY_Library_v1.1_annotated.csv'):
        r_cols = dict(
            index='sgrna', CHRM='chr', STARTpos='start', ENDpos='end', GENES='gene', EXONE='exon', CODE='code', STRAND='strand'
        )

        lib = pd.read_csv('{}/{}'.format(cls.DPATH, dfile), index_col=0).reset_index()

        lib = lib.rename(columns=r_cols)

        lib['chr'] = lib['chr'].apply(lambda v: f'chr{v}' if str(v) != 'nan' else np.nan)

        return lib

    @classmethod
    def get_adam_core_essential(cls, dfile='gene_sets/pancan_core.csv'):
        return set(pd.read_csv('{}/{}'.format(cls.DPATH, dfile))['ADAM PanCancer Core-Fitness genes'].rename('adam_essential'))

    @classmethod
    def get_cytobands(cls, dfile='cytoBand.txt', chrm=None):
        cytobands = pd.read_csv('{}/{}'.format(cls.DPATH, dfile), sep='\t')

        if chrm is not None:
            cytobands = cytobands[cytobands['chr'] == chrm]

        assert cytobands.shape[0] > 0, '{} not found in cytobands file'

        return cytobands


class DotDict(dict):
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__
