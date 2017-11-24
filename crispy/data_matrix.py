"""
This module implements necessary functions to preprocess sequencing raw counts data from CRISPR dropout screens.

Copyright (C) 2017 Emanuel Goncalves
"""

import scipy.stats as st
from pandas import DataFrame
from scipy_sugar.stats import quantile_gaussianize


class DataMatrix(DataFrame):

    def __init__(self, data=None, index=None, columns=None, dtype=None, copy=False):
        """
        Initialise a DataMatrix containing

        :param DataFrame matrix: Data
        """

        DataFrame.__init__(self, data, index, columns, dtype, copy)

    @property
    def _constructor(self):
        return DataMatrix

    def estimate_size_factors(self):
        """
        Normalise count data per sample.

        Normalisation coefficients are estimated similarly to DESeq2 (DOI: 10.1186/s13059-014-0550-8).

        :returns DataMatrix: Normalised positive raw counts matrix

        """
        return self.divide(st.gmean(self, axis=1), axis=0).median().rename('size_factors')

    def zscore(self):
        """
        Calculates a z score for each sample per column.

        :return DataMatrix:
        """
        return DataMatrix(st.zscore(self), index=self.index, columns=self.columns)

    def quantile_normalize(self):
        """
        Quantile normalises by column

        :return DataMatrix:
        """
        return DataMatrix({c: quantile_gaussianize(self[c].values) for c in self}, index=self.index)
