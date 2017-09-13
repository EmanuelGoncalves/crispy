"""
This module implements necessary functions to preprocess sequencing raw counts data from CRISPR dropout screens.

Copyright (C) 2017 Emanuel Goncalves
"""

import numpy as np
import scipy.stats as st
from pandas import DataFrame, Series
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

    def counts_normalisation(self):
        """
        Normalise count data per sample.

        Normalisation coefficients are estimated similarly to DESeq2 (DOI: 10.1186/s13059-014-0550-8).

        :returns DataMatrix: Normalised positive raw counts matrix

        """

        # Estimate normalisation coefficients
        m_gmean = Series(st.gmean(self, axis=1), index=self.index)
        m_gmean = m_gmean.T.divide(m_gmean).T
        m_gmean = m_gmean.median()

        return self.divide(m_gmean)

    def estimate_fold_change(self, design):
        """
        Calculate fold-changes for conditions defined in the design matrix.

        :param DataFrame design: Design matrix (samples x conditions)

        :returns DataFrame: log2 fold-changes per condition (genes x conditions)
        """

        return self.dot(design)

    def log_transform(self):
        """
        Log tansform matrix

        :return DataMatrix: Log tansform
        """
        return np.log2(self)

    def add_constant(self, constant=0.5):
        """
        Add constant to counts matrix

        :param DataFrame matrix: Read counts matrix
        :param float constant: Constant to add to read counts matrix

        :return DataMatrix: Matrix counts
        """

        return self.add(constant)

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
