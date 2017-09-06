""" This module implements necessary functions to preprocess sequencing raw counts data from CRISPR dropout screens.

Copyright (C) 2017 Emanuel Goncalves
"""

import numpy as np
import pandas as pd
import scipy.stats as st


def counts_normalisation(matrix):
    """ Normalise count data per sample.

    Normalisation coefficients are estimated similarly to DESeq2 (DOI: 10.1186/s13059-014-0550-8).

    :param DataFrame matrix: Positive raw counts matrix
    :returns DataFrame: Normalised positive raw counts matrix

    """

    # Estimate normalisation coefficients
    m_gmean = pd.Series(st.gmean(matrix, axis=1), index=matrix.index)
    m_gmean = m_gmean.T.divide(m_gmean).T
    m_gmean = m_gmean.median()

    return matrix.divide(m_gmean)


def estimate_fold_change(matrix, design):
    """ Calculate fold-changes for conditions defined in the design matrix.

    :param DataFrame matrix: Normalised raw counts matrix (genes x samples)
    :param DataFrame design: Design matrix (samples x conditions)

    :returns DataFrame: log2 fold-changes per condition (genes x conditions)
    """

    foldchange = np.log2(matrix.copy())
    foldchange = foldchange.dot(design)

    return foldchange
