"""
This module implements all the necessary functions to perform genome-wide fitting of colocalized biases
of CRISPR dropout effects.

Copyright (C) 2017 Emanuel Goncalves
"""

import scipy as sp
import pandas as pd
from sklearn import clone
from functools import partial
from datetime import datetime as dt
from crispy.ratio import GFF_HEADERS
from concurrent.futures import ThreadPoolExecutor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import WhiteKernel, ConstantKernel, RBF


# TODO: docs
class CRISPRCorrection(GaussianProcessRegressor):

    def __init__(self, kernel=None, alpha=1e-10, n_restarts_optimizer=3, optimizer='fmin_l_bfgs_b', normalize_y=True, copy_x_train=True, random_state=None):
        kernel = self.get_default_kernel() if kernel is None else kernel

        self.name = ''
        self.fit_by_value = None
        self.index = None
        self.x_columns = None
        self.y_name = None

        super().__init__(
            alpha=alpha,
            kernel=kernel,
            optimizer=optimizer,
            normalize_y=normalize_y,
            random_state=random_state,
            copy_X_train=copy_x_train,
            n_restarts_optimizer=n_restarts_optimizer,
        )

    @property
    def _constructor(self):
        return CRISPRCorrection

    @staticmethod
    def get_default_kernel(length_scale=(1e-5, 10)):
        return ConstantKernel() * RBF(length_scale_bounds=length_scale) + WhiteKernel()

    def rename(self, name=''):
        self.name = name
        return self

    def predict(self, X, return_std=True, return_cov=False):
        return super().predict(X, return_std=return_std, return_cov=return_cov)

    def regress_out(self, X=None, y=None):
        if X is None or y is None:
            X, y = self.X_train_, self.y_train_

        y_pred = self.predict(X, return_std=False, return_cov=False)

        return y - y_pred

    def to_dataframe(self, X=None, y=None):
        if X is None or y is None:
            X = pd.DataFrame(self.X_train_, index=self.index, columns=self.x_columns)
            y = pd.Series(self.y_train_, index=self.index, name=self.y_name)

        res = pd.concat([X, y], axis=1)\
            .assign(k_mean=self.predict(X, return_std=False) - self.y_train_.mean())\
            .assign(fit_by=self.fit_by_value)

        res = res.assign(regressed_out=res[self.y_name] - res['k_mean'])

        res = res.assign(var_exp=self.var_explained(X)['K1'])

        return res

    """
    Method taken from LIMIX (github.com/limix/limix)
    
    """
    @staticmethod
    def covar_rescaling(c):
        n = c.shape[0]

        p = sp.eye(n) - sp.ones((n, n)) / n

        cp = c - c.mean(0)[:, sp.newaxis]

        return (n - 1) / sp.sum(p * cp)

    def var_explained(self, X=None):
        X = self.X_train_ if X is None else X

        kernels = [('K1', self.kernel_.k1), ('K2', self.kernel_.k2)]

        cov_var = pd.Series({n: (1 / self.covar_rescaling(k.__call__(X))) for n, k in kernels}, name='Variance')

        cov_var /= cov_var.sum()

        return cov_var

    def fit(self, X, y):
        return super().fit(X, y)

    def fit_by(self, by, X, y):
        values = set(by)
        with ThreadPoolExecutor() as pool:
            res = pool.map(partial(self._fit_by, by=by, X=X, y=y), values)
        return dict(zip(*(values, res)))

    def _fit_by(self, value, by, X, y):
        print('[%s] CRISPRCorrection %s fit_by %s %s' % (dt.now().strftime('%Y-%m-%d %H:%M:%S'), self.name, by.name, value))

        # Creat a deep copy of the sklearn Gaussian model
        estimator = clone(self)
        estimator.fit_by_value = value

        by_index = list(by[by == value].index)
        estimator.index = by_index

        estimator.x_columns = list(X)
        estimator.y_name = y.name

        return estimator.fit(X.loc[by_index], y.loc[by_index])


# TODO: separate data for testing
if __name__ == '__main__':
    # Sample
    sample = 'AU565'
    print('Crispy bias correction test: {}'.format(sample))

    # Import sgRNA library
    lib = pd.read_csv('data/crispr_libs/KY_Library_v1.1_updated.csv', index_col=0).groupby('gene')['chr'].first().dropna()

    # CRISPR fold-changes
    crispr = pd.read_csv('data/crispr_gdsc_logfc.csv', index_col=0)[sample]

    # Copy-number
    cnv = pd.read_csv('data/crispy_copy_number_gene_snp.csv', index_col=0)[sample]

    # Build data-frame
    df = pd.concat([crispr.rename('fc'), cnv.rename('cnv'), lib.rename('chr')], axis=1).dropna()
    df = df[df['chr'].isin(['chr1', 'chr17'])]
    assert df.shape[0] > 0, 'Empty data-frame'

    # Correction
    crispy = CRISPRCorrection().rename(sample).fit_by(by=df['chr'], X=df[['cnv']], y=df['fc'])
    crispy = pd.concat([v.to_dataframe() for k, v in crispy.items()])
    assert crispy.groupby('fit_by')['var_exp'].first()['chr17'] > crispy.groupby('fit_by')['var_exp'].first()['chr1'], 'Var explained failled fitting'

    print(crispy)
