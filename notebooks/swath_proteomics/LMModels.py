#!/usr/bin/env python
# Copyright (C) 2019 Emanuel Goncalves

import logging
import numpy as np
import pandas as pd
import pkg_resources
from statsmodels.stats.multitest import multipletests


LOG = logging.getLogger("Crispy")
DPATH = pkg_resources.resource_filename("crispy", "data/")


class LMModels:
    RES_ORDER = [
        "y_id",
        "x_id",
        "beta",
        "beta_se",
        "pval",
        "fdr",
        "nsamples",
        "ncovariates",
    ]

    def __init__(
        self,
        y,
        x,
        k=None,
        m=None,
        m2=None,
        x_feature_type="all",
        m2_feature_type="same_y",
        add_intercept=True,
        lik="normal",
        transform_y="scale",
        transform_x="scale",
        transform_m2="scale",
        x_min_events=None,
        institute=True,
        verbose=1,
    ):
        # Misc
        self.verbose = verbose
        self.x_feature_type = x_feature_type
        self.m2_feature_type = m2_feature_type
        self.add_intercept = add_intercept

        # LIMIX parameters
        self.lik = lik

        # Preprocessing steps
        self.transform_y = transform_y
        self.transform_x = transform_x
        self.transform_m2 = transform_m2
        self.x_min_events = x_min_events

        # Build random effects and covariates matrices
        self.k = self.kinship(x) if k is None else k.copy()
        self.m = self.define_covariates(institute=institute) if m is None else m.copy()

        # Samples overlap
        self.samples = list(
            set.intersection(
                set(y.index), set(x.index), set(self.m.index), set(self.k.index), set(y.index) if m2 is None else set(m2.index)
            )
        )
        LOG.info(f"Samples: {len(self.samples)}")

        # Y matrix
        self.y, self.y_columns = self.__build_y(y.copy())

        # X matrix
        self.x, self.x_columns = self.__build_x(x.copy())

        # Covariates
        self.m = self.m.loc[self.samples, self.m.std() > 0]
        self.m, self.m_columns = self.m.values, np.array(list(self.m.columns))

        # Random effects matrix
        self.k = self.k.loc[self.samples, self.samples].values

        LOG.info(f"Y: {self.y.shape[1]}; X: {self.x.shape[1]}; M: {self.m.shape[1]}; K: {self.k.shape[1]}")

        # Second covariates matrix
        if m2 is not None:
            self.m2, self.m2_columns = self.__build_m2(m2.copy())
            LOG.info(f"M2: {self.m2.shape[1]}")

        else:
            self.m2, self.m2_columns = None, None

    def __build_y(self, y):
        y_ = self.transform_matrix(y.loc[self.samples], t_type=self.transform_y)
        return y_.values, np.array(list(y_.columns))

    def __build_m2(self, m2):
        m2_ = self.transform_matrix(m2.loc[self.samples], t_type=self.transform_m2)
        return m2_.values, np.array(list(m2_.columns))

    def __build_x(self, x):
        x_ = x.loc[self.samples, x.std() > 0]

        if self.x_min_events is not None:
            x_ = x_.loc[:, x_.sum() >= self.x_min_events]
        else:
            x_ = self.transform_matrix(x_, t_type=self.transform_x)

        return x_.values, np.array(list(x_.columns))

    def lmm(self, y_var):
        import limix

        # Define samples with NaNs
        y_idx = list(self.y_columns).index(y_var)
        y_nans_idx = np.isnan(self.y[:, y_idx])

        if self.verbose > 0:
            LOG.info(f"y_id: {y_var} ({y_idx}); N samples: {sum(1 - y_nans_idx)}")

        # Remove NaNs from y
        y_ = self.y[y_nans_idx == 0][:, [y_idx]]

        # Subset X
        x_ = self.x[y_nans_idx == 0]

        if self.x_feature_type == "drop_y":
            if y_var not in self.x_columns:
                LOG.warning(f"[x_feature_type=drop_y] Y feature {y_idx} not in X")

            x_ = x_[:, self.x_columns != y_var]
            x_vars = self.x_columns[self.x_columns != y_var]

        elif self.x_feature_type == "same_y":
            if y_var not in self.x_columns:
                LOG.error(f"[x_feature_type=same_y] Y feature {y_idx} not in X")

            x_ = x_[:, self.x_columns == y_var]
            x_vars = self.x_columns[self.x_columns == y_var]

        else:
            x_ = x_[:, np.std(x_, axis=0) > 0]
            x_vars = self.x_columns[np.std(x_, axis=0) > 0]

        # Subset m
        m_ = self.m[y_nans_idx == 0]
        m_ = m_[:, np.std(m_, axis=0) > 0]

        if (self.m2 is not None) and (self.m2_feature_type == "same_y"):
            m_ = np.append(m_, self.m2[y_nans_idx == 0][:, self.m2_columns == y_var], axis=1)

        if self.add_intercept:
            m_ = np.insert(m_, m_.shape[1], values=1, axis=1)

        # Subset random effects matrix
        k_ = self.k[:, y_nans_idx == 0][y_nans_idx == 0, :]

        # Linear Mixed Model
        lmm = limix.qtl.scan(G=x_, Y=y_, K=k_, M=m_, lik=self.lik, verbose=False)

        # Build results
        lmm_betas = lmm.effsizes["h2"].query("effect_type == 'candidate'")
        lmm = pd.DataFrame(
            dict(
                y_id=y_var,
                x_id=x_vars,
                beta=list(lmm_betas["effsize"].round(5)),
                beta_se=list(lmm_betas["effsize_se"].round(5)),
                pval=list(lmm.stats.loc[lmm_betas["test"], "pv20"]),
                nsamples=sum(1 - y_nans_idx),
                ncovariates=m_.shape[1],
            )
        )

        return lmm

    def matrix_lmm(self, pval_adj="fdr_bh", pval_adj_overall=False):
        # Iterate through Y variables
        res = pd.concat([self.lmm(y_var=i) for i in self.y_columns], ignore_index=True)

        # Multiple p-value correction
        if pval_adj_overall:
            res = res.assign(fdr=multipletests(res["pval"], method=pval_adj)[1])
        else:
            res = self.multipletests(res, field="pval", pval_method=pval_adj)

        return res.sort_values("fdr")[self.RES_ORDER]

    def write_lmm(self, output_folder):
        for i in self.y_columns:
            self.lmm(y_var=i).to_csv(
                f"{output_folder}/{i}.csv.gz", index=False, compression="gzip"
            )

    @staticmethod
    def multipletests(
        parsed_results, pval_method="fdr_bh", field="pval", idx_cols=None
    ):
        idx_cols = ["y_id"] if idx_cols is None else idx_cols

        parsed_results_adj = []

        for idx, df in parsed_results.groupby(idx_cols):
            df = df.assign(fdr=multipletests(df[field], method=pval_method)[1])
            parsed_results_adj.append(df)

        parsed_results_adj = pd.concat(parsed_results_adj, ignore_index=True)

        return parsed_results_adj

    @staticmethod
    def transform_matrix(matrix, t_type="scale"):
        if t_type == "scale":
            from sklearn.preprocessing import StandardScaler
            matrix = pd.DataFrame(
                StandardScaler().fit_transform(matrix),
                index=matrix.index,
                columns=matrix.columns,
            )

        elif t_type == "rank":
            matrix = matrix.rank(axis=1).values

        else:
            LOG.warning(
                f"{t_type} transformation not supported. Original matrix returned."
            )

        return matrix

    @staticmethod
    def define_covariates(
        std_filter=True,
        medium=True,
        cancertype=True,
        mburden=True,
        ploidy=True,
        institute=False,
        institute_file="crispr/CRISPR_Institute_Origin_20191108.csv.gz",
    ):
        from crispy.DataImporter import Sample

        # Imports
        samplesheet = Sample().samplesheet
        cinstitute = pd.read_csv(
            f"{DPATH}/{institute_file}", index_col=0, header=None
        ).iloc[:, 0]

        # Covariates
        covariates = []

        # CRISPR institute of origin
        if institute:
            covariates.append(pd.get_dummies(cinstitute).astype(int))

        # Cell lines culture conditions
        if medium:
            culture = pd.get_dummies(samplesheet["growth_properties"]).drop(
                columns=["Unknown"]
            )
            covariates.append(culture)

        # Cancer type
        if cancertype:
            ctype = pd.get_dummies(samplesheet["cancer_type"])
            covariates.append(ctype)

        # Mutation burden
        if mburden:
            m_burdern = samplesheet["mutational_burden"]
            covariates.append(m_burdern)

        # Ploidy
        if ploidy:
            ploidy = samplesheet["ploidy"]
            covariates.append(ploidy)

        # Merge covariates
        covariates = pd.concat(covariates, axis=1, sort=False)

        # Remove covariates with zero standard deviation
        if std_filter:
            covariates = covariates.loc[:, covariates.std() > 0]

        return covariates.dropna()

    @staticmethod
    def kinship(k, decimal_places=5):
        K = k.dot(k.T)
        K /= K.values.diagonal().mean()
        return K.round(decimal_places)
