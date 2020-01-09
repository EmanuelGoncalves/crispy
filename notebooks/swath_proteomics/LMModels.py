#!/usr/bin/env python
# Copyright (C) 2019 Emanuel Goncalves

import logging
import numpy as np
import pandas as pd
import pkg_resources
import dtrace.DataImporter as DataImporter
from limix.qtl import scan
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.multitest import multipletests
from crispy.DataImporter import GeneExpression, CRISPR, Proteomics, WES, Sample


LOG = logging.getLogger("Crispy")
DPATH = pkg_resources.resource_filename("crispy", "data/")


class LMModels:
    def __init__(self):
        self.samplesheet = Sample().samplesheet

    def define_covariates(
        self,
        std_filter=True,
        add_institute=True,
        add_medium=True,
        add_cancertype=True,
        add_mburden=True,
        add_ploidy=True,
    ):
        covariates = []

        # # CRISPR institute of origin
        # if add_institute:
        #     crispr_insitute = pd.get_dummies(self.crispr_obj.institute).astype(int)
        #     covariates.append(crispr_insitute)

        # Cell lines culture conditions
        if add_medium:
            culture = pd.get_dummies(self.samplesheet["growth_properties"]).drop(
                columns=["Unknown"]
            )
            covariates.append(culture)

        # Cancer type
        if add_cancertype:
            ctype = pd.get_dummies(self.samplesheet["cancer_type"])
            covariates.append(ctype)

        # Mutation burden
        if add_mburden:
            m_burdern = self.samplesheet["mutational_burden"]
            covariates.append(m_burdern)

        # Ploidy
        if add_ploidy:
            ploidy = self.samplesheet["ploidy"]
            covariates.append(ploidy)

        # Merge covariates
        covariates = pd.concat(covariates, axis=1, sort=False)

        # Remove covariates with zero standard deviation
        if std_filter:
            covariates = covariates.loc[:, covariates.std() > 0]

        return covariates.dropna()

    @staticmethod
    def kinship(k):
        K = k.dot(k.T)
        K /= K.values.diagonal().mean()
        return K

    @staticmethod
    def limix_lmm(
        y,
        x,
        m=None,
        k=None,
        lik="normal",
        transform_y="scale",
        transform_x="scale",
        filter_std=True,
        min_events=10,
        parse_results=True,
        verbose=0,
    ):
        if verbose > 0:
            LOG.info(f"y_id: {y.columns[0]}")

        # Build Y
        Y = y.dropna()

        if transform_y == "scale":
            Y = pd.DataFrame(
                StandardScaler().fit_transform(Y), index=Y.index, columns=Y.columns
            )

        elif transform_y == "rank":
            Y = Y.rank(axis=1)

        # Build X
        X = x.loc[Y.index]
        X = X.loc[:, X.std() > 0]

        if transform_x == "scale":
            X = pd.DataFrame(
                StandardScaler().fit_transform(X), index=X.index, columns=X.columns
            )

        elif transform_x == "rank":
            X = X.rank(axis=1)

        elif transform_x == "min_events":
            X = X.loc[:, X.sum() >= min_events]

        # Random effects matrix
        if k is None:
            K = None

        elif k is False:
            K = LMModels.kinship(x.loc[Y.index]).values

        else:
            K = k.loc[Y.index, Y.index].values

        # Covariates + Intercept
        if m is not None:
            m = m.loc[Y.index]
            m = m.assign(intercept=1)

            if filter_std:
                m = m.loc[:, m.std() > 0]

        # Linear Mixed Model
        lmm = scan(X, Y, K=K, M=m, lik=lik, verbose=False)

        # Assemble results
        res = lmm, dict(x=X, y=Y, k=K, m=m)

        # Build melted data-frame of associations
        if parse_results:
            res = LMModels.parse_limix_lmm_results(res[0], res[1])

        return res

    @staticmethod
    def parse_limix_lmm_results(lmm_res, lmm_params):
        # Limix column names - rename dict
        cols_rename = dict(
            effect_name="x_id", pv20="pval", effsize="beta", effsize_se="beta_se"
        )

        # Export effect sizes of candidate associations (not covariates)
        effect_sizes = lmm_res.effsizes["h2"].query("effect_type == 'candidate'")
        effect_sizes = effect_sizes.set_index("test")
        effect_sizes = effect_sizes.drop(columns=["trait", "effect_type"])

        # Export p-values
        pvalues = lmm_res.stats["pv20"]

        # Assemble melted data-frame
        res = pd.concat([effect_sizes, pvalues], axis=1, sort=False)
        res = res.reset_index(drop=True).rename(columns=cols_rename)
        res = res.assign(y_id=lmm_params["y"].columns[0])
        res = res.assign(samples=lmm_params["y"].shape[0])
        res = res.assign(
            ncovariates=0 if lmm_params["m"] is None else lmm_params["m"].shape[1]
        )

        return res

    @staticmethod
    def matrix_limix_lmm(
        Y,
        X,
        M,
        K,
        lik="normal",
        transform_y="scale",
        transform_x="scale",
        filter_std=True,
        min_events=10,
        pval_adj="fdr_bh",
        verbose=1,
    ):
        # Sample intersection
        samples = list(
            set.intersection(set(Y.index), set(X.index), set(M.index), set(K.index))
        )
        LOG.info(f"Samples: {len(samples)}")

        # Iterate through X variables
        res = pd.concat(
            [
                LMModels.limix_lmm(
                    y=Y.loc[samples, [idx]],
                    x=X.loc[samples].drop(columns=idx),
                    m=M.loc[samples],
                    k=K.loc[samples, samples],
                    lik=lik,
                    transform_y=transform_y,
                    transform_x=transform_x,
                    filter_std=filter_std,
                    min_events=min_events,
                    parse_results=True,
                    verbose=verbose,
                )
                for idx in Y
            ],
            ignore_index=True,
        )

        # Multiple p-value correction
        res = LMModels.multipletests(res, field="pval", pval_method=pval_adj)

        # Order columns
        order = ["y_id", "x_id", "beta", "beta_se", "pval", "fdr", "samples", "ncovariates"]

        return res[order].sort_values("fdr")

    @staticmethod
    def multipletests(
        parsed_results,
        pval_method="fdr_bh",
        field="pval",
        fdr_field="fdr",
        idx_cols=None,
    ):
        idx_cols = ["y_id"] if idx_cols is None else idx_cols

        parsed_results_adj = []

        for idx, df in parsed_results.groupby(idx_cols):
            df[fdr_field] = multipletests(df[field], method=pval_method)[1]
            parsed_results_adj.append(df)

        parsed_results_adj = pd.concat(parsed_results_adj, ignore_index=True)

        return parsed_results_adj
