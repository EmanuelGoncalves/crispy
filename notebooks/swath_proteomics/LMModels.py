#!/usr/bin/env python
# Copyright (C) 2019 Emanuel Goncalves

import logging
import pandas as pd
import pkg_resources
from limix.qtl import scan
from crispy.DataImporter import Sample
from sklearn.preprocessing import StandardScaler
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

    @staticmethod
    def define_covariates(
        std_filter=True,
        add_institute=False,
        add_medium=True,
        add_cancertype=True,
        add_mburden=True,
        add_ploidy=True,
        crispr_institute_file="crispr/CRISPR_Institute_Origin_20191108.csv.gz",
    ):
        # Imports
        samplesheet = Sample().samplesheet
        institute = pd.read_csv(
            f"{DPATH}/{crispr_institute_file}", index_col=0, header=None
        ).iloc[:, 0]

        # Covariates
        covariates = []

        # CRISPR institute of origin
        if add_institute:
            institute = pd.get_dummies(institute).astype(int)
            covariates.append(institute)

        # Cell lines culture conditions
        if add_medium:
            culture = pd.get_dummies(samplesheet["growth_properties"]).drop(
                columns=["Unknown"]
            )
            covariates.append(culture)

        # Cancer type
        if add_cancertype:
            ctype = pd.get_dummies(samplesheet["cancer_type"])
            covariates.append(ctype)

        # Mutation burden
        if add_mburden:
            m_burdern = samplesheet["mutational_burden"]
            covariates.append(m_burdern)

        # Ploidy
        if add_ploidy:
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
        output_file=None,
        verbose=0,
    ):
        if verbose > 0:
            LOG.info(f"y_id: {y.columns[0]}")

        # Build Y
        Y = y.dropna()
        y_name = Y.columns[0]
        y_obs = list(Y.index)

        if transform_y == "scale":
            Y = StandardScaler().fit_transform(Y)
        elif transform_y == "rank":
            Y = Y.rank(axis=1).values

        # Build X
        X = x.loc[y_obs]
        X = X.loc[:, X.std() > 0]
        x_columns = list(X.columns)

        if transform_x == "scale":
            X = StandardScaler().fit_transform(X)
        elif transform_x == "rank":
            X = X.rank(axis=1).values
        elif transform_x == "min_events":
            X = X.loc[:, X.sum() >= min_events]
            x_columns = list(X.columns)
            X = X.values

        # Random effects matrix
        if k is None:
            K = None
        elif k is False:
            K = LMModels.kinship(x.loc[y_obs]).values
        else:
            K = k.loc[y_obs, y_obs].values

        # Covariates + Intercept
        if m is not None:
            m = m.loc[y_obs]
            if filter_std:
                m = m.loc[:, m.std() > 0]
            m = m.assign(intercept=1).values

        # Linear Mixed Model
        lmm = scan(X, Y, K=K, M=m, lik=lik, verbose=False)

        # Build melted data-frame of associations
        if parse_results:
            res = lmm.effsizes["h2"].query("effect_type == 'candidate'")
            res = pd.DataFrame(
                dict(
                    y_id=y_name,
                    x_id=x_columns,
                    beta=res["effsize"].round(5).values,
                    beta_se=res["effsize_se"].round(5).values,
                    pval=lmm.stats.loc[res["test"], "pv20"].values,
                    nsamples=Y.shape[0],
                    ncovariates=0 if m is None else m.shape[1],
                )
            )

            if output_file is None:
                return res

            else:
                res.to_csv(output_file, index=False, compression="gzip")

        else:
            return lmm, dict(x=X, y=Y, k=K, m=m)

    @staticmethod
    def matrix_limix_lmm(
        Y,
        X,
        M,
        K,
        x_feature_type="all",
        lik="normal",
        transform_y="scale",
        transform_x="scale",
        filter_std=True,
        min_events=10,
        pval_adj="fdr_bh",
        pval_adj_overall=False,
        verbose=1,
    ):

        # Sample intersection
        samples = list(
            set.intersection(set(Y.index), set(X.index), set(M.index), set(K.index))
        )
        LOG.info(f"Samples: {len(samples)}")

        # Prepare X
        x = X.loc[samples]

        # Iterate through Y variables
        lmm_res = []

        for idx in Y:
            if x_feature_type == "drop_y":
                x = x.drop(columns=idx)

            elif x_feature_type == "same_y":
                if idx not in x.columns:
                    LOG.warning(f"[same_y option] Y feature not in X: {idx}")
                    continue

                x = x[[idx]]

            # LMM
            lmm_res.append(
                LMModels.limix_lmm(
                    y=Y.loc[samples, [idx]],
                    x=x,
                    m=M,
                    k=K,
                    lik=lik,
                    transform_y=transform_y,
                    transform_x=transform_x,
                    filter_std=filter_std,
                    min_events=min_events,
                    parse_results=True,
                    verbose=verbose,
                )
            )

        lmm_res = pd.concat(lmm_res, ignore_index=True)

        # Multiple p-value correction
        if pval_adj_overall:
            lmm_res = lmm_res.assign(
                fdr=multipletests(lmm_res["pval"], method=pval_adj)[1]
            )

        else:
            lmm_res = LMModels.multipletests(
                lmm_res, field="pval", pval_method=pval_adj
            )

        return lmm_res.sort_values("fdr")[LMModels.RES_ORDER]

    @staticmethod
    def write_limix_lmm(
            Y,
            X,
            M,
            K,
            output_folder,
            lik="normal",
            transform_y="scale",
            transform_x="scale",
            filter_std=True,
            min_events=10,
            verbose=1,
    ):
        # Sample intersection
        samples = list(
            set.intersection(set(Y.index), set(X.index), set(M.index), set(K.index))
        )
        LOG.info(f"Samples: {len(samples)}")

        for idx in Y:
            LMModels.limix_lmm(
                y=Y.loc[samples, [idx]],
                x=X,
                m=M,
                k=K,
                lik=lik,
                transform_y=transform_y,
                transform_x=transform_x,
                filter_std=filter_std,
                min_events=min_events,
                parse_results=True,
                output_file=f"{output_folder}/{idx}.csv.gz",
                verbose=verbose,
            )

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
