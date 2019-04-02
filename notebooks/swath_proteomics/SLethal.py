#!/usr/bin/env python
# Copyright (C) 2019 Emanuel Goncalves

import logging
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from limix.qtl import st_scan
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.multitest import multipletests
from crispy.DataImporter import GeneExpression, CRISPR, Proteomics, Sample


class SLethal:
    def __init__(self, pvaladj_method="bonferroni", min_events=5, verbose=1):
        self.verbose = verbose
        self.min_events = min_events
        self.pvaladj_method = pvaladj_method

        self.ss_obj = Sample()

        self.gexp_obj = GeneExpression()
        self.prot_obj = Proteomics()
        self.crispr_obj = CRISPR()

        self.samples = list(
            set.intersection(
                set(self.gexp_obj.get_data().columns),
                set(self.crispr_obj.get_data().columns),
                set(self.prot_obj.get_data().columns),
            )
        )

        self.logger = logging.getLogger("Crispy")
        self.logger.info(f"Samples={len(self.samples)}")

        self.gexp = self.gexp_obj.filter(
            subset=self.samples, dtype="voom"
        )
        self.logger.info(f"Gene-expression={self.gexp.shape}")

        self.prot = self.prot_obj.filter(
            subset=self.samples, dtype="protein"
        )
        self.logger.info(f"Proteomics={self.prot.shape}")

        self.crispr = self.crispr_obj.filter(
            subset=self.samples, scale=True
        )
        self.logger.info(f"CRISPR-Cas9={self.crispr.shape}")

    def get_covariates(self):
        # CRISPR institute of origin PC
        crispr_insitute = pd.get_dummies(self.crispr_obj.institute)

        # Cell lines culture conditions
        culture = pd.get_dummies(self.ss_obj.samplesheet["growth_properties"]).drop(
            columns=["Unknown"]
        )

        # Merge covariates
        covariates = pd.concat([crispr_insitute, culture], axis=1, sort=False).loc[
            self.samples
        ]

        return covariates

    @staticmethod
    def multipletests_per_phenotype(associations, method):
        phenotypes = set(associations["phenotype"])

        df = associations.set_index("phenotype")

        df = pd.concat(
            [
                df.loc[p].assign(
                    adjpval=multipletests(df.loc[p, "pval"], method=method)[1]
                )
                for p in phenotypes
            ]
        ).reset_index()

        return df

    @staticmethod
    def kinship(k):
        K = k.dot(k.T)
        K /= K.values.diagonal().mean()
        return K

    @staticmethod
    def lmm_limix(y, x, m=None, k=None, add_intercept=True, lik="normal"):
        # Build matrices
        Y = y.dropna()
        Y = pd.DataFrame(
            StandardScaler().fit_transform(Y), index=Y.index, columns=Y.columns
        )

        X = x.loc[Y.index]
        X = pd.DataFrame(
            StandardScaler().fit_transform(X), index=X.index, columns=X.columns
        )

        # Random effects matrix
        if k is None:
            K = SLethal.kinship(x.loc[Y.index])

        else:
            K = k.loc[Y.index, Y.index]

        # Covariates
        if m is not None:
            m = m.loc[Y.index]

        # Add intercept
        if add_intercept and (m is not None):
            m["intercept"] = 1
        elif add_intercept and (m is None):
            m = pd.DataFrame(
                np.ones((Y.shape[0], 1)), index=Y.index, columns=["intercept"]
            )
        else:
            m = pd.DataFrame(
                np.zeros((Y.shape[0], 1)), index=Y.index, columns=["zeros"]
            )

        # Linear Mixed Model
        lmm = st_scan(X, Y, K=K, M=m, lik=lik, verbose=False)

        return lmm, dict(x=X, y=Y, k=K, m=m)

    @staticmethod
    def lmm_single_phenotype(y, x, m=None, k=None):
        lmm, params = SLethal.lmm_limix(y, x, m, k)

        res = pd.DataFrame(
            dict(
                beta=lmm.variant_effsizes.values,
                pval=lmm.variant_pvalues.values,
                gene=params["x"].columns,
            )
        )

        res = res.assign(phenotype=y.columns[0])
        res = res.assign(samples=params["y"].shape[0])

        return res

    def genetic_interactions_gexp(self, add_covariates=True, add_random_effects=True):
        # - Kinship matrix (random effects)
        k = self.kinship(self.gexp.T) if add_random_effects else None
        m = self.get_covariates() if add_covariates else None

        # - Single feature linear mixed regression
        # Association
        gi_gexp = []

        for gene in self.crispr.index:
            if self.verbose > 0:
                self.logger.info(f"GIs LMM test of {gene}")

            gis = self.lmm_single_phenotype(
                self.crispr.loc[[gene]].T, self.gexp.T, k=k, m=m
            )

            gi_gexp.append(gis)

        gi_gexp = pd.concat(gi_gexp)

        # Multiple p-value correction
        gi_gexp = self.multipletests_per_phenotype(gi_gexp, method=self.pvaladj_method)

        # Sort p-values
        gi_gexp = gi_gexp.sort_values(["adjpval", "pval"])

        return gi_gexp
