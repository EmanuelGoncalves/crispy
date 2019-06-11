#!/usr/bin/env python
# Copyright (C) 2019 Emanuel Goncalves

import logging
import numpy as np
import pandas as pd
from limix.qtl import st_scan
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.multitest import multipletests
from crispy.DataImporter import GeneExpression, CRISPR, Proteomics, WES, Sample


class SLethal:
    def __init__(
        self,
        pvaladj_method="bonferroni",
        min_events=5,
        verbose=1,
        use_crispr=True,
        use_wes=True,
        use_proteomics=True,
        use_transcriptomics=True,
        filter_crispr_kws=None,
        filter_gexp_kws=None,
        filter_prot_kws=None,
        filter_wes_kws=None,
    ):
        self.verbose = verbose
        self.min_events = min_events
        self.pvaladj_method = pvaladj_method

        self.ss_obj = Sample()

        self.gexp_obj = GeneExpression()
        self.prot_obj = Proteomics()
        self.crispr_obj = CRISPR()
        self.wes_obj = WES()

        # Samples overlap
        set_samples = [
            (self.crispr_obj, use_crispr),
            (self.gexp_obj, use_transcriptomics),
            (self.prot_obj, use_proteomics),
            (self.wes_obj, use_wes),
        ]
        set_samples = [list(o.get_data().columns) for o, f in set_samples if f]
        self.samples = set.intersection(*map(set, set_samples))

        # Get logger
        self.logger = logging.getLogger("Crispy")
        self.logger.info(f"Samples={len(self.samples)}")

        # Filter kws
        self.filter_crispr_kws = (
            dict(scale=True, abs_thres=0.5)
            if filter_crispr_kws is None
            else filter_crispr_kws
        )
        self.filter_crispr_kws["subset"] = self.samples

        self.filter_gexp_kws = (
            dict(dtype="voom") if filter_gexp_kws is None else filter_gexp_kws
        )
        self.filter_gexp_kws["subset"] = self.samples

        self.filter_prot_kws = (
            dict(dtype="noimputed") if filter_prot_kws is None else filter_prot_kws
        )
        self.filter_prot_kws["subset"] = self.samples

        self.filter_wes_kws = (
            dict(as_matrix=True) if filter_wes_kws is None else filter_wes_kws
        )
        self.filter_wes_kws["subset"] = self.samples

        # Load and filter data-sets
        self.gexp = self.gexp_obj.filter(**self.filter_gexp_kws)
        self.logger.info(f"Gene-expression={self.gexp.shape}")

        self.prot = self.prot_obj.filter(**self.filter_prot_kws)
        self.logger.info(f"Proteomics={self.prot.shape}")

        self.crispr = self.crispr_obj.filter(**self.filter_crispr_kws)
        self.logger.info(f"CRISPR-Cas9={self.crispr.shape}")

        self.wes = self.wes_obj.filter(**self.filter_wes_kws)
        self.logger.info(f"WES={self.wes.shape}")

    def get_covariates(self):
        """
        Add CRISPR institute of origin as covariate;
        Add mutational burden and cancer type as covariate to remove potential superious associations (https://doi.org/10.1016/j.cell.2019.05.005)

        """

        # CRISPR institute of origin
        crispr_insitute = pd.get_dummies(self.crispr_obj.institute)

        # Cell lines culture conditions
        culture = pd.get_dummies(self.ss_obj.samplesheet["growth_properties"]).drop(
            columns=["Unknown"]
        )

        # Cancer type
        ctype = pd.get_dummies(self.ss_obj.samplesheet["cancer_type"])

        # Mutation burden
        m_burdern = self.ss_obj.samplesheet["mutational_burden"].copy()

        # Merge covariates
        covariates = pd.concat(
            [crispr_insitute, culture, ctype, m_burdern], axis=1, sort=False
        )
        covariates = covariates.loc[self.samples]

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
    def lmm_limix(
        y,
        x,
        m=None,
        k=None,
        add_intercept=True,
        lik="normal",
        scale_y=True,
        scale_x=True,
    ):
        # Build matrices
        Y = y.dropna()

        if scale_y:
            Y = pd.DataFrame(
                StandardScaler().fit_transform(Y), index=Y.index, columns=Y.columns
            )

        X = x.loc[Y.index]

        if scale_x:
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
        lmm, params = SLethal.lmm_limix(y, x, m, k, scale_x=(x.dtypes[0] != np.int))

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

    def genetic_interactions(self, x, m=None, k=None):
        # - Kinship matrix (random effects)
        k = self.kinship(x.T) if k is None else k
        m = self.get_covariates() if m is None else m

        # - Single feature linear mixed regression
        # Association
        gi = []

        for gene in self.crispr.index:
            if self.verbose > 0:
                self.logger.info(f"GIs LMM test of {gene}")

            gis = self.lmm_single_phenotype(self.crispr.loc[[gene]].T, x.T, k=k, m=m)

            gi.append(gis)

        gi = pd.concat(gi)

        # Multiple p-value correction
        gi = self.multipletests_per_phenotype(gi, method=self.pvaladj_method)

        # Sort p-values
        gi = gi.sort_values(["adjpval", "pval"])

        return gi
