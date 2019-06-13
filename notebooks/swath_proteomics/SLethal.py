#!/usr/bin/env python
# Copyright (C) 2019 Emanuel Goncalves

import logging
import numpy as np
import pandas as pd
from limix.qtl import scan
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.multitest import multipletests
from crispy.DataImporter import GeneExpression, CRISPR, Proteomics, WES, Sample


class SLethal:
    def __init__(
        self,
        pvaladj_method="fdr_bh",
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
            dict(dtype="noimputed", perc_measures=0.85)
            if filter_prot_kws is None
            else filter_prot_kws
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

    def get_covariates(self, std_filter=True):
        """
        Add CRISPR institute of origin as covariate;
        Add mutational burden and cancer type as covariate to remove potential superious associations (https://doi.org/10.1016/j.cell.2019.05.005)

        """

        # CRISPR institute of origin
        crispr_insitute = (
            pd.get_dummies(self.crispr_obj.institute).astype(int).loc[self.samples]
        )

        # Cell lines culture conditions
        culture = (
            pd.get_dummies(self.ss_obj.samplesheet["growth_properties"])
            .drop(columns=["Unknown"])
            .loc[self.samples]
        )

        # Cancer type
        ctype = pd.get_dummies(self.ss_obj.samplesheet["cancer_type"]).loc[self.samples]

        # Mutation burden
        m_burdern = (
            self.ss_obj.samplesheet["mutational_burden"].round(3).loc[self.samples]
        )

        # Merge covariates
        covariates = pd.concat(
            [crispr_insitute, culture, ctype, m_burdern], axis=1, sort=False
        )

        # Remove covariates with zero standard deviation
        if std_filter:
            covariates = covariates.loc[:, covariates.std() > 0]

        return covariates

    @staticmethod
    def multipletests_per_phenotype(associations, method):
        phenotypes = set(associations["y_gene"])

        df = associations.set_index("y_gene")

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
        y, x, m=None, k=None, lik="normal", scale_y=True, scale_x=True, filter_std=True
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
            m = m.assign(intercept=1)

            if filter_std:
                m = m.loc[:, m.std() > 0]

        # Linear Mixed Model
        lmm = scan(X, Y, K=K.values, M=m, lik=lik, verbose=False)

        return lmm, dict(x=X, y=Y, k=K, m=m)

    @staticmethod
    def lmm_single_phenotype(y, x, m=None, k=None):
        cols_rename = dict(effect_name="x_gene", pv20="pval", effsize="beta", effsize_se="beta_se")

        lmm, params = SLethal.lmm_limix(y, x, m, k, scale_x=(x.dtypes[0] != np.int))

        effect_sizes = lmm.effsizes["h2"].query("effect_type == 'candidate'")
        effect_sizes = effect_sizes.set_index("test").drop(
            columns=["trait", "effect_type"]
        )
        pvalues = lmm.stats["pv20"]

        res = (
            pd.concat([effect_sizes, pvalues], axis=1, sort=False)
            .reset_index(drop=True)
            .rename(columns=cols_rename)
        )
        res = res.assign(y_gene=y.columns[0])
        res = res.assign(samples=params["y"].shape[0])
        res = res.assign(ncovariates=params["m"].shape[1])

        return res.sort_values("pval")

    def genetic_interactions(self, y, x, m=None, k=None, z=None):
        """
        Determine genetic interactions between gene-products in y and x considering covariates m and sample structure
        defined in k. z can be provided to introduce an extra covariate of gene-product considered in y, e.g. predicting
        protein abundance of Gene A using gene expression of as a covariate.

        :param y: pandas.DataFrame
        :param x: pandas.DataFrame
        :param m: pandas.DataFrame
        :param k: pandas.DataFrame
        :param z: pandas.DataFrame
        :return: pandas.DataFrame
        """
        # - Kinship matrix (random effects)
        k = self.kinship(x.T) if k is None else k
        m = self.get_covariates() if m is None else m

        # - Single feature linear mixed regression
        # Association
        gi = []

        for gene in y.index:
            if self.verbose > 0:
                self.logger.info(f"GIs LMM test of {gene}")

            if z is None:
                m_gene = m.copy()
            else:
                m_gene = z.loc[[gene]].T
                m_gene = pd.Series(
                    StandardScaler().fit_transform(m_gene)[:, 0],
                    index=m_gene.index,
                    name="z_gene",
                )
                m_gene = pd.concat([m_gene, m], axis=1, sort=False)

            gis = self.lmm_single_phenotype(y=y.loc[[gene]].T, x=x.T, k=k, m=m_gene)

            # res = self.lmm_limix(y=y.loc[[gene]].T, x=x.T, k=k, m=m_gene)
            # print(res[0])

            gi.append(gis)

        gi = pd.concat(gi)

        # Multiple p-value correction
        gi = self.multipletests_per_phenotype(gi, method=self.pvaladj_method)

        # Sort p-values
        gi = gi.sort_values(["adjpval", "pval"])

        return gi
