#!/usr/bin/env python
# Copyright (C) 2019 Emanuel Goncalves

import os
import logging
import numpy as np
import pandas as pd
import pkg_resources
import seaborn as sns
import matplotlib.pyplot as plt
from crispy.Utils import Utils
from crispy.QCPlot import QCplot
from scipy.stats import shapiro, iqr
from sklearn.mixture import GaussianMixture
from statsmodels.stats.multitest import multipletests


LOG = logging.getLogger("Crispy")
DPATH = pkg_resources.resource_filename("crispy", "data/")


class Sample:
    """
    Import module that handles the sample list (i.e. list of cell lines) and their descriptive information.

    """

    def __init__(
        self, index="model_id", samplesheet_file="model_list_2018-09-28_1452.csv"
    ):
        self.index = index

        # Import samplesheet
        self.samplesheet = (
            pd.read_csv(f"{DPATH}/{samplesheet_file}")
            .dropna(subset=[self.index])
            .set_index(self.index)
        )

    def get_covariates(self, culture_conditions=True, cancer_type=True):
        covariates = []

        # Cell lines culture conditions
        if culture_conditions:
            culture = pd.get_dummies(self.samplesheet["growth_properties"]).drop(
                columns=["Unknown"]
            )
            covariates.append(culture)

        # Cell lines culture conditions
        if cancer_type:
            ctype = pd.get_dummies(self.samplesheet["cancer_type"])
            covariates.append(ctype)

        # Merge covariates
        covariates = pd.concat(covariates, axis=1, sort=False)

        return covariates


class WES:
    def __init__(self, wes_file="WES_variants.csv.gz"):
        self.wes = pd.read_csv(f"{DPATH}/wes/{wes_file}")

    def get_data(self, as_matrix=True, mutation_class=None):
        df = self.wes.copy()

        # Filter my mutation types
        if mutation_class is not None:
            df = df[df["Classification"].isin(mutation_class)]

        if as_matrix:
            df["value"] = 1

            df = pd.pivot_table(
                df,
                index="Gene",
                columns="model_id",
                values="value",
                aggfunc="first",
                fill_value=0,
            )

        return df

    def filter(self, subset=None, min_events=5, as_matrix=False, mutation_class=None):
        df = self.get_data(as_matrix=as_matrix, mutation_class=mutation_class)

        # Subset samples
        if subset is not None:
            if as_matrix:
                df = df.loc[:, df.columns.isin(subset)]

            else:
                df = df[df["model_id"].isin(subset)]

            assert df.shape[1] != 0, "No columns after filter by subset"

        # Minimum number of events
        df = df[df.sum(1) >= min_events]

        return df


class GeneExpression:
    """
    Import module of gene-expression data-set.

    """

    def __init__(
        self, voom_file="gexp/rnaseq_voom.csv.gz"
    ):
        self.voom = pd.read_csv(f"{DPATH}/{voom_file}", index_col=0)

    def get_data(self, dtype="voom"):
        if dtype != "voom":
            assert False, f"Dtype {dtype} not supported"

        return self.voom.copy()

    def filter(self, dtype="voom", subset=None, iqr_range=None, normality=False):
        df = self.get_data(dtype=dtype)

        # Subset matrices
        if subset is not None:
            df = df.loc[:, df.columns.isin(subset)]

        # Filter by IQR
        if iqr_range is not None:
            iqr_ranges = (
                df.apply(lambda v: iqr(v, rng=iqr_range), axis=1)
                .rename("iqr")
                .to_frame()
            )

            gm_iqr = GaussianMixture(n_components=2).fit(iqr_ranges[["iqr"]])
            iqr_ranges["gm"] = gm_iqr.predict(iqr_ranges[["iqr"]])

            df = df.loc[iqr_ranges["gm"] != gm_iqr.means_.argmin()]

            LOG.info(f"IQR {iqr_range}")
            LOG.info(iqr_ranges.groupby("gm").agg({"min", "mean", "median", "max"}))

        # Filter by normality
        if normality:
            normality = df.apply(lambda v: shapiro(v)[1], axis=1)
            normality = multipletests(normality, method="bonferroni")
            df = df.loc[normality[0]]

        return df

    def is_not_expressed(self, rpkm_threshold=1, subset=None):
        rpkm = self.filter(dtype="rpkm", subset=subset)
        rpkm = (rpkm < rpkm_threshold).astype(int)
        return rpkm


class CRISPR:
    """
    Importer module for CRISPR-Cas9 screens acquired at Sanger and Broad Institutes.

    """

    LOW_QUALITY_SAMPLES = ["SIDM00096", "SIDM01085", "SIDM00958"]

    def __init__(
        self,
        sanger_fc_file="crispr/sanger_depmap18_fc_corrected.csv.gz",
        sanger_qc_file="crispr/sanger_depmap18_fc_ess_aucs.csv.gz",
        broad_fc_file="crispr/broad_depmap18q4_fc_corrected.csv.gz",
        broad_qc_file="crispr/broad_depmap18q4_fc_ess_aucs.csv.gz",
    ):
        self.SANGER_FC_FILE = sanger_fc_file
        self.SANGER_QC_FILE = sanger_qc_file

        self.BROAD_FC_FILE = broad_fc_file
        self.BROAD_QC_FILE = broad_qc_file

        self.crispr, self.institute = self.__merge_matricies()

        self.crispr = self.crispr.drop(columns=self.LOW_QUALITY_SAMPLES)

        self.qc_ess = self.__merge_qc_arrays()

    def __merge_qc_arrays(self):
        gdsc_qc = pd.read_csv(
            f"{DPATH}/{self.SANGER_QC_FILE}", header=None, index_col=0
        ).iloc[:, 0]
        broad_qc = pd.read_csv(
            f"{DPATH}/{self.BROAD_QC_FILE}", header=None, index_col=0
        ).iloc[:, 0]

        qcs = pd.concat(
            [
                gdsc_qc[self.institute[self.institute == "Sanger"].index],
                broad_qc[self.institute[self.institute == "Broad"].index],
            ]
        )

        return qcs

    def import_sanger_foldchanges(self):
        return pd.read_csv(f"{DPATH}/{self.SANGER_FC_FILE}", index_col=0)

    def import_broad_foldchanges(self):
        return pd.read_csv(f"{DPATH}/{self.BROAD_FC_FILE}", index_col=0)

    def __merge_matricies(self):
        gdsc_fc = self.import_sanger_foldchanges().dropna()
        broad_fc = self.import_broad_foldchanges().dropna()

        genes = list(set(gdsc_fc.index).intersection(broad_fc.index))

        merged_matrix = pd.concat(
            [
                gdsc_fc.loc[genes],
                broad_fc.loc[genes, [i for i in broad_fc if i not in gdsc_fc.columns]],
            ],
            axis=1,
            sort=False,
        )

        institute = pd.Series(
            {s: "Sanger" if s in gdsc_fc.columns else "Broad" for s in merged_matrix}
        )

        return merged_matrix, institute

    def get_data(self, scale=True):
        df = self.crispr.copy()

        if scale:
            df = self.scale(df)

        return df

    def filter(
        self,
        subset=None,
        scale=True,
        normality=False,
        iqr_range=None,
        iqr_range_thres=None,
        abs_thres=None,
        drop_core_essential=False,
        min_events=5,
        drop_core_essential_broad=False,
    ):
        df = self.get_data(scale=scale)

        # - Filters
        # Subset matrices
        if subset is not None:
            df = df.loc[:, df.columns.isin(subset)]

        # Filter by IQR
        if iqr_range is not None:
            iqr_ranges = (
                df.apply(lambda v: iqr(v, rng=iqr_range), axis=1)
                .rename("iqr")
                .to_frame()
            )

            if iqr_range_thres is None:
                gm_iqr = GaussianMixture(n_components=2).fit(iqr_ranges[["iqr"]])
                iqr_ranges["gm"] = gm_iqr.predict(iqr_ranges[["iqr"]])

                df = df.loc[iqr_ranges["gm"] != gm_iqr.means_.argmin()]
                LOG.info(iqr_ranges.groupby("gm").agg({"min", "mean", "median", "max"}))

            else:
                df = df.loc[iqr_ranges["iqr"] > iqr_range_thres]
                LOG.info(f"IQR threshold {iqr_range_thres}")

            LOG.info(f"IQR {iqr_range}")

        # Filter by normality
        if normality:
            normality = df.apply(lambda v: shapiro(v)[1], axis=1)
            normality = multipletests(normality, method="bonferroni")
            df = df.loc[normality[0]]

        # Filter by scaled scores
        if abs_thres is not None:
            df = df[(df.abs() > abs_thres).sum(1) >= min_events]

        # Filter out core essential genes
        if drop_core_essential:
            df = df[~df.index.isin(Utils.get_adam_core_essential())]

        if drop_core_essential_broad:
            df = df[~df.index.isin(Utils.get_broad_core_essential())]

        # - Subset matrices
        return df

    @staticmethod
    def scale(df, essential=None, non_essential=None, metric=np.median):
        if essential is None:
            essential = Utils.get_essential_genes(return_series=False)

        if non_essential is None:
            non_essential = Utils.get_non_essential_genes(return_series=False)

        assert (
            len(essential.intersection(df.index)) != 0
        ), "DataFrame has no index overlapping with essential list"

        assert (
            len(non_essential.intersection(df.index)) != 0
        ), "DataFrame has no index overlapping with non essential list"

        essential_metric = metric(df.reindex(essential).dropna(), axis=0)
        non_essential_metric = metric(df.reindex(non_essential).dropna(), axis=0)

        df = df.subtract(non_essential_metric).divide(
            non_essential_metric - essential_metric
        )

        return df


class Mobem:
    """
    Import module for Genomic binary feature table (containing mutations and copy-number calls)
    Iorio et al., Cell, 2016.

    """

    def __init__(
        self, mobem_file="mobem/PANCAN_mobem.csv.gz", drop_factors=True, add_msi=True
    ):
        self.sample = Sample()

        idmap = (
            self.sample.samplesheet.reset_index()
            .dropna(subset=["COSMIC_ID", "model_id"])
            .set_index("COSMIC_ID")["model_id"]
        )

        mobem = pd.read_csv(f"{DPATH}/{mobem_file}", index_col=0)
        mobem = mobem[mobem.index.astype(str).isin(idmap.index)]
        mobem = mobem.set_index(idmap[mobem.index.astype(str)].values)

        if drop_factors is not None:
            mobem = mobem.drop(columns={"TISSUE_FACTOR", "MSI_FACTOR", "MEDIA_FACTOR"})

        if add_msi:
            self.msi = self.sample.samplesheet.loc[mobem.index, "msi_status"]
            mobem["msi_status"] = (self.msi == "MSI-H").astype(int)[mobem.index].values

        self.mobem = mobem.astype(int).T

    def get_data(self):
        return self.mobem.copy()

    def filter(self, subset=None, min_events=5):
        df = self.get_data()

        # Subset matrices
        if subset is not None:
            df = df.loc[:, df.columns.isin(subset)]

        # Minimum number of events
        df = df[df.sum(1) >= min_events]

        return df

    @staticmethod
    def mobem_feature_to_gene(f):
        if f.endswith("_mut"):
            genes = {f.split("_")[0]}

        elif f.startswith("gain.") or f.startswith("loss."):
            genes = {
                g
                for fs in f.split("..")
                if not (fs.startswith("gain.") or fs.startswith("loss."))
                for g in fs.split(".")
                if g != ""
            }

        else:
            raise ValueError(f"{f} is not a valid MOBEM feature.")

        return genes

    @staticmethod
    def mobem_feature_type(f):
        if f.endswith("_mut"):
            return "Mutation"

        elif f.startswith("gain."):
            return "CN gain"

        elif f.startswith("loss."):
            return "CN loss"

        elif f == "msi_status":
            return f

        else:
            raise ValueError(f"{f} is not a valid MOBEM feature.")


class CopyNumber:
    def __init__(self, cnv_file="copy_number/copynumber_total_new_map.csv.gz"):
        self.copynumber = pd.read_csv(f"{DPATH}/{cnv_file}", index_col=0)

    def get_data(self):
        return self.copynumber.copy()

    def filter(self, subset=None):
        df = self.get_data()

        # Subset matrices
        if subset is not None:
            df = df.loc[:, df.columns.isin(subset)]

        return df

    @staticmethod
    def is_amplified(
        cn, ploidy, cn_threshold_low=5, cn_thresholds_high=9, ploidy_threshold=2.7
    ):
        if (ploidy <= ploidy_threshold) and (cn >= cn_threshold_low):
            return 1

        elif (ploidy > ploidy_threshold) and (cn >= cn_thresholds_high):
            return 1

        else:
            return 0


class CopyNumberSegmentation:
    def __init__(self, cnv_file="copy_number/Summary_segmentation_data_994_lines_picnic.csv.gz"):
        self.copynumber = pd.read_csv(f"{DPATH}/{cnv_file}")

    def get_data(self):
        return self.copynumber.copy()

    def filter(self, subset=None):
        df = self.get_data()

        # Subset matrices
        if subset is not None:
            df = df[df["model_id"].isin(subset)]

        return df
