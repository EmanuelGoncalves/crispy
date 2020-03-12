#!/usr/bin/env python
# Copyright (C) 2019 Emanuel Goncalves

import logging
import numpy as np
import pandas as pd
import pkg_resources
import itertools as it
from crispy.Utils import Utils
from scipy.stats import shapiro, iqr
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import quantile_transform
from statsmodels.stats.multitest import multipletests


LOG = logging.getLogger("Crispy")
DPATH = pkg_resources.resource_filename("crispy", "data/")


class Sample:
    """
    Import module that handles the sample list (i.e. list of cell lines) and their descriptive information.

    """

    def __init__(
        self,
        index="model_id",
        samplesheet_file="model_list_20200107.csv",
        growth_file="GrowthRates_v1.3.0_20190222.csv",
        institute_file="crispr/CRISPR_Institute_Origin_20191108.csv.gz",
    ):
        self.index = index

        # Import samplesheet
        self.samplesheet = (
            pd.read_csv(f"{DPATH}/{samplesheet_file}")
            .dropna(subset=[self.index])
            .set_index(self.index)
        )

        # Growth rates
        self.growth = pd.read_csv(f"{DPATH}/{growth_file}")
        self.samplesheet["growth"] = (
            self.growth.groupby(self.index)["GROWTH_RATE"]
            .mean()
            .reindex(self.samplesheet.index)
            .values
        )

        # CRISPR institute
        self.samplesheet["institute"] = (
            pd.read_csv(f"{DPATH}/{institute_file}", index_col=0, header=None)
            .iloc[:, 0]
            .reindex(self.samplesheet.index)
            .values
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

    def get_data(self, as_matrix=True, mutation_class=None, recurrence=False):
        df = self.wes.copy()

        # Filter my mutation types
        if mutation_class is not None:
            df = df[df["Classification"].isin(mutation_class)]

        if recurrence:
            df = df[df["Recurrence Filter"] == "Yes"]

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

    def filter(
        self,
        subset=None,
        min_events=5,
        as_matrix=True,
        mutation_class=None,
        recurrence=False,
    ):
        df = self.get_data(
            as_matrix=as_matrix, mutation_class=mutation_class, recurrence=recurrence
        )

        # Subset samples
        if subset is not None:
            if as_matrix:
                df = df.loc[:, df.columns.isin(subset)]

            else:
                df = df[df["model_id"].isin(subset)]

            assert df.shape[1] != 0, "No columns after filter by subset"

        # Minimum number of events
        if min_events is not None:
            df = df[df.sum(1) >= min_events]

        return df


class GeneExpression:
    """
    Import module of gene-expression data-set.

    """

    def __init__(
        self,
        voom_file="gexp/rnaseq_voom.csv.gz",
        read_count="gexp/rnaseq_20191101/rnaseq_read_count_20191101.csv",
    ):
        self.voom = pd.read_csv(f"{DPATH}/{voom_file}", index_col=0)
        self.readcount = pd.read_csv(f"{DPATH}/{read_count}", index_col=1).drop(
            columns=["model_id"]
        )

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

            df = df.reindex(iqr_ranges["gm"] != gm_iqr.means_.argmin())

            LOG.info(f"IQR {iqr_range}")
            LOG.info(iqr_ranges.groupby("gm").agg({"min", "mean", "median", "max"}))

        # Filter by normality
        if normality:
            normality = df.apply(lambda v: shapiro(v)[1], axis=1)
            normality = multipletests(normality, method="bonferroni")
            df = df.reindex(normality[0])

        return df

    def is_not_expressed(self, rpkm_threshold=1, subset=None):
        rpkm = self.filter(dtype="rpkm", subset=subset)
        rpkm = (rpkm < rpkm_threshold).astype(int)
        return rpkm


class BioGRID:
    def __init__(
        self,
        biogrid_file="BIOGRID-ALL-3.5.180.tab2.zip",
        organism=9606,
        etype="physical",
        stypes_exclude=None,
        homodymers_exclude=True,
    ):
        self.etype = etype
        self.organism = organism
        self.homodymers_exclude = homodymers_exclude
        self.stypes_exclude = (
            {"Affinity Capture-RNA", "Protein-RNA"}
            if stypes_exclude is None
            else stypes_exclude
        )

        # Import
        self.biogrid = pd.read_csv(f"{DPATH}/{biogrid_file}", sep="\t")

        # Filter by organism
        self.biogrid = self.biogrid[
            self.biogrid["Organism Interactor A"] == self.organism
        ]
        self.biogrid = self.biogrid[
            self.biogrid["Organism Interactor B"] == self.organism
        ]

        # Filter by type of interaction
        if self.etype is not None:
            self.biogrid = self.biogrid[
                self.biogrid["Experimental System Type"] == self.etype
            ]

        # Exlude experimental systems
        self.biogrid = self.biogrid[
            ~self.biogrid["Experimental System"].isin(self.stypes_exclude)
        ]

        # Exclude homodymers
        if self.homodymers_exclude:
            self.biogrid = self.biogrid[
                self.biogrid["Official Symbol Interactor A"]
                != self.biogrid["Official Symbol Interactor B"]
            ]

        # Build set of interactions (both directions, i.e. p1-p2, p2-p1)
        self.biogrid = {
            (p1, p2)
            for p1, p2 in self.biogrid[
                ["Official Symbol Interactor A", "Official Symbol Interactor B"]
            ].values
        }
        self.biogrid = list(
            {(p1, p2) for p in self.biogrid for p1, p2 in [(p[0], p[1]), (p[1], p[0])]}
        )


class CORUM:
    def __init__(
        self,
        corum_file="coreComplexes.txt",
        organism="Human",
        homodymers_exclude=True,
        protein_subset=None,
    ):
        self.organism = organism
        self.homodymers_exclude = homodymers_exclude
        self.protein_subset = protein_subset

        # Load CORUM DB
        self.db = pd.read_csv(f"{DPATH}/{corum_file}", sep="\t")
        self.db = self.db.query(f"Organism == '{organism}'")
        self.db_name = self.db.groupby("ComplexID")["ComplexName"].first()

        # Melt into list of protein pairs (both directions, i.e. p1-p2, p2-p1)
        self.db_melt = self.melt_ppi()

        # Map to gene symbols
        self.gmap = self.map_gene_name()
        self.db_melt_symbol = {
            (self.gmap.loc[p1, "GeneSymbol"], self.gmap.loc[p2, "GeneSymbol"]): i
            for (p1, p2), i in self.db_melt.items()
            if p1 in self.gmap.index and p2 in self.gmap.index
        }

        # Exclude homodymers
        if self.homodymers_exclude:
            self.db_melt_symbol = {
                (p1, p2): i for (p1, p2), i in self.db_melt_symbol.items() if p1 != p2
            }

        # Subset interactions
        if self.protein_subset is not None:
            self.db_melt_symbol = {
                (p1, p2): i
                for (p1, p2), i in self.db_melt_symbol.items()
                if p1 in self.protein_subset and p2 in self.protein_subset
            }

    def melt_ppi(self, idx_id="ComplexID", idx_sub="subunits(UniProt IDs)"):
        db_melt = self.db[[idx_id, idx_sub]].copy()
        db_melt[idx_sub] = db_melt[idx_sub].apply(
            lambda v: list(it.permutations(v.split(";"), 2))
        )
        db_melt = {p: i for i, c in db_melt[[idx_id, idx_sub]].values for p in c}
        return db_melt

    @staticmethod
    def map_gene_name(index_col="Entry"):
        idmap = pd.read_csv(f"{DPATH}/uniprot_human_idmap.tab.gz", sep="\t")

        if index_col is not None:
            idmap = idmap.dropna(subset=[index_col]).set_index(index_col)

        idmap["GeneSymbol"] = idmap["Gene names  (primary )"].apply(
            lambda v: v.split("; ")[0] if str(v).lower() != "nan" else v
        )

        return idmap


class Proteomics:
    def __init__(
        self,
        protein_matrix="proteomics/E0022_P05_protein_intensities.txt",
        manifest="proteomics/E0022_P01-P05_sample_map.txt",
        replicates_corr="proteomics/replicates_corr.csv.gz",
        broad_tmt="proteomics/broad_tmt.csv.gz",
        coread_tmt="proteomics/proteomics_coread_processed_parsed.csv",
        hgsc_prot="proteomics/hgsc_cell_lines_proteomics.csv",
        brca_prot="proteomics/brca_cell_lines_proteomics_preprocessed.csv",
    ):
        ss = Sample().samplesheet

        deprecated_ids = self.map_deprecated()

        # Import imputed protein levels
        self.protein = pd.read_csv(f"{DPATH}/{protein_matrix}", sep="\t", index_col=0).T
        self.protein["Protein"] = (
            self.protein.reset_index()["index"]
            .replace(deprecated_ids["Entry name"])
            .values
        )
        self.protein = self.protein.set_index("Protein")

        exclude_proteins = ["CEIP2_HUMAN"]
        self.protein = self.protein.drop(exclude_proteins)

        # Import manifest
        self.manifest = pd.read_csv(f"{DPATH}/{manifest}", index_col=0, sep="\t")

        # Remove excluded samples
        self.exclude_man = self.manifest[~self.manifest["SIDM"].isin(ss.index)]

        self.manifest = self.manifest[~self.manifest.index.isin(self.exclude_man.index)]
        self.protein = self.protein.drop(columns=self.exclude_man.index)

        # Import replicates correlation
        self.rep_corrs = pd.read_csv(f"{DPATH}/{replicates_corr}")
        self.sample_corrs = self.rep_corrs.groupby("SIDM_1")["pearson"].mean()

        # Import Broad TMT data-set
        self.broad = pd.read_csv(f"{DPATH}/{broad_tmt}", compression="gzip")
        self.broad = (
            self.broad.dropna(subset=["Gene_Symbol"])
            .groupby("Gene_Symbol")
            .agg(np.nanmean)
        )

        # Import CRC COREAD TMT
        self.coread = pd.read_csv(f"{DPATH}/{coread_tmt}", index_col=0)
        self.coread = self.coread.reindex(
            columns=self.coread.columns.isin(ss["model_name"])
        )

        coread_ss = ss[ss["model_name"].isin(self.coread.columns)]
        coread_ss = coread_ss.reset_index().set_index("model_name")
        self.coread = self.coread.rename(columns=coread_ss["model_id"])

        # Import HGSC proteomics
        self.hgsc = (
            pd.read_csv(f"{DPATH}/{hgsc_prot}")
            .dropna(subset=["Gene names"])
            .drop(columns=["Majority protein IDs"])
        )
        self.hgsc = self.hgsc.groupby("Gene names").mean()
        self.hgsc = self.hgsc.reindex(columns=self.hgsc.columns.isin(ss["model_name"]))

        hgsc_ss = ss[ss["model_name"].isin(self.hgsc.columns)]
        hgsc_ss = hgsc_ss.reset_index().set_index("model_name")
        self.hgsc = self.hgsc.rename(columns=hgsc_ss["model_id"])

        # Import BRCA proteomics
        self.brca = pd.read_csv(f"{DPATH}/{brca_prot}", index_col=0)
        self.brca = self.brca.reindex(columns=self.brca.columns.isin(ss["model_name"]))

        brca_ss = ss[ss["model_name"].isin(self.brca.columns)]
        brca_ss = brca_ss.reset_index().set_index("model_name")
        self.brca = self.brca.rename(columns=brca_ss["model_id"])

    def get_data(
        self,
        dtype="protein",
        average_replicates=True,
        map_ids=True,
        replicate_thres=None,
        quantile_normalise=False,
    ):
        if dtype.lower() == "protein":
            data = self.protein.copy()
        else:
            assert False, f"{dtype} not supported"

        if replicate_thres is not None:
            reps = self.rep_corrs.query(f"pearson > {replicate_thres}")
            reps = set(reps["sample1"]).union(set(reps["sample2"]))
            data = data[reps]

        if quantile_normalise:
            data = pd.DataFrame(
                quantile_transform(data, ignore_implicit_zeros=True),
                index=data.index,
                columns=data.columns,
            )

        if average_replicates:
            data = data.groupby(
                self.manifest.reindex(data.columns)["SIDM"], axis=1
            ).agg(np.nanmean)

        if map_ids:
            pmap = self.map_gene_name().reindex(data.index)["GeneSymbol"].dropna()

            data = data[data.index.isin(pmap.index)]
            data = data.groupby(pmap.reindex(data.index)).mean()

        return data

    def filter(
        self,
        dtype="protein",
        subset=None,
        normality=False,
        iqr_range=None,
        perc_measures=None,
        replicate_thres=None,
        quantile_normalise=False,
    ):
        df = self.get_data(
            dtype=dtype,
            quantile_normalise=quantile_normalise,
            replicate_thres=replicate_thres,
        )

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

            df = df.reindex(iqr_ranges["gm"] != gm_iqr.means_.argmin())

            LOG.info(f"IQR {iqr_range}")
            LOG.info(iqr_ranges.groupby("gm").agg({"min", "mean", "median", "max"}))

        # Filter by normality
        if normality:
            normality = df.apply(lambda v: shapiro(v)[1], axis=1)
            normality = multipletests(normality, method="bonferroni")
            df = df.reindex(normality[0])

        # Filter by number of obvservations
        if perc_measures is not None:
            df = df[df.count(1) > (perc_measures * df.shape[1])]

        return df

    @staticmethod
    def map_deprecated():
        return pd.read_csv(
            f"{DPATH}/uniprot_human_idmap_deprecated.tab", sep="\t", index_col=0
        )

    @staticmethod
    def map_gene_name(index_col="Entry name"):
        idmap = pd.read_csv(f"{DPATH}/uniprot_human_idmap.tab.gz", sep="\t")

        if index_col is not None:
            idmap = idmap.dropna(subset=[index_col]).set_index(index_col)

        idmap["GeneSymbol"] = idmap["Gene names  (primary )"].apply(
            lambda v: v.split("; ")[0] if str(v).lower() != "nan" else v
        )

        return idmap


class CRISPR:
    """
    Importer module for CRISPR-Cas9 screens acquired at Sanger and Broad Institutes.

    """

    def __init__(
        self,
        fc_file="crispr/CRISPR_corrected_qnorm_20191108.csv.gz",
        institute_file="crispr/CRISPR_Institute_Origin_20191108.csv.gz",
    ):
        self.crispr = pd.read_csv(f"{DPATH}/{fc_file}", index_col=0)
        self.institute = pd.read_csv(
            f"{DPATH}/{institute_file}", index_col=0, header=None
        ).iloc[:, 0]

    def get_data(self, scale=True):
        df = self.crispr.copy()

        if scale:
            df = self.scale(df)

        return df

    def filter(
        self,
        subset=None,
        scale=True,
        std_filter=False,
        abs_thres=None,
        drop_core_essential=False,
        min_events=5,
        drop_core_essential_broad=False,
        binarise_thres=None,
    ):
        df = self.get_data(scale=True)

        # - Filters
        # Subset matrices
        if subset is not None:
            df = df.loc[:, df.columns.isin(subset)]

        # Filter by scaled scores
        if abs_thres is not None:
            df = df[(df.abs() > abs_thres).sum(1) >= min_events]

        # Filter out core essential genes
        if drop_core_essential:
            df = df[~df.index.isin(Utils.get_adam_core_essential())]

        if drop_core_essential_broad:
            df = df[~df.index.isin(Utils.get_broad_core_essential())]

        # - Subset matrices
        x = self.get_data(scale=scale).reindex(index=df.index, columns=df.columns)

        if binarise_thres is not None:
            x = (x < binarise_thres).astype(int)

        if std_filter:
            x = x.reindex(x.std(1) > 0)

        return x

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
            self.msi = self.sample.samplesheet.reindex(mobem.index)["msi_status"]
            mobem["msi_status"] = (self.msi == "MSI-H").astype(int)[mobem.index].values

        self.mobem = mobem.astype(int).T

    def get_data(self):
        return self.mobem.copy()

    def filter(self, subset=None, min_events=5):
        df = self.get_data()

        # Subset matrices
        if subset is not None:
            df = df.reindex(columns=df.columns.isin(subset))

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
    def __init__(
        self,
        cnv_file="copy_number/cnv_abs_copy_number_picnic_20191101.csv.gz",
        calculate_deletions=False,
        calculate_amplifications=False,
    ):
        self.ss_obj = Sample()

        self.copynumber = pd.read_csv(f"{DPATH}/{cnv_file}", index_col=0)

        self.ploidy = self.ss_obj.samplesheet["ploidy"]

        if calculate_deletions:
            self.copynumber_del = pd.DataFrame(
                {
                    s: self.copynumber[s].apply(
                        lambda v: CopyNumber.is_deleted(v, self.ploidy[s])
                    )
                    for s in self.copynumber
                    if s in self.ploidy
                }
            )

        if calculate_amplifications:
            self.copynumber_amp = pd.DataFrame(
                {
                    s: self.copynumber[s].apply(
                        lambda v: CopyNumber.is_amplified(v, self.ploidy[s])
                    )
                    for s in self.copynumber
                    if s in self.ploidy
                }
            )

    def get_data(self, dtype="matrix"):
        if dtype == "del":
            res = self.copynumber_del.copy()

        elif dtype == "amp":
            res = self.copynumber_amp.copy()

        else:
            res = self.copynumber.copy()

        return res

    def filter(self, subset=None, dtype="cn"):
        df = self.get_data(dtype=dtype)

        # Subset matrices
        if subset is not None:
            df = df.loc[:, df.columns.isin(subset)]

        return df

    def ploidy_from_segments(
        self, seg_file="copy_number/Summary_segmentation_data_994_lines_picnic.csv.gz"
    ):
        copynumber_seg = pd.read_csv(f"{DPATH}/{seg_file}")

        return pd.Series(
            {
                s: self.calculate_ploidy(df)[1]
                for s, df in copynumber_seg.groupby("model_id")
            }
        )

    @staticmethod
    def calculate_ploidy(cn_seg):
        cn_seg = cn_seg[~cn_seg["chr"].isin(["chrX", "chrY"])]

        cn_seg = cn_seg.assign(length=cn_seg["end"] - cn_seg["start"])
        cn_seg = cn_seg.assign(
            cn_by_length=cn_seg["length"] * (cn_seg["copy_number"] + 1)
        )

        chrm = (
            cn_seg.groupby("chr")["cn_by_length"]
            .sum()
            .divide(cn_seg.groupby("chr")["length"].sum())
            - 1
        )

        ploidy = (cn_seg["cn_by_length"].sum() / cn_seg["length"].sum()) - 1

        return chrm, ploidy

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

    @staticmethod
    def is_deleted(cn, ploidy, ploidy_threshold=2.7):
        if (ploidy <= ploidy_threshold) and (cn == 0):
            return 1

        elif (ploidy > ploidy_threshold) and (cn < (ploidy - ploidy_threshold)):
            return 1

        else:
            return 0


class Methylation:
    """
    Import module for Illumina Methylation 450k arrays
    """

    def __init__(
        self, methy_gene_promoter="methylation/methy_beta_gene_promoter.csv.gz"
    ):
        self.methy_promoter = pd.read_csv(f"{DPATH}/{methy_gene_promoter}", index_col=0)

    def get_data(self):
        return self.methy_promoter.copy()

    def filter(self, subset=None):
        df = self.get_data()

        # Subset matrices
        if subset is not None:
            df = df.loc[:, df.columns.isin(subset)]

        return df

    @staticmethod
    def discretise(beta):
        if beta < 0.33:
            return "umethylated"
        elif beta > 0.66:
            return "methylated"
        else:
            return "hemimethylated"


class CopyNumberSegmentation:
    def __init__(
        self, cnv_file="copy_number/Summary_segmentation_data_994_lines_picnic.csv.gz"
    ):
        self.copynumber = pd.read_csv(f"{DPATH}/{cnv_file}")

    def get_data(self):
        return self.copynumber.copy()

    def filter(self, subset=None):
        df = self.get_data()

        # Subset matrices
        if subset is not None:
            df = df[df["model_id"].isin(subset)]

        return df


class PipelineResults:
    def __init__(
        self,
        results_dir,
        import_fc=False,
        import_bagel=False,
        import_mageck=False,
        mageck_fdr_thres=0.1,
        fc_thres=-0.5,
    ):
        self.results_dir = results_dir

        if import_fc:
            self.fc_thres = fc_thres
            self.fc, self.fc_c, self.fc_cq, self.fc_s, self.fc_b = (
                self.import_fc_results()
            )
            LOG.info("Fold-changes imported")

        if import_bagel:
            self.bf, self.bf_q, self.bf_b = self.import_bagel_results()
            LOG.info("BAGEL imported")

        if import_mageck:
            self.mageck_fdr_thres = mageck_fdr_thres
            self.mdep_fdr, self.mdep_bin = self.import_mageck_results()
            LOG.info("MAGeCK imported")

    def import_bagel_results(self):
        # Bayesian factors
        bf = pd.read_csv(
            f"{self.results_dir}/_BayesianFactors.tsv", sep="\t", index_col=0
        )

        # Quantile normalised bayesian factors
        bf_q = pd.read_csv(
            f"{self.results_dir}/_scaledBayesianFactors.tsv", sep="\t", index_col=0
        )

        # Binarised bayesian factors
        bf_b = pd.read_csv(
            f"{self.results_dir}/_binaryDepScores.tsv", sep="\t", index_col=0
        )

        return bf, bf_q, bf_b

    def import_fc_results(self):
        # Fold-changes
        fc = pd.read_csv(f"{self.results_dir}/_logFCs.tsv", sep="\t", index_col=0)

        # Copy-number corrected fold-changes
        fc_c = pd.read_csv(
            f"{self.results_dir}/_corrected_logFCs.tsv", sep="\t", index_col=0
        )

        # Quantile normalised copy-number corrected fold-changes
        fc_cq = pd.read_csv(
            f"{self.results_dir}/_qnorm_corrected_logFCs.tsv", sep="\t", index_col=0
        )

        # Scale fold-changes
        fc_s = CRISPR.scale(fc)

        # Fold-change binary
        fc_b = (fc_s < self.fc_thres).astype(int)

        return fc, fc_c, fc_cq, fc_s, fc_b

    def import_mageck_results(self):
        # Dependencies FDR
        mdep_fdr = pd.read_csv(
            f"{self.results_dir}/_MageckFDRs.tsv", sep="\t", index_col=0
        )
        mdep_bin = (mdep_fdr < self.mageck_fdr_thres).astype(int)

        return mdep_fdr, mdep_bin
