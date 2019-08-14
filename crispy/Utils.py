#!/usr/bin/env python
# Copyright (C) 2019 Emanuel Goncalves

import numpy as np
import pandas as pd
import pkg_resources
from scipy.stats import gaussian_kde, rankdata, norm


class Utils(object):
    DPATH = pkg_resources.resource_filename("crispy", "data/")

    CHR_SIZES_HG19 = {
        "chr1": 249_250_621,
        "chr2": 243_199_373,
        "chr3": 198_022_430,
        "chr4": 191_154_276,
        "chr5": 180_915_260,
        "chr6": 171_115_067,
        "chr7": 159_138_663,
        "chr8": 146_364_022,
        "chr9": 141_213_431,
        "chr10": 135_534_747,
        "chr11": 135_006_516,
        "chr12": 133_851_895,
        "chr13": 115_169_878,
        "chr14": 107_349_540,
        "chr15": 102_531_392,
        "chr16": 90_354_753,
        "chr17": 81_195_210,
        "chr18": 78_077_248,
        "chr19": 59_128_983,
        "chr20": 63_025_520,
        "chr21": 48_129_895,
        "chr22": 51_304_566,
        "chrX": 155_270_560,
        "chrY": 59_373_566,
    }

    BRASS_HEADERS = [
        "chr1",
        "start1",
        "end1",
        "chr2",
        "start2",
        "end2",
        "id/name",
        "brass_score",
        "strand1",
        "strand2",
        "sample",
        "svclass",
        "bkdist",
        "assembly_score",
        "readpair names",
        "readpair count",
        "bal_trans",
        "inv",
        "occL",
        "occH",
        "copynumber_flag",
        "range_blat",
        "Brass Notation",
        "non-template",
        "micro-homology",
        "assembled readnames",
        "assembled read count",
        "gene1",
        "gene_id1",
        "transcript_id1",
        "strand1",
        "end_phase1",
        "region1",
        "region_number1",
        "total_region_count1",
        "first/last1",
        "gene2",
        "gene_id2",
        "transcript_id2",
        "strand2",
        "phase2",
        "region2",
        "region_number2",
        "total_region_count2",
        "first/last2",
        "fusion_flag",
    ]

    ASCAT_HEADERS = ["chr", "start", "end", "copy_number"]

    @staticmethod
    def bin_bkdist(distance):
        """
        Discretise genomic distances

        :param distance: float

        :return:

        """
        if distance == -1:
            bin_distance = "diff. chr."

        elif distance == 0:
            bin_distance = "0"

        elif 1 < distance < 10e3:
            bin_distance = "1-10 kb"

        elif 10e3 < distance < 100e3:
            bin_distance = "10-100 kb"

        elif 100e3 < distance < 1e6:
            bin_distance = "0.1-1 Mb"

        elif 1e6 < distance < 10e6:
            bin_distance = "1-10 Mb"

        else:
            bin_distance = ">10 Mb"

        return bin_distance

    @staticmethod
    def svtype(strand1, strand2, svclass, unfold_inversions):
        if svclass == "translocation":
            svtype = "translocation"

        elif strand1 == "+" and strand2 == "+":
            svtype = "deletion"

        elif strand1 == "-" and strand2 == "-":
            svtype = "tandem-duplication"

        elif strand1 == "+" and strand2 == "-":
            svtype = "inversion_h_h" if unfold_inversions else "inversion"

        elif strand1 == "-" and strand2 == "+":
            svtype = "inversion_t_t" if unfold_inversions else "inversion"

        else:
            assert (
                False
            ), "SV class not recognised: strand1 == {}; strand2 == {}; svclass == {}".format(
                strand1, strand2, svclass
            )

        return svtype

    @staticmethod
    def bin_cnv(value, thresold):
        if not np.isfinite(value):
            return np.nan

        value = int(round(value, 0))
        value = f"{value}" if value < thresold else f"{thresold}+"

        return value

    @staticmethod
    def qnorm(x):
        y = rankdata(x)
        y = -norm.isf(y / (len(x) + 1))
        return y

    @classmethod
    def get_example_data(cls):
        raw_counts = pd.read_csv(
            "{}/{}".format(cls.DPATH, "example_rawcounts.csv"), index_col=0
        )
        copynumber = pd.read_csv("{}/{}".format(cls.DPATH, "example_copynumber.csv"))
        return raw_counts, copynumber

    @classmethod
    def get_essential_genes(
        cls, dfile="gene_sets/EssentialGenes.csv", return_series=True
    ):
        geneset = set(pd.read_csv(f"{cls.DPATH}/{dfile}", sep="\t")["gene"])

        if return_series:
            geneset = pd.Series(list(geneset)).rename("essential")

        return geneset

    @classmethod
    def get_non_essential_genes(
        cls, dfile="gene_sets/NonessentialGenes.csv", return_series=True
    ):
        geneset = set(pd.read_csv(f"{cls.DPATH}/{dfile}", sep="\t")["gene"])

        if return_series:
            geneset = pd.Series(list(geneset)).rename("non-essential")

        return geneset

    @classmethod
    def get_crispr_lib(cls, dfile="crispr_libs/Yusa_v1.1.csv.gz"):
        r_cols = dict(
            index="sgrna",
            CHRM="chr",
            STARTpos="start",
            ENDpos="end",
            GENES="gene",
            EXONE="exon",
            CODE="code",
            STRAND="strand",
        )

        lib = pd.read_csv("{}/{}".format(cls.DPATH, dfile), index_col=0).reset_index()

        lib = lib.rename(columns=r_cols)

        lib["chr"] = lib["chr"].apply(
            lambda v: f"chr{v}" if str(v) != "nan" else np.nan
        )

        return lib

    @classmethod
    def get_adam_core_essential(cls, dfile="gene_sets/pancan_core.csv"):
        return set(
            pd.read_csv(f"{cls.DPATH}/{dfile}")[
                "ADAM PanCancer Core-Fitness genes"
            ]
        )

    @classmethod
    def get_sanger_essential(cls, dfile="gene_sets/essential_sanger_depmap19.tsv"):
        return pd.read_csv(f"{cls.DPATH}/{dfile}", sep="\t")

    @classmethod
    def get_broad_core_essential(
        cls, dfile="gene_sets/pan_core_essential_broad_depmap18Q4.txt"
    ):
        return set(
            pd.read_csv("{}/{}".format(cls.DPATH, dfile))
            .iloc[:, 0]
            .apply(lambda v: v.split(" ")[0])
        )

    @classmethod
    def get_dummy_aa_vep(cls, dfile="bedit/aa_vep_jak1.csv"):
        return pd.read_csv("{}/{}".format(cls.DPATH, dfile))

    @classmethod
    def get_cytobands(cls, dfile="cytoBand.txt", chrm=None):
        cytobands = pd.read_csv("{}/{}".format(cls.DPATH, dfile), sep="\t")

        if chrm is not None:
            cytobands = cytobands[cytobands["chr"] == chrm]

        assert cytobands.shape[0] > 0, "{} not found in cytobands file"

        return cytobands

    @classmethod
    def import_brass_bedpe(cls, bedpe_file, bkdist=None, splitreads=True):
        # Import BRASS bedpe
        bedpe_df = pd.read_csv(
            bedpe_file, sep="\t", names=cls.BRASS_HEADERS, comment="#"
        )

        # Correct sample name
        bedpe_df["sample"] = bedpe_df["sample"].apply(lambda v: v.split(",")[0])

        # SV larger than threshold
        if bkdist is not None:
            bedpe_df = bedpe_df[bedpe_df["bkdist"] >= bkdist]

        # BRASS2 annotated SV
        if splitreads:
            bedpe_df = bedpe_df.query("assembly_score != '_'")

        # Parse chromosome name
        bedpe_df = bedpe_df.assign(
            chr1=bedpe_df["chr1"].apply(lambda x: "chr{}".format(x)).values
        )
        bedpe_df = bedpe_df.assign(
            chr2=bedpe_df["chr2"].apply(lambda x: "chr{}".format(x)).values
        )

        return bedpe_df

    @staticmethod
    def gkn(values, bound=1e7):
        kernel = gaussian_kde(values)
        kernel = pd.Series(
            {
                k: np.log(
                    kernel.integrate_box_1d(-bound, v)
                    / kernel.integrate_box_1d(v, bound)
                )
                for k, v in values.to_dict().items()
            }
        )
        return kernel


class DotDict(dict):
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__
