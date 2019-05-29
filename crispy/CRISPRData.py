#!/usr/bin/env python
# Copyright (C) 2019 Emanuel Goncalves

import os
import logging
import numpy as np
import pandas as pd
import pkg_resources
import seaborn as sns
import scipy.stats as st
import matplotlib.pyplot as plt
from pandas import DataFrame
from crispy.Utils import Utils


LOG = logging.getLogger("Crispy")

DATA_DIR = pkg_resources.resource_filename("crispy", "data/crispr_rawcounts/")
LIBS_DIR = pkg_resources.resource_filename("crispy", "data/crispr_libs/")
MANIFESTS_DIR = pkg_resources.resource_filename("crispy", "data/crispr_manifests/")

DATASETS = {
    "Yusa_v1": dict(
        name="Yusa v1",
        read_counts="Yusa_v1_Score_readcount.csv.gz",
        library="Yusa_v1.csv.gz",
        plasmids=["ERS717283.plasmid"],
    ),
    "Yusa_v1.1": dict(
        name="Yusa v1.1",
        read_counts="Yusa_v1.1_Score_readcount.csv.gz",
        library="Yusa_v1.1.csv.gz",
        plasmids=["CRISPR_C6596666.sample"],
    ),
    "GeCKOv2": dict(
        name="GeCKO v2",
        read_counts="GeCKO2_Achilles_v3.3.8_readcounts.csv.gz",
        library="GeCKO_v2.csv.gz",
        plasmids=["pDNA_pXPR003_120K_20140624"],
        exclude_guides=set(
            pd.read_csv(
                f"{MANIFESTS_DIR}/GeCKO2_Achilles_v3.3.8_dropped_guides.csv.gz"
            )["sgRNA"]
        ),
    ),
    "Avana": dict(
        name="Avana DepMap19Q2",
        read_counts="Avana_DepMap19Q2_readcount.csv.gz",
        library="Avana_v1.csv.gz",
        plasmids=pd.read_csv(
            f"{MANIFESTS_DIR}/Avana_DepMap19Q2_sample_map.csv.gz", index_col="sample"
        )["controls"]
        .apply(lambda v: v.split(";"))
        .to_dict(),
        exclude_guides=set(
            pd.read_csv(f"{MANIFESTS_DIR}/Avana_DepMap19Q2_dropped_guides.csv.gz")[
                "guide"
            ]
        ),
    ),
    "Sabatini_Lander_AML": dict(
        name="Sabatini Lander AML",
        read_counts="Sabatini_Lander_v2_AML_readcounts.csv.gz",
        library="Sabatini_Lander_v3.csv.gz",
        plasmids={
            "P31/FUJ-final": ["P31/FUJ-initial"],
            "NB4 (replicate A)-final": ["NB4-initial"],
            "NB4 (replicate B)-final": ["NB4-initial"],
            "OCI-AML2-final": ["OCI-AML2-initial"],
            "OCI-AML3-final": ["OCI-AML3-initial"],
            "SKM-1-final": ["SKM-1-initial"],
            "EOL-1-final": ["EOL-1-initial"],
            "HEL-final": ["HEL-initial"],
            "Molm-13-final": ["Molm-13-initial"],
            "MonoMac1-final": ["MonoMac1-initial"],
            "MV4;11-final": ["MV4;11-initial"],
            "PL-21-final": ["PL-21-initial"],
            "OCI-AML5-final": ["OCI-AML5-initial"],
        },
    ),
    "DepMap19Q1": dict(
        name="DepMap19Q1",
        read_counts="Avana_DepMap19Q1_readcount.csv.gz",
        library="Avana_v1.csv.gz",
        plasmids=[
            "pDNA Avana4_010115_1.5Ex_batch1",
            "pDNA Avana4_060115_1.5Ex_batch1",
            "pDNA Avana4_0101215_0.55Ex_batch1",
            "pDNA Avana4_060115_0.55Ex_batch1",
            "pDNA Avana4_010115_1.5Ex_batch0",
            "pDNA Avana4_060115_1.5Ex_batch0",
            "Avana 4+ Hu pDNA (M-AA40, 9/30/15)_batch3",
            "Avana 4+ Hu pDNA (M-AA40, 9/30/15) (0.2pg/uL)_batch3",
            "Avana4pDNA20160601-311cas9 RepG09_batch2",
            "Avana4pDNA20160601-311cas9 RepG10_batch2",
            "Avana4pDNA20160601-311cas9 RepG11_batch2",
            "Avana4pDNA20160601-311cas9 RepG12_batch2",
        ],
    ),
    "Brunello_A375": dict(
        name="Brunello A375",
        read_counts="Brunello_A375_readcount.csv.gz",
        library="Brunello_v1.csv.gz",
        plasmids={
            "orig_tracr_A375_RepA": ["orig_tracr_A375_pDNA"],
            "orig_tracr_A375_RepB": ["orig_tracr_A375_pDNA"],
            "mod_tracr_A375_RepA": ["mod_tracr_A375_pDNA"],
            "mod_tracr_A375_RepB": ["mod_tracr_A375_pDNA"],
            "mod_tracr_A375_RepC": ["mod_tracr_A375_pDNA"],
        },
    ),
}


class Library:
    @staticmethod
    def load_library(lib_file, set_index=True):
        lib_path = f"{LIBS_DIR}/{lib_file}"

        assert os.path.exists(lib_path), f"CRISPR library {lib_file} not supported"

        clib = pd.read_csv(lib_path)

        if set_index:
            clib = clib.set_index("sgRNA_ID")

        return clib

    @staticmethod
    def load_library_sgrnas(verbose=0):
        lib_files = os.listdir(LIBS_DIR)
        lib_files = filter(lambda v: not v.startswith("."), lib_files)

        sgrnas = []

        for f in lib_files:
            if verbose > 0:
                LOG.info(f"Loading library {f}")

            flib = pd.read_csv(f"{LIBS_DIR}/{f}")["sgRNA"]
            flib = flib.value_counts().rename("count")
            flib = flib.reset_index().rename(columns={"index": "sgRNA"})
            flib = flib.assign(lib=f.replace(".csv.gz", ""))

            sgrnas.append(flib)

        sgrnas = pd.concat(sgrnas).reset_index(drop=True)

        sgrnas = sgrnas.assign(len=sgrnas["sgRNA"].apply(len))

        return sgrnas


class ReadCounts(DataFrame):
    PSEUDO_COUNT = 1

    def __init__(self, data=None, index=None, columns=None, dtype=None, copy=False):
        super().__init__(data, index, columns, dtype, copy)

    @property
    def _constructor(self):
        return ReadCounts

    def norm_rpm(self, scale=1e6):
        factors = self.sum() / scale
        return self.divide(factors)

    def norm_mean(self):
        factors = self.sum().mean() / self.sum()
        return self.divide(factors)

    def norm_gmean(self):
        sgrna_gmean = st.gmean(self, axis=1)
        factors = self.divide(sgrna_gmean, axis=0).median()
        return self.divide(factors)

    def foldchange(self, controls):
        if type(controls) == dict:
            fc = self.add(self.PSEUDO_COUNT)
            fc = pd.DataFrame(
                {c: fc[c].divide(fc[controls[c]].mean(1), axis=0) for c in controls if c in fc}
            )
            fc = fc.apply(np.log2)

        else:
            fc = (
                self.add(self.PSEUDO_COUNT)
                .divide(self[controls].mean(1), axis=0)
                .drop(controls, axis=1)
                .apply(np.log2)
            )

        return fc

    def remove_low_counts(self, controls, counts_threshold=30):
        return self[self[controls].mean(1) >= counts_threshold]

    def scale(self, essential=None, non_essential=None, metric=np.median):
        if essential is None:
            essential = Utils.get_essential_genes(return_series=False)

        if non_essential is None:
            non_essential = Utils.get_non_essential_genes(return_series=False)

        assert (
            len(essential.intersection(self.index)) != 0
        ), "DataFrame has no index overlapping with essential list"

        assert (
            len(non_essential.intersection(self.index)) != 0
        ), "DataFrame has no index overlapping with non essential list"

        essential_metric = metric(self.reindex(essential).dropna(), axis=0)
        non_essential_metric = metric(self.reindex(non_essential).dropna(), axis=0)

        return self.subtract(non_essential_metric).divide(
            non_essential_metric - essential_metric
        )


class CRISPRDataSet:
    def __init__(self, dataset, ddir=None, exclude_samples=None, exclude_guides=None):
        # Load data-set dict
        if type(dataset) is dict:
            self.dataset_dict = dataset

        else:
            assert (
                dataset in DATASETS
            ), f"CRISPR data-set {dataset} not supported: {DATASETS}"

            self.dataset_dict = DATASETS[dataset]

        self.ddir = DATA_DIR if ddir is None else ddir

        # Build object arguments
        self.plasmids = self.dataset_dict["plasmids"]

        self.lib = Library.load_library(self.dataset_dict["library"])

        data = pd.read_csv(
            f"{self.ddir}/{self.dataset_dict['read_counts']}", index_col=0
        )

        # Drop excluded samples
        if exclude_samples is not None:
            data = data.drop(exclude_samples, axis=1, errors="ignore")

        # Drop excluded guides
        if exclude_guides is not None:
            data = data.drop(exclude_guides, axis=0, errors="ignore")
            self.lib = self.lib.drop(exclude_guides, axis=0, errors="ignore")

        elif "exclude_guides" in self.dataset_dict:
            data = data.drop(
                self.dataset_dict["exclude_guides"], axis=0, errors="ignore"
            )
            self.lib = self.lib.drop(
                self.dataset_dict["exclude_guides"], axis=0, errors="ignore"
            )

        self.counts = ReadCounts(data=data)

        LOG.info(f"#(sgRNAs)={self.lib.shape[0]}")
        LOG.info(f"#(samples)={self.counts.shape[1]}")

    def get_plasmids_counts(self):
        return self.counts[self.plasmids]
