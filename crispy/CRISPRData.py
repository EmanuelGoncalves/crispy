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


LOG = logging.getLogger("Crispy")

DATA_DIR = pkg_resources.resource_filename("crispy", "data/crispr_rawcounts/")
LIBS_DIR = pkg_resources.resource_filename("crispy", "data/crispr_libs/")

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

    def __init__(
        self,
        data=None,
        index=None,
        columns=None,
        dtype=None,
        copy=False,
    ):
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
        return (
            self.add(self.PSEUDO_COUNT)
            .divide(self[controls].mean(1), axis=0)
            .drop(controls, axis=1)
            .apply(np.log2)
        )

    def remove_low_counts(self, controls, counts_threshold=30):
        return self[self[controls].mean(1) >= counts_threshold]


class CRISPRDataSet:
    def __init__(self, dataset_name):
        self.name = dataset_name

        assert (
            self.name in DATASETS
        ), f"CRISPR data-set {self.name} not supported: {DATASETS}"

        self.plasmids = DATASETS[self.name]["plasmids"]

        self.lib = Library.load_library(DATASETS[self.name]["library"])

        data = pd.read_csv(
            f"{DATA_DIR}/{DATASETS[self.name]['read_counts']}", index_col=0
        )
        self.counts = ReadCounts(data=data)

        LOG.info(f"#(sgRNAs)={self.lib.shape[0]}")
        LOG.info(f"#(samples)={self.counts.shape[1]}")

    def get_plasmids_counts(self):
        return self.counts[self.plasmids]