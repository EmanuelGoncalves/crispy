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

DATASETS = dict(
    Yusa_v1=dict(
        read_counts="Yusa_v1_Score_readcount.csv.gz",
        library="Yusa_v1.csv.gz",
        plasmids=["ERS717283.plasmid"]
    ),
)


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
    def __init__(self, raw_counts):
        if type(raw_counts) == str:
            counts = pd.read_csv(raw_counts, index_col=0)

        elif type(raw_counts) == pd.DataFrame:
            counts = raw_counts

        else:
            assert False, "Only str (file path) and pandas.DataFrame types supported"

        super().__init__(counts)

    def norm_mean(self):
        factors = self.sum().mean() / self.sum()
        return factors

    def norm_total(self, scale=1e7):
        factors = self.sum() * scale
        return factors

    def norm_gmean(self):
        sgrna_gmean = st.gmean(self, axis=1)
        factors = self.divide(sgrna_gmean, axis=0).median()
        return factors


class CRISPRDataSet(ReadCounts):
    def __init__(self, dataset_name):
        assert dataset_name in DATASETS, f"CRISPR data-set {dataset_name} not supported: {DATASETS}"

        super().__init__(f"{DATA_DIR}/{DATASETS[dataset_name]['read_counts']}")

        self.clib = Library.load_library(DATASETS[dataset_name]["library"])
        self.plasmids = DATASETS[dataset_name]["plasmids"]

        LOG.info(f"#(sgRNAs)={self.clib.shape[0]}")
        LOG.info(f"#(samples)={self.shape[1]}")
