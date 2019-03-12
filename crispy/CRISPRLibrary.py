#!/usr/bin/env python
# Copyright (C) 2019 Emanuel Goncalves

import os
import logging
import numpy as np
import pandas as pd
import pkg_resources
import seaborn as sns
import matplotlib.pyplot as plt


class CRISPRLibrary:
    def __init__(self):
        self.ddir = pkg_resources.resource_filename("crispy", "data/crispr_libs/")

    def load_library_sgrnas(self, verbose=0):
        lib_files = os.listdir(self.ddir)
        lib_files = filter(lambda v: not v.startswith("."), lib_files)

        sgrnas = []

        for f in lib_files:
            if verbose > 0:
                logging.getLogger("Crispy").info(f"Loading library {f}")

            flib = pd.read_csv(f"{self.ddir}/{f}")["sgRNA"]
            flib = flib.value_counts().rename("count")
            flib = flib.reset_index().rename(columns={"index": "sgRNA"})
            flib = flib.assign(lib=f.replace(".csv.gz", ""))

            sgrnas.append(flib)

        sgrnas = pd.concat(sgrnas).reset_index(drop=True)

        sgrnas = sgrnas.assign(len=sgrnas["sgRNA"].apply(len))

        return sgrnas
