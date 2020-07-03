#!/usr/bin/env python
# Copyright (C) 2019 Emanuel Goncalves

import os
import logging
import numpy as np
import pandas as pd
import pkg_resources
import seaborn as sns
import matplotlib.pyplot as plt

LOG = logging.getLogger("Crispy")
DPATH = pkg_resources.resource_filename("data", "organoids/bme2/")
RPATH = pkg_resources.resource_filename("notebooks", "bme/reports/")

CONDITION_MAP = {
    "c9R1": "BME 5%",
    "c9R2": "BME 5%",
    "c9R3": "BME 5%",
    "c9R4": "BME 80%",
    "c9R5": "BME 80%",
    "c9R6": "BME 80%",
    "c9R7": "BME 5%",
    "c9R8": "BME 5%",
    "c9R9": "BME 5%",
}

rawcount_files = [
    i for i in os.listdir(f"{DPATH}/raw_counts/") if i.endswith(".tsv.gz")
]

samplesheet, rawcounts, lib = [], [], None
for f in rawcount_files:
    LOG.info(f"Importing {f}")
    f_counts = pd.read_csv(f"{DPATH}/raw_counts/{f}", index_col=0, sep="\t")

    # Samplesheet
    f_sample = f.split(".")[0]

    f_dict = dict(
        file=f,
        id=f_counts.columns[1],
        plasmid=f_counts.columns[2],
        sample=f_sample,
        organoid="-".join(f_sample.split("_")[:-1]),
        replicate=f_sample.split("_")[-1][-2:],
        medium=CONDITION_MAP[f_sample.split("_")[-1]],
    )
    f_dict["name"] = f"{f_dict['organoid']} {f_dict['medium']} {f_dict['replicate']}"

    samplesheet.append(f_dict)

    # Lib
    if lib is None:
        lib = f_counts["gene"]

    # Raw counts
    rawcounts.append(f_counts.drop(columns=["gene"]))

# Export samplesheet
samplesheet = pd.DataFrame(samplesheet)

ORGS_PALETTE = [f"{g} {c}" for g in set(samplesheet["organoid"]) for c in ["BME 5%", "BME 80%"]]
ORGS_PALETTE = pd.Series((sns.color_palette("tab20b").as_hex() + sns.color_palette("tab20c").as_hex() + sns.color_palette("tab20").as_hex())[:len(ORGS_PALETTE)], index=ORGS_PALETTE)
samplesheet["palette"] = ORGS_PALETTE[[f"{o} {c}" for o, c in samplesheet[["organoid", "medium"]].values]].values

samplesheet.to_excel(f"{DPATH}/samplesheet.xlsx", index=False)

# Export raw counts
rawcounts = pd.concat(rawcounts, axis=1, sort=False)
rawcounts = rawcounts.loc[:, ~rawcounts.columns.duplicated(keep="last")]
rawcounts.to_csv(f"{DPATH}/rawcounts.csv.gz", compression="gzip")
