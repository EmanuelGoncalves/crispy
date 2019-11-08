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
}

ORGS_PALETTE = {
    "COLO-021 BME 5%": "#3182bd",
    "COLO-021 BME 80%": "#6baed6",
    "COLO-023 BME 5%": "#e6550d",
    "COLO-023 BME 80%": "#fd8d3c",
    "COLO-027 BME 5%": "#31a354",
    "COLO-027 BME 80%": "#74c476",
    "COLO-141 BME 5%": "#756bb1",
    "COLO-141 BME 80%": "#9e9ac8",
    "OESO-131 BME 5%": "#636363",
    "OESO-131 BME 80%": "#969696",
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
    f_dict["palette"] = ORGS_PALETTE[f"{f_dict['organoid']} {f_dict['medium']}"]

    samplesheet.append(f_dict)

    # Lib
    if lib is None:
        lib = f_counts["gene"]

    # Raw counts
    rawcounts.append(f_counts.drop(columns=["gene"]))

# Export samplesheet
samplesheet = pd.DataFrame(samplesheet)
samplesheet.to_excel(f"{DPATH}/samplesheet.xlsx", index=False)

# Export raw counts
rawcounts = pd.concat(rawcounts, axis=1, sort=False)
rawcounts = rawcounts.loc[:, ~rawcounts.columns.duplicated(keep="last")]
rawcounts.to_csv(f"{DPATH}/rawcounts.csv.gz", compression="gzip")
