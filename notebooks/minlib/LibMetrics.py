# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.3'
#       jupytext_version: 1.0.0
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %matplotlib inline
# %autosave 0
# %load_ext autoreload
# %autoreload 2

import re
import logging
import numpy as np
import pandas as pd
import pkg_resources
import seaborn as sns
import matplotlib.pyplot as plt
from math import sqrt
from natsort import natsorted
from crispy.QCPlot import QCplot
from scipy.stats import spearmanr
from sklearn.metrics import mean_squared_error
from crispy.CRISPRData import CRISPRDataSet, Library


LOG = logging.getLogger("Crispy")
DPATH = pkg_resources.resource_filename("crispy", "data/")
RPATH = pkg_resources.resource_filename("notebooks", "minlib/reports/")


# Master library CRISPOR scores
#

pattern = re.compile(r"\((\d+)\)")


def parse_mit_score(value):
    if value == "not_calculated;too_high":
        return 0
    elif value == "not_available":
        return np.nan
    else:
        return float(value)


mlib_crispor = Library.load_library("MasterLib_v1.1.csv.gz", set_index=False, sep="\t").query("WGE != 'not_found'")

mlib_crispor["MIT_SpecificityScore"] = [parse_mit_score(i) for i in mlib_crispor["MIT_SpecificityScore"]]

mlib_crispor["Doench_raw"] = [float(pattern.findall(i)[0]) if len(pattern.findall(i)) > 0 else np.nan for i in mlib_crispor["Doench"]]
mlib_crispor["Doench_perc"] = [float(i.split("%")[0]) if len(pattern.findall(i)) > 0 else np.nan for i in mlib_crispor["Doench"]]

mlib_crispor["CrisprScan_raw"] = [float(pattern.findall(i)[0]) if len(pattern.findall(i)) > 0 else np.nan for i in mlib_crispor["CrisprScan"]]
mlib_crispor["CrisprScan_perc"] = [float(i.split("%")[0]) if len(pattern.findall(i)) > 0 else np.nan for i in mlib_crispor["CrisprScan"]]

mlib_crispor = mlib_crispor.groupby("WGE")[["MIT_SpecificityScore", "Doench_raw", "Doench_perc", "CrisprScan_raw", "CrisprScan_perc"]].mean()


# Master library
#

mlib = Library.load_library("MasterLib_v1.csv.gz", set_index=False).dropna(subset=["WGE_ID"])
mlib["WGE_ID"] = mlib["WGE_ID"].apply(lambda v: f"{v:.0f}")

mlib["MIT_SpecificityScore"] = mlib_crispor.loc[mlib["WGE_ID"], "MIT_SpecificityScore"].values

mlib["Doench_raw"] = mlib_crispor.loc[mlib["WGE_ID"], "Doench_raw"].values
mlib["Doench_perc"] = mlib_crispor.loc[mlib["WGE_ID"], "Doench_perc"].values

mlib["CrisprScan_raw"] = mlib_crispor.loc[mlib["WGE_ID"], "CrisprScan_raw"].values
mlib["CrisprScan_perc"] = mlib_crispor.loc[mlib["WGE_ID"], "CrisprScan_perc"].values


# MinLibCas9 library information
#

minlib = Library.load_library("MinLibCas9.csv.gz", set_index=False).dropna(subset=["WGE_ID"])
minlib = minlib[[not (isinstance(i, str) and i.startswith("CTRL")) for i in minlib["WGE_ID"]]]
minlib["WGE_ID"] = minlib["WGE_ID"].apply(lambda v: f"{float(v):.0f}")

minlib["MIT_SpecificityScore"] = mlib_crispor.loc[minlib["WGE_ID"], "MIT_SpecificityScore"].values

minlib["Doench_raw"] = mlib_crispor.loc[minlib["WGE_ID"], "Doench_raw"].values
minlib["Doench_perc"] = mlib_crispor.loc[minlib["WGE_ID"], "Doench_perc"].values

minlib["CrisprScan_raw"] = mlib_crispor.loc[minlib["WGE_ID"], "CrisprScan_raw"].values
minlib["CrisprScan_perc"] = mlib_crispor.loc[minlib["WGE_ID"], "CrisprScan_perc"].values

#
#

order = ['Avana', 'Brunello', 'TKOv3', 'KosukeYusa', 'MinLibCas9']

for mtype in ["MIT_SpecificityScore", "Doench_raw", "Doench_perc", "CrisprScan_raw", "CrisprScan_perc"]:
    plot_df = pd.concat([
        mlib[["Library", mtype]],
        minlib[[mtype]].assign(Library="MinLibCas9"),
    ], ignore_index=True).dropna()

    fig, ax = plt.subplots(1, 1, figsize=(1.5, 2), dpi=600)

    QCplot.bias_boxplot(
        plot_df,
        x="Library",
        y=mtype,
        add_n=False,
        tick_base=None,
        order=order,
        ax=ax,
    )

    ax.set_xticklabels(order, rotation=45, ha='right')

    ax.grid(True, ls=":", lw=0.1, alpha=1.0, zorder=0, axis="y")

    ax.set_xlabel("")
    ax.set_ylabel(f"{mtype} score")

    plt.savefig(f"{RPATH}/lib_metrics_library_boxplot_{mtype}.pdf", bbox_inches="tight", transparent=True)
    plt.close("all")
