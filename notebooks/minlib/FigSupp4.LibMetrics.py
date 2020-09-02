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

import logging
import pandas as pd
import pkg_resources
import matplotlib.pyplot as plt
from crispy.QCPlot import QCplot
from crispy.CRISPRData import Library


LOG = logging.getLogger("Crispy")
DPATH = pkg_resources.resource_filename("crispy", "data/")
RPATH = pkg_resources.resource_filename("notebooks", "minlib/reports/")


# Master library
#

mlib = Library.load_library("MasterLib_v1.csv.gz", set_index=False).dropna(
    subset=["WGE_ID"]
)
mlib["WGE_ID"] = mlib["WGE_ID"].apply(lambda v: f"{v:.0f}")


# MinLibCas9 library information
#

minlib = Library.load_library("MinLibCas9.csv.gz", set_index=False).dropna(
    subset=["WGE_ID"]
)
minlib = minlib[
    [not (isinstance(i, str) and i.startswith("CTRL")) for i in minlib["WGE_ID"]]
]


#
#

order = ["Avana", "Brunello", "TKOv3", "KosukeYusa", "MinLibCas9"]

for mtype in ["CRISPOR_MIT_SpecificityScore", "CRISPOR_Doench", "CRISPOR_CrisprScan"]:
    plot_df = pd.concat(
        [mlib[["Library", mtype]], minlib[[mtype]].assign(Library="MinLibCas9")],
        ignore_index=True,
    ).dropna()

    fig, ax = plt.subplots(1, 1, figsize=(1.5, 2), dpi=600)

    QCplot.bias_boxplot(
        plot_df,
        x="Library",
        y=mtype,
        add_n=False,
        tick_base=None,
        order=order,
        draw_violin=True,
        ax=ax,
    )

    ax.set_xticklabels(order, rotation=45, ha="right")

    ax.grid(True, ls=":", lw=0.1, alpha=1.0, zorder=0, axis="y")

    ax.set_xlabel("")
    ax.set_ylabel(f"{mtype.split('_')[1]}")

    plt.savefig(
        f"{RPATH}/lib_metrics_library_boxplot_{mtype}.pdf",
        bbox_inches="tight",
        transparent=True,
    )
    plt.close("all")
