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

import logging
import numpy as np
import pandas as pd
import pkg_resources
import seaborn as sns
import matplotlib.pyplot as plt
from natsort import natsorted
from crispy.QCPlot import QCplot
from crispy.CRISPRData import Library


LOG = logging.getLogger("Crispy")
DPATH = pkg_resources.resource_filename("crispy", "data/")
RPATH = pkg_resources.resource_filename("notebooks", "minlib/reports/")


# Libraries
#

master_lib = (
    Library.load_library("MasterLib_v1.csv.gz", set_index=False)
    .dropna(subset=["WGE_Sequence"])
    .set_index(["sgRNA_ID", "Library"])
)

#
#

polyt = dict(polyt4="TTTT", polyt5="TTTTT")

# Count
polyt_df = pd.DataFrame(
    {
        i: {p: int(polyt[p] in s[:-3]) for p in polyt}
        for i, s in master_lib["WGE_Sequence"].iteritems()
    }
).T
polyt_count_lib = polyt_df.reset_index().groupby("level_1").sum()

# Start position
polyt_pos_df = pd.DataFrame(
    {
        i: {p: s.index(polyt[p]) if polyt[p] in s else np.nan for p in polyt}
        for i, s in master_lib.loc[
            polyt_df[polyt_df.sum(1) != 0].index, "WGE_Sequence"
        ].iteritems()
    }
).T


# Plot counts per library
#

plot_df = polyt_count_lib.unstack().reset_index()
plot_df.columns = ["polyt", "library", "count"]

g = sns.catplot(
    "polyt",
    "count",
    data=plot_df,
    kind="bar",
    col="library",
    color=QCplot.PAL_DBGD[0],
    sharey=True,
    height=2.5,
    aspect=0.5,
    linewidth=0,
)
for ax in g.axes[0]:
    ax.grid(True, ls=":", lw=0.1, alpha=1.0, zorder=0, axis="y")
sns.despine(top=False, right=False)
plt.savefig(f"{RPATH}/ployt_count_library.pdf", bbox_inches="tight", transparent=True)
plt.close("all")


#
#

row_order = ["KS", "JACKS", "RuleSet2"]

fig, axs = plt.subplots(
    len(row_order),
    2,
    figsize=(5, 2 * len(row_order)),
    sharey="none",
    sharex="none",
    dpi=600,
    gridspec_kw={"width_ratios": [10, 1]},
)

for i, mtype in enumerate(row_order):
    plot_df = pd.concat([polyt_pos_df, master_lib.loc[polyt_pos_df.index, mtype]], axis=1)

    for j, p in enumerate(polyt):
        ax = axs[i][j]

        order = natsorted(set(plot_df[p].dropna().astype(int)))

        QCplot.bias_boxplot(
            plot_df[plot_df["polyt5"].isnull()]
            if p == "polyt4"
            else plot_df[~plot_df["polyt5"].isnull()],
            x="polyt4",
            y=mtype,
            add_n=True,
            tick_base=None,
            order=order,
            ax=ax,
        )

        ax.set_xticklabels(
            ax.get_xticklabels() if i == (len(row_order) - 1) else [], rotation=0, horizontalalignment="right"
        )

        ax.grid(True, ls=":", lw=0.1, alpha=1.0, zorder=0, axis="y")

        if mtype == "JACKS":
            ax.axhline(1, ls="-", lw=0.3, alpha=1.0, zorder=2, c=QCplot.PAL_DBGD[1])

        ax.set_xlabel(f"Start position in sgRNA" if i == (len(row_order) - 1) else "")
        ax.set_ylabel(f"{mtype} score" if j == 0 else "")
        ax.set_title(f"{len(polyt[p])} T-stretch ({polyt[p]})" if i == 0 else "")

plt.subplots_adjust(wspace=0.1, hspace=0.05)
plt.savefig(f"{RPATH}/ployt_metrics_boxplot.pdf", bbox_inches="tight", transparent=True)
plt.close("all")
