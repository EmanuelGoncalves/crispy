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
import pandas as pd
import pkg_resources
import seaborn as sns
import matplotlib.pyplot as plt
from crispy.QCPlot import QCplot
from crispy.CrispyPlot import CrispyPlot
from crispy.CRISPRData import CRISPRDataSet, Library
from minlib.Utils import define_sgrnas_sets, project_score_sample_map


LOG = logging.getLogger("Crispy")
DPATH = pkg_resources.resource_filename("crispy", "data/")
RPATH = pkg_resources.resource_filename("notebooks", "minlib/reports/")


# Project Score samples acquired with Kosuke_Yusa v1.1 library
#

ky = CRISPRDataSet("Yusa_v1.1")

ky_smap = project_score_sample_map()

ky_counts = ky.counts.remove_low_counts(ky.plasmids)

ky_fc = ky_counts.norm_rpm().norm_rpm().foldchange(ky.plasmids)

ky_gsets = define_sgrnas_sets(ky.lib, ky_fc, add_controls=True)


master_lib = Library.load_library("MasterLib_v1.csv.gz").query("Library == 'KosukeYusa'")

# sgRNAs sets AURC
#

for m in ky_gsets:
    LOG.info(f"AURC: {m}")
    ky_gsets[m]["aurc"] = pd.Series(
        {
            s: QCplot.recall_curve(ky_fc[s], index_set=ky_gsets[m]["sgrnas"])[2]
            for s in ky_fc
        }
    )

ky_gsets_aurc = pd.concat(
    [ky_gsets[m]["aurc"].rename("aurc").to_frame().assign(dtype=m) for m in ky_gsets]
)
ky_gsets_aurc.to_excel(f"{RPATH}/ky_v11_guides_aurcs.xlsx")


# sgRNA sets histograms
#

plt.figure(figsize=(2.5, 1.5))
for c in ky_gsets:
    sns.distplot(
        ky_gsets[c]["fc"],
        hist=False,
        label=c,
        kde_kws={"cut": 0, "shade": True},
        color=ky_gsets[c]["color"],
    )
plt.grid(True, ls=":", lw=0.1, alpha=1.0, zorder=0, axis="x")
plt.xlabel("sgRNAs fold-change")
plt.ylabel("Density")
plt.legend(frameon=False)
plt.savefig(f"{RPATH}/ky_v11_guides_distributions.pdf", bbox_inches="tight", transparent=True)
plt.close("all")


# sgRNAs AURC boxplots
#

pal = {s: ky_gsets[s]["color"] for s in ky_gsets}

plt.figure(figsize=(2.5, 1.))
sns.boxplot(
    "aurc",
    "dtype",
    data=ky_gsets_aurc,
    orient="h",
    palette=pal,
    saturation=1,
    showcaps=False,
    boxprops=dict(linewidth=0.3),
    whiskerprops=dict(linewidth=0.3),
    flierprops=CrispyPlot.FLIERPROPS,
    notch=True,
)
plt.axvline(0.5, ls="-", lw=0.1, zorder=0, c="k")
plt.grid(True, ls=":", lw=0.1, alpha=1.0, zorder=0, axis="x")
plt.xlabel("Area under sgRNAs recall curve")
plt.ylabel(None)
plt.savefig(f"{RPATH}/ky_v11_guides_aucs.pdf", bbox_inches="tight", transparent=True)
plt.close("all")


# Representative AROC example
#

s = "22RV1_c908R1"

plt.figure(figsize=(2., 2.), dpi=600)

for gs in ky_gsets:
    x, y, xy_auc = QCplot.recall_curve(ky_fc[s], index_set=ky_gsets[gs]["sgrnas"])
    plt.plot(x, y, label=gs, color=ky_gsets[gs]["color"])
    plt.fill_between(x, 0, y, facecolor=ky_gsets[gs]["color"], alpha=0.2)

lims = [1, 0]
plt.plot(lims, lims, "k-", lw=0.3, zorder=0)

plt.grid(True, ls=":", lw=0.1, alpha=1.0, zorder=0)

plt.xlabel("Precent-rank of sgRNAs")
plt.ylabel("Cumulative fraction")
plt.title(f"{s.split('_')[0]} (Replicate {s[-1:]})")

plt.legend(prop={'size': 4}, frameon=False)

plt.savefig(f"{RPATH}/ky_v11_guides_recall_curve_{s}.pdf", bbox_inches="tight", transparent=True)
plt.close("all")
