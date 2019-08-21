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

import datetime
import pandas as pd
import pkg_resources
import seaborn as sns
import matplotlib.pyplot as plt


RPATH = pkg_resources.resource_filename("notebooks", "minlib/reports/")


# CRISPR-Cas9 library sizes
#

clib_size = pd.read_excel(f"{RPATH}/LibrarySizes_Human_CRISPR_Cas9")


# Scatterplot of library sizes
#

fig, ax = plt.subplots(1, 1, figsize=(2.0, 2.0), dpi=600)

sns.scatterplot(
    "Date",
    "Number of guides",
    "Citation",
    data=clib_size,
    size="sgRNAs per Gene",
    palette=clib_size.set_index("Citation")["Palette"],
    hue_order=list(clib_size["Citation"]),
    sizes=(10, 50),
    ax=ax,
)

ax.set_xlim(datetime.date(2013, 11, 1), datetime.date(2020, 3, 1))

ax.grid(True, ls="-", lw=0.1, alpha=1.0, zorder=0)

ax.set_xlabel("Publication date")
ax.set_ylabel("Number of sgRNAs (log)")

ax.set_yscale("log")

ax.legend(loc="center left", bbox_to_anchor=(1, 0.5), frameon=False, prop={"size": 4})

plt.savefig(f"{RPATH}/LibrarySizes.pdf", bbox_inches="tight")
plt.close("all")
