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
import seaborn as sns
import matplotlib.pyplot as plt
from minlib import rpath, clib_palette, clib_order


# CRISPR-Cas9 library sizes

clib_size = pd.read_excel(f"{rpath}/Human CRISPR-Cas9 dropout libraries.xlsx")


plt.figure(figsize=(2.5, 1.5), dpi=600)

sns.scatterplot(
    "Date",
    "Number of guides",
    "Library ID",
    data=clib_size,
    size="sgRNAs per Gene",
    palette=clib_palette,
    hue_order=clib_order,
    sizes=(10, 50),
)

plt.grid(True, ls="-", lw=0.1, alpha=1.0, zorder=0)

plt.xlim(datetime.date(2013, 11, 1), datetime.date(2020, 3, 1))

plt.title("CRISPR-Cas9 libraries")
plt.xlabel("Publication date")
plt.ylabel("Number of sgRNAs")

plt.legend(loc="center left", bbox_to_anchor=(1, 0.5), frameon=False, prop={"size": 4})

plt.savefig(f"{rpath}/crispr_libraries_coverage.pdf", bbox_inches="tight")
plt.close("all")
