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

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from crispy.CrispyPlot import CrispyPlot
from minlib import rpath


# Metrics replicate correlations

rep_corrs = pd.read_csv(
    f"{rpath}/KosukeYusa_v1.1_benchmark_rep_correlation.csv.gz",
)


# Plot replicates correlation across different metrics

dtypes = rep_corrs.groupby("metric")

palette = dict(Yes="#d62728", No="#E1E1E1")

f, axs = plt.subplots(
    1,
    len(dtypes),
    sharex="all",
    sharey="row",
    figsize=(len(dtypes) * 0.5, 1.5),
    dpi=600,
)

all_rep_corr = rep_corrs.query("(metric == 'all') & (replicate == 'Yes')")["corr"].median()

for i, (dtype, df) in enumerate(dtypes):
    ax = axs[i]

    sns.boxplot(
        "replicate",
        "corr",
        data=df,
        palette=palette,
        saturation=1,
        showcaps=False,
        flierprops=CrispyPlot.FLIERPROPS,
        boxprops=dict(linewidth=0.3),
        whiskerprops=dict(linewidth=0.3),
        notch=True,
        ax=ax,
    )

    ax.axhline(all_rep_corr, ls="-", lw=1.0, zorder=0, c="#484848")

    ax.set_title(dtype)
    ax.set_ylabel(f"Replicates correlation\n(Pearson R)" if i == 0 else None)

plt.subplots_adjust(hspace=0.05, wspace=0.1)
plt.savefig(f"{rpath}/ky_v11_rep_corrs.png", bbox_inches="tight")
plt.close("all")
