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
from C50K import rpath


# Metrics replicate correlations

rep_corrs = pd.read_csv(
    f"{rpath}/KosukeYusa_v1.1_benchmark_rep_correlation.csv.gz",
)


# Plot replicates correlation across different metrics

metrics = ["ks_control_min", "ks_control", "median_fc", "jacks_min", "doenchroot", "doenchroot_min", "all"]

palette = dict(Yes="#d62728", No="#E1E1E1")

f, axs = plt.subplots(
    2,
    len(metrics),
    sharex="all",
    sharey="all",
    figsize=(len(metrics) * 0.9, 3),
    dpi=600,
)

all_rep_corr = rep_corrs.query("(metric == 'all') & (replicate == 'Yes')")["corr"].median()

for i, m in enumerate(metrics):
    for j, n_guides in enumerate([2, 3]):
        ax = axs[j, i]

        sns.boxplot(
            "replicate",
            "corr",
            data=rep_corrs.query(f"(metric == '{m}') & (n_guides == '{n_guides}')"),
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

        ax.set_title(m if j == 0 else "")
        ax.set_xlabel("Replicate" if j == 1 else "")
        ax.set_ylabel(f"Top {n_guides} sgRNAs\n(Pearson R)" if i == 0 else None)

plt.subplots_adjust(hspace=0.05, wspace=0.1)
plt.savefig(f"{rpath}/ky_v11_rep_corrs.png", bbox_inches="tight")
plt.close("all")
