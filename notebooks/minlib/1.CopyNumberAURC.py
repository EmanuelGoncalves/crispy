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
import matplotlib.pyplot as plt
from minlib import rpath
from crispy.QCPlot import QCplot


# Copy-number AURC

ky_v11_cn_auc = pd.read_excel(f"{rpath}/KosukeYusa_v1.1_benchmark_cn.xlsx")


# Plot

f, axs = plt.subplots(3, 3, sharex="all", sharey="all", figsize=(7, 6.5), dpi=600)

for i, n_guides in enumerate(["all", "top3", "top2"]):
    for j, dtype in enumerate(["original", "ccleanr", "crispy"]):
        d_m_df = ky_v11_cn_auc.query(f"(n_guides == '{n_guides}') & (dtype == '{dtype}')")

        ax = axs[i, j]

        QCplot.bias_boxplot(
            d_m_df, x="cn", notch=True, add_n=True, n_text_y=0.05, ax=ax
        )

        ax.set_ylim(0, 1)
        ax.axhline(0.5, lw=0.3, c=QCplot.PAL_DBGD[0], ls=":", zorder=0)

        ax.set_xlabel("Copy-number" if i == 2 else None)
        ax.set_ylabel(f"AURC" if j == 0 else None)
        ax.set_title(f"{dtype} - {n_guides}")

plt.subplots_adjust(hspace=0.25, wspace=0.05)
plt.savefig(
    f"{rpath}/ky_v11_metrics_cn_aucs_boxplots.pdf", bbox_inches="tight", dpi=600
)
plt.show()
