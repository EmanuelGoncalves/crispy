#!/usr/bin/env python
# Copyright (C) 2020 Emanuel Goncalves

import logging
import textwrap
import numpy as np
import pandas as pd
import pkg_resources
import seaborn as sns
import matplotlib.pyplot as plt
from crispy import CrispyPlot
from crispy.GIPlot import GIPlot
from adjustText import adjust_text
from scipy.stats import mannwhitneyu
from LMModels import LModel, LMModels
from sklearn.decomposition import PCA
from DataImporter import CRISPR, Sample, PPI
from sklearn.preprocessing import MinMaxScaler


# Misc
#

LOG = logging.getLogger("Crispy")
DPATH = pkg_resources.resource_filename("notebooks", "probes/data/")
RPATH = pkg_resources.resource_filename("notebooks", "probes/reports/")


# PPI
#
ppi = PPI()
ppi_string = ppi.build_string_ppi(score_thres=900)


# DepMap CRISPR
#
ccrispr_obj = CRISPR()
ccrispr = ccrispr_obj.filter(dtype="merged")


# Probes
#
probes_targets = pd.read_csv(
    f"{DPATH}/20200505_Chemical_probes_with_Sanger_DRUG_ID_and_targets.csv", index_col=0
)
probes_targets = probes_targets.set_index(["DRUG_ID", "Probe.Name"])
probes_targets.index = [";".join(map(str, i)) for i in probes_targets.index]
probes_targets = probes_targets.dropna(subset=["Gene.Target"])

probes = pd.read_csv(
    f"{DPATH}/GDSC_chemical_probes_portal_IC50_results.csv", index_col=0
)
probes_ic50 = pd.pivot_table(
    probes, index=["drug_id", "drug_name"], columns=["model_id"], values="ln_IC50"
)
probes_ic50.index = [";".join(map(str, i)) for i in probes_ic50.index]


# probes_ic50 PCA
#
probes_ic50_fillna = probes_ic50[(probes_ic50.count(1) / probes_ic50.shape[1]) > 0.5]
probes_ic50_fillna = probes_ic50_fillna.T.fillna(probes_ic50_fillna.T.mean()).T

n_components = 10
pc_labels = [f"PC{i + 1}" for i in range(n_components)]

drug_pca = PCA(n_components=n_components).fit(probes_ic50_fillna.T)
drug_vexp = pd.Series(drug_pca.explained_variance_ratio_, index=pc_labels)
drug_pcs = pd.DataFrame(
    drug_pca.transform(probes_ic50_fillna.T),
    index=probes_ic50_fillna.columns,
    columns=pc_labels,
)

drug_pca_df = pd.concat(
    [Sample().samplesheet.reindex(index=drug_pcs.index, columns=["growth"]), drug_pcs],
    axis=1,
)

#
g = sns.clustermap(
    drug_pca_df.corr(),
    cmap="Spectral",
    annot=True,
    center=0,
    fmt=".2f",
    annot_kws=dict(size=4),
    lw=0.05,
    figsize=(3, 3),
)

plt.savefig(f"{RPATH}/probes_ic50_pca_clustermap.pdf", bbox_inches="tight", dpi=600)
plt.close("all")

#
y_var = "PC1"
g = GIPlot.gi_regression("growth", y_var, drug_pca_df, lowess=True)
g.set_axis_labels("Growth rate", f"{y_var} ({drug_vexp[y_var] * 100:.1f}%)")
plt.savefig(
    f"{RPATH}/probes_ic50_pca_regression_growth.pdf", bbox_inches="tight", dpi=600
)
plt.close("all")


# Covariates
#

covs = LMModels.define_covariates(
    institute=ccrispr_obj.merged_institute,
    medium=True,
    cancertype=False,
    tissuetype=False,
    mburden=False,
    ploidy=True,
)
# covs = pd.concat([covs, drug_pcs["PC1"]], axis=1)
covs = covs.reindex(index=set(ccrispr).intersection(probes_ic50)).dropna()


# Probes associations
#

ccrispr_lm = LModel(
    Y=ccrispr[covs.index].T, X=probes_ic50[covs.index].T, M=covs
).fit_matrix()

ccrispr_lm = pd.concat(
    [ccrispr_lm, probes_targets.reindex(ccrispr_lm["x"]).set_index(ccrispr_lm.index)],
    axis=1,
)

ccrispr_lm = ppi.ppi_annotation(
    ccrispr_lm, ppi_string, target_thres=5, y_var="y", x_var="Gene.Target", ppi_var="ppi"
)

ccrispr_lm.to_csv(
    f"{RPATH}/lm_crispr_probes_ic50.csv.gz", index=False, compression="gzip"
)


# Define subgroups
#

df_genes = set(ccrispr_lm["y"])

d_sets_name = dict(all=set(ccrispr_lm["x"]))
d_sets_name["significant"] = set(ccrispr_lm.query("fdr < .1")["x"])
d_sets_name["not_significant"] = {
    d for d in d_sets_name["all"] if d not in d_sets_name["significant"]
}
d_sets_name["annotated"] = {
    d for d in d_sets_name["all"] if d in probes_targets.index
}
d_sets_name["tested"] = {
    d
    for d in d_sets_name["annotated"]
    if len(set(probes_targets.loc[d, "Gene.Target"].split(';')).intersection(df_genes)) > 0
}
d_sets_name["tested_significant"] = {
    d for d in d_sets_name["tested"] if d in d_sets_name["significant"]
}
d_sets_name["tested_corrected"] = {
    d
    for d in d_sets_name["tested_significant"]
    if d in set(ccrispr_lm.query("(fdr < .1) & (ppi == 'T')")["x"])
}


# Volcano
#

plot_df = ccrispr_lm.query("fdr < .1")
plot_df["size"] = (
    MinMaxScaler().fit_transform(plot_df[["b"]].abs())[:, 0] * 10 + 1
)

_, ax = plt.subplots(1, 1, figsize=(3, 1.5), dpi=600)

for t, df in plot_df.groupby("ppi"):
    plt.scatter(
        -np.log10(df["pval"]),
        df["b"],
        s=df["size"],
        color=CrispyPlot.PPI_PAL[t],
        marker="o",
        label=t,
        edgecolor="white",
        lw=0.1,
        alpha=0.5,
    )

texts = [
    ax.text(-np.log10(x), y, f"{g} ~ {d}", color="k", fontsize=4)
    for i, (x, y, g, d) in plot_df.head(5)[["pval", "b", "y", "x"]].iterrows()
]
adjust_text(
    texts,
    arrowprops=dict(arrowstyle="-", color="k", alpha=0.75, lw=0.3),
    ax=ax,
)

plt.grid(True, ls="-", lw=0.1, alpha=1.0, zorder=0, axis="both")

plt.legend(
    frameon=False,
    prop={"size": 4},
    title="PPI distance",
    loc="center left",
    bbox_to_anchor=(1, 0.5),
).get_title().set_fontsize("4")

plt.ylabel("Effect size (beta)")
plt.xlabel("Association p-value (-log10)")

plt.savefig(
    f"{RPATH}/crispr_assoc_volcano.pdf", bbox_inches="tight"
)
plt.close("all")


# Top associations barplot
#

# Filter for signif associations
df = ccrispr_lm.query("fdr < .1").sort_values(["pval"])
df = df.groupby(["x", "y"]).first()
df = df.sort_values("pval").reset_index()
df = df.assign(logpval=-np.log10(df["pval"]).values)

#
ntop, n_cols = 50, 10

# Drug order
order = list(df.groupby("x")["fdr"].min().sort_values().index)[:ntop]

# Build plot dataframe
df_, xpos = [], 0
for i, drug_name in enumerate(order):
    if i % n_cols == 0:
        xpos = 0

    df_drug = df[df["x"] == drug_name].head(10)
    df_drug = df_drug.assign(xpos=np.arange(xpos, xpos + df_drug.shape[0]))
    df_drug = df_drug.assign(irow=int(np.floor(i / n_cols)))

    xpos += df_drug.shape[0] + 2

    df_.append(df_drug)

df = pd.concat(df_).reset_index()

# Plot
n_rows = int(np.ceil(ntop / n_cols))

f, axs = plt.subplots(
    n_rows,
    1,
    sharex="none", sharey="all",
    gridspec_kw=dict(hspace=0.0),
    figsize=(n_cols, n_rows * 1.7),
)

# Barplot
for irow in set(df["irow"]):
    ax = axs[irow]
    df_irow = df[df["irow"] == irow]

    for t_type, c_idx in [("ppi != 'T'", 2), ("ppi == 'T'", 1)]:
        ax.bar(
            df_irow.query(t_type)["xpos"].values,
            df_irow.query(t_type)["logpval"].values,
            color=CrispyPlot.PAL_DTRACE[c_idx],
            align="center",
            linewidth=0,
        )

    for k, v in (
        df_irow.groupby("x")["xpos"]
        .min()
        .sort_values()
        .to_dict()
        .items()
    ):
        ax.text(
            v - 1.2,
            0.1,
            textwrap.fill(k.split(" / ")[0], 15),
            va="bottom",
            fontsize=7,
            zorder=10,
            rotation="vertical",
            color=CrispyPlot.PAL_DTRACE[2],
        )

    for g, p in df_irow[["y", "xpos"]].values:
        ax.text(
            p,
            0.1,
            g,
            ha="center",
            va="bottom",
            fontsize=5,
            zorder=10,
            rotation="vertical",
            color="white",
        )

    for x, y, t, b in df_irow[["xpos", "logpval", "ppi", "b"]].values:
        c = CrispyPlot.PAL_DTRACE[1] if t == "T" else CrispyPlot.PAL_DTRACE[2]

        ax.text(
            x, y + 0.25, t, color=c, ha="center", fontsize=6, zorder=10
        )
        ax.text(
            x,
            -3,
            f"{b:.1f}",
            color=c,
            ha="center",
            fontsize=6,
            rotation="vertical",
            zorder=10,
        )

    ax.axes.get_xaxis().set_ticks([])
    ax.grid(True, ls="-", lw=0.1, alpha=1.0, zorder=0, axis="y")
    ax.set_ylabel("Drug association\n(-log10 p-value)")

plt.savefig(
    f"{RPATH}/crispr_assoc_top_barplot.pdf", bbox_inches="tight"
)
plt.close("all")


# Histogram
#
kde_kws = dict(cut=0, lw=1, zorder=1, alpha=0.8)
hist_kws = dict(alpha=0.4, zorder=1, linewidth=0)


plot_df = {
    c: ccrispr_lm.query(f"ppi {c} 'T'")["b"]
    for c in ["!=", "=="]
}

_, ax = plt.subplots(1, 1, figsize=(2, 2), dpi=600)

for c in plot_df:
    sns.distplot(
        plot_df[c],
        hist_kws=hist_kws,
        bins=30,
        kde_kws=kde_kws,
        label="Target" if c == "==" else "All",
        color=CrispyPlot.PAL_DTRACE[1] if c == "==" else CrispyPlot.PAL_DTRACE[0],
        ax=ax
    )

t, p = mannwhitneyu(plot_df["!="], plot_df["=="])
LOG.info(
    f"Mann-Whitney U statistic={t:.2f}, p-value={p:.2e}"
)

plt.grid(True, ls="-", lw=0.1, alpha=1.0, zorder=0, axis="y")

plt.xlabel("Association beta")
plt.ylabel("Density")

plt.legend(prop={"size": 6}, loc=2, frameon=False)

plt.savefig(
    f"{RPATH}/crispr_assoc_target_beta_histogram.pdf", bbox_inches="tight"
)
plt.close("all")


# Drug barplot count
#

df = ccrispr_lm.query("fdr < .1")
df = df[df["x"].isin(d_sets_name["tested_significant"])]

d_signif_ppi = []
for d in d_sets_name["tested_significant"]:
    df_ppi = df[df["x"] == d].sort_values("fdr")
    df_ppi["ppi"] = pd.Categorical(df_ppi["ppi"], CrispyPlot.PPI_ORDER)

    d_signif_ppi.append(df_ppi.sort_values("ppi").iloc[0])

d_signif_ppi = pd.DataFrame(d_signif_ppi)
d_signif_ppi["x"] = pd.Categorical(d_signif_ppi["ppi"], CrispyPlot.PPI_ORDER)
d_signif_ppi = d_signif_ppi.set_index("x").sort_values("ppi")


plot_df = d_signif_ppi["ppi"].value_counts().to_dict()
plot_df["X"] = len(
    [
        d
        for d in d_sets_name["tested"]
        if d not in d_sets_name["significant"]
    ]
)
plot_df = pd.Series(plot_df).reindex(CrispyPlot.PPI_ORDER + ["X"]).fillna(0).reset_index().rename(columns={"index": "ppi", 0: "drugs"})

_, ax = plt.subplots(1, 1, figsize=(2, 2))
sns.barplot("ppi", "drugs", data=plot_df, palette=CrispyPlot.PPI_PAL, linewidth=0, ax=ax)
for i, row in plot_df.iterrows():
    ax.text(
        i,
        row["drugs"],
        f"{(row['drugs'] / plot_df['drugs'].sum() * 100):.1f}%",
        va="bottom",
        ha="center",
        fontsize=5,
        zorder=10,
        color=CrispyPlot.PAL_DTRACE[2],
    )

ax.tick_params(axis='x', which='both', bottom=False)

ax.grid(True, ls="-", lw=0.1, alpha=1.0, zorder=0, axis="y")

ax.set_xlabel("Drug significant asociations\nshortest distance to target")
ax.set_ylabel("Number of drugs (with target in PPI)")

plt.savefig(
    f"{RPATH}/crispr_assoc_ppi_count.pdf", bbox_inches="tight"
)
plt.close("all")
