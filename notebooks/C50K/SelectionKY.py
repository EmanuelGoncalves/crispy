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

import numpy as np
import pandas as pd
import pkg_resources
import seaborn as sns
import matplotlib.pyplot as plt
from crispy.Utils import Utils
from crispy.QCPlot import QCplot
from scipy.stats import pearsonr
from scipy.interpolate import interpn
from crispy.CrispyPlot import CrispyPlot
from crispy.CRISPRData import CRISPRDataSet
from scipy.stats import ks_2samp, spearmanr


def define_sgrnas_sets(counts, fc):
    # sgRNA essential
    sgrnas_essential = Utils.get_sanger_essential()
    sgrnas_essential = set(
        sgrnas_essential[sgrnas_essential["ADM_PanCancer_CF"]]["Gene"]
    )
    sgrnas_essential = set(counts.lib[data.lib["GENES"].isin(sgrnas_essential)].index)
    sgrnas_essential_fc = fc.reindex(sgrnas_essential).median(1).dropna()

    # sgRNA non-essential
    sgrnas_nonessential = Utils.get_non_essential_genes(return_series=False)
    sgrnas_nonessential = set(
        counts.lib[data.lib["GENES"].isin(sgrnas_nonessential)].index
    )
    sgrnas_nonessential_fc = fc.reindex(sgrnas_nonessential).median(1).dropna()

    # sgRNA non-targeting
    sgrnas_control = {i for i in counts.lib.index if i.startswith("CTRL0")}
    sgrnas_control_fc = fc.reindex(sgrnas_control).median(1).dropna()

    # Build dictionary
    sgrna_sets = dict(
        control=dict(color="#31a354", sgrnas=sgrnas_control, fc=sgrnas_control_fc),
        essential=dict(
            color="#e6550d", sgrnas=sgrnas_essential, fc=sgrnas_essential_fc
        ),
        nonessential=dict(
            color="#3182bd", sgrnas=sgrnas_nonessential, fc=sgrnas_nonessential_fc
        ),
    )

    return sgrna_sets


def sgrnas_scores_scatter(df, x="ks_control", y="jacks", z=None):
    df = df.dropna(subset=[x, y])

    if z is None:
        data, x_e, y_e = np.histogram2d(df[x], df[y], bins=20)
        z_var = interpn(
            (0.5 * (x_e[1:] + x_e[:-1]), 0.5 * (y_e[1:] + y_e[:-1])),
            data,
            np.vstack([df[x], df[y]]).T,
            method="splinef2d",
            bounds_error=False,
        )

    else:
        df = ks_sgrna.dropna(subset=[x, y, z])
        z_var = df[z]

    x_var, y_var = df[x], df[y]

    plt.figure(figsize=(2.5, 2), dpi=600)
    g = plt.scatter(
        x_var,
        y_var,
        c=z_var,
        marker="o",
        edgecolor="",
        cmap="Spectral_r",
        s=1,
        alpha=0.85,
    )

    cbar = plt.colorbar(g, spacing="uniform", extend="max")
    cbar.ax.set_ylabel("Density" if z is None else z)

    plt.xlabel(x)
    plt.ylabel(y)


def guides_aroc(sgrna_fc, sgrna_lib):
    sg_fc_sgrna = sgrna_fc.loc[sgrna_lib.index]
    sg_fc_genes = sg_fc_sgrna.groupby(sgrna_lib["gene"]).mean()
    print(f"Genes={sg_fc_genes.shape}")

    ess_aroc = dict(auc={s: QCplot.pr_curve(sg_fc_genes[s]) for s in sg_fc_genes})

    res = pd.DataFrame(ess_aroc).reset_index()

    return res


def random_guides_aroc(sgrna_fc, sgrna_lib, n_guides=1, n_iterations=10):
    scores = []

    for i in range(n_iterations):
        sgrnas = sgrna_lib.groupby("gene").apply(lambda x: x.sample(n=n_guides))
        sgrnas = sgrnas.drop("gene", axis=1).reset_index().set_index("sgrna")

        res = guides_aroc(sgrna_fc, sgrnas).assign(n_guides=n_guides)

        scores.append(res)

    scores = pd.concat(scores)

    return scores


def guides_aroc_benchmark(fc_sgrna, ks_sgrna, nguides_thres=5, rand_iterations=10):
    guides = ks_sgrna.dropna()

    gene_nguides = guides["gene"].value_counts()
    guides = guides[
        ~guides["gene"].isin(gene_nguides[gene_nguides < nguides_thres].index)
    ]

    # Random
    print(f"Metric = Random")
    aroc_scores_rand = pd.concat(
        [
            random_guides_aroc(
                fc_sgrna, guides, n_guides=n, n_iterations=rand_iterations
            )
            for n in range(1, (nguides_thres + 1))
        ]
    )
    aroc_scores_rand = (
        aroc_scores_rand.groupby(["index", "n_guides"])
        .mean()
        .assign(dtype="random")
        .reset_index()
    )

    # Jacks
    print(f"Metric = JACKS")
    jacks_metric = guides.loc[abs(guides["jacks"] - 1).sort_values().index]
    aroc_scores_jacks = pd.concat(
        [
            guides_aroc(fc_sgrna, jacks_metric.groupby("gene").head(n=n)).assign(
                n_guides=n
            )
            for n in range(1, (nguides_thres + 1))
        ]
    ).assign(dtype="jacks")

    # KS non-targeting
    print(f"Metric = KS-control")
    ks_control_metric = guides.sort_values("ks_control", ascending=False)
    aroc_scores_ks_control = pd.concat(
        [
            guides_aroc(fc_sgrna, ks_control_metric.groupby("gene").head(n=n)).assign(
                n_guides=n
            )
            for n in range(1, (nguides_thres + 1))
        ]
    ).assign(dtype="ks_control")

    # Combined metric
    print(f"Metric = Combined")
    comb_metric = guides.apply(
        lambda v: v["ks_control"] if 0.5 < v["jacks"] < 1.5 else v["ks_control"] - 0.5,
        axis=1,
    )
    comb_metric = guides.loc[comb_metric.sort_values(ascending=False).index]
    aroc_scores_comb = pd.concat(
        [
            guides_aroc(fc_sgrna, comb_metric.groupby("gene").head(n=n)).assign(
                n_guides=n
            )
            for n in range(1, (nguides_thres + 1))
        ]
    ).assign(dtype="combined")

    # Combined metric
    print(f"Metric = KS inverse")
    ks_control_inverse_metric = guides.sort_values("ks_control", ascending=True)

    aroc_scores_ksinv = pd.concat(
        [
            guides_aroc(
                fc_sgrna, ks_control_inverse_metric.groupby("gene").head(n=n)
            ).assign(n_guides=n)
            for n in range(1, (nguides_thres + 1))
        ]
    ).assign(dtype="ks_control_inverse")

    # Merge
    aroc_scores = pd.concat(
        [
            aroc_scores_rand,
            aroc_scores_jacks,
            aroc_scores_ks_control,
            aroc_scores_comb,
            aroc_scores_ksinv,
        ],
        sort=False,
    )

    return aroc_scores


def replicates_correlation(fc_sgrna, guides=None):
    if guides is not None:
        df = fc_sgrna.reindex(guides)
    else:
        df = fc_sgrna.copy()

    df.columns = [c.split("_")[0] for c in df]

    df_corr = df.corr()
    df_corr = df_corr.where(np.triu(np.ones(df_corr.shape), 1).astype(np.bool))
    df_corr = df_corr.unstack().dropna().reset_index()
    df_corr.columns = ["sample_1", "sample_2", "corr"]

    df_corr["replicate"] = (
        (df_corr["sample_1"] == df_corr["sample_2"])
        .replace({True: "Yes", False: "No"})
        .values
    )

    return df_corr


def guide_metric(ks_sgrna, alpha=0.5, beta=0.5):
    comb_metric = alpha * (1 - ks_sgrna["ks_control"]) + beta * (
        np.abs(ks_sgrna["jacks"] - 1)
    )
    return ks_sgrna.loc[comb_metric.sort_values().index]


if __name__ == "__main__":
    rpath = pkg_resources.resource_filename("notebooks", "C50K/reports/")
    dpath = pkg_resources.resource_filename("crispy", "data/")

    # - CRISPR raw counts
    data = CRISPRDataSet("Yusa_v1.1")

    fc_sgrna = (
        data.counts.remove_low_counts(data.plasmids)
        .norm_rpm()
        .foldchange(data.plasmids)
    )

    # - Sets of sgRNAs
    sgrna_sets = define_sgrnas_sets(data, fc_sgrna)

    # - JACKS efficiency scores
    jacks = pd.read_csv(f"{dpath}/jacks/Yusa_v1.0.csv", index_col=0)
    doenchroot = pd.read_csv(f"{dpath}/jacks/Yusa_v1.0_DoenchRoot.csv", index_col=0)
    forecast = pd.read_csv(
        f"{dpath}/jacks/Yusa_v1.0_ForecastFrameshift.csv", index_col=0
    )

    # - Kolmogorov-Smirnov sgRNA distribution
    ks_sgrna = []
    for sgrna in fc_sgrna.index:
        print(f"sgRNA={sgrna}")
        d_ctrl, p_ctrl = ks_2samp(fc_sgrna.loc[sgrna], sgrna_sets["control"]["fc"])

        ks_sgrna.append({"sgrna": sgrna, "ks_control": d_ctrl, "pval_control": p_ctrl})

    ks_sgrna = pd.DataFrame(ks_sgrna).set_index("sgrna")
    ks_sgrna["median_fc"] = fc_sgrna.loc[ks_sgrna.index].median(1).values
    ks_sgrna["jacks"] = jacks.reindex(ks_sgrna.index)["X1"].values
    ks_sgrna["doenchroot"] = doenchroot.reindex(ks_sgrna.index)["score_with_ppi"].values
    ks_sgrna["forecast"] = forecast.reindex(ks_sgrna.index)[
        "In Frame Percentage"
    ].values
    ks_sgrna["gene"] = data.lib.reindex(ks_sgrna.index)["GENES"].values

    # - Plot scores scatters
    for x, y in [
        ("ks_control", "jacks"),
        ("ks_control", "doenchroot"),
        ("ks_control", "forecast"),
    ]:
        sgrnas_scores_scatter(ks_sgrna, x, y)
        plt.savefig(f"{rpath}/KY_{x}_{y}.png", bbox_inches="tight", dpi=600)
        plt.close("all")

        z = "median_fc"
        sgrnas_scores_scatter(ks_sgrna, x, y, z)
        plt.savefig(f"{rpath}/KY_{x}_{y}_{z}.png", bbox_inches="tight", dpi=600)
        plt.close("all")

    # - Benchmark different metrics
    nguides_thres = 5
    aroc_scores = guides_aroc_benchmark(fc_sgrna, ks_sgrna, nguides_thres=nguides_thres)

    # Plot
    x_feature, y_features = (
        "ks_control",
        ["jacks", "random", "combined", "ks_control_inverse"],
    )

    f, axs = plt.subplots(
        4,
        nguides_thres,
        sharex="all",
        sharey="all",
        figsize=(6, len(y_features)),
        dpi=600,
    )

    x_min, x_max = aroc_scores["auc"].min(), aroc_scores["auc"].max()

    for i in range(nguides_thres):
        plot_df = aroc_scores.query(f"(n_guides == {(i+1)})")

        plot_df = pd.concat(
            [
                plot_df.query("dtype == 'ks_control'")
                .set_index("index")["auc"]
                .rename("x"),
                pd.concat(
                    [
                        plot_df.query(f"dtype == '{f}'")
                        .set_index("index")["auc"]
                        .rename(f"y{i+1}")
                        for i, f in enumerate(y_features)
                    ],
                    axis=1,
                ),
            ],
            axis=1,
        )

        for j in range(len(y_features)):
            data, x_e, y_e = np.histogram2d(plot_df["x"], plot_df[f"y{j+1}"], bins=20)
            z_var = interpn(
                (0.5 * (x_e[1:] + x_e[:-1]), 0.5 * (y_e[1:] + y_e[:-1])),
                data,
                np.vstack([plot_df["x"], plot_df[f"y{j+1}"]]).T,
                method="splinef2d",
                bounds_error=False,
            )

            axs[j][i].scatter(
                plot_df["x"],
                plot_df[f"y{j+1}"],
                c=z_var,
                marker="o",
                edgecolor="",
                cmap="Spectral_r",
                s=3,
                alpha=0.7,
            )

            lims = [x_max, x_min]
            axs[j][i].plot(lims, lims, "k-", lw=0.3, zorder=0)

        axs[0][i].set_title(f"#(guides) = {i+1}")

        if i == 0:
            for l_index, l in enumerate(y_features):
                axs[l_index][i].set_ylabel(f"{l}\n(AROC)")

        axs[len(y_features) - 1][i].set_xlabel("KS (AROC)")

    plt.subplots_adjust(hspace=0.05, wspace=0.05)
    plt.savefig(f"{rpath}/KY_guides_benchmark.png", bbox_inches="tight")
    plt.close("all")

    # - Jacks metric distribution
    f, axs = plt.subplots(1, 2, sharey="all", figsize=(4, 1.5), dpi=600)

    sns.distplot(
        ks_sgrna["jacks"],
        kde=False,
        hist_kws=dict(linewidth=0),
        color=CrispyPlot.PAL_DBGD[0],
        ax=axs[0],
        bins=50,
    )
    axs[0].set_xlabel("JACKS")
    axs[0].axvline(1, ls="-", lw=0.1, color="k")

    sns.distplot(
        np.abs(ks_sgrna["jacks"] - 1),
        kde=False,
        hist_kws=dict(linewidth=0),
        color=CrispyPlot.PAL_DBGD[0],
        ax=axs[1],
        bins=50,
    )
    axs[1].set_xlabel("|JACKS - 1|")
    axs[1].axvline(0, ls="-", lw=0.1, color="k")

    plt.subplots_adjust(hspace=0.05, wspace=0.1)
    plt.savefig(f"{rpath}/KY_jacks_distribution.png", bbox_inches="tight")
    plt.close("all")

    # - KS-control metric distribution
    f, axs = plt.subplots(1, 2, sharey="all", figsize=(3, 1.5), dpi=600)

    sns.distplot(
        ks_sgrna["ks_control"],
        kde=False,
        hist_kws=dict(linewidth=0),
        color=CrispyPlot.PAL_DBGD[0],
        ax=axs[0],
        bins=50,
    )
    axs[0].set_xlabel("KS control")
    axs[0].axvline(1, ls="-", lw=0.1, color="k")

    sns.distplot(
        1 - df["ks_control"],
        kde=False,
        hist_kws=dict(linewidth=0),
        color=CrispyPlot.PAL_DBGD[0],
        ax=axs[1],
        bins=50,
    )
    axs[1].set_xlabel("1 - KS control")
    axs[1].axvline(0, ls="-", lw=0.1, color="k")

    plt.subplots_adjust(hspace=0.05, wspace=0.1)
    plt.savefig(f"{rpath}/KY_ks_distribution.png", bbox_inches="tight")
    plt.close("all")

    # -
    guides_all = ks_sgrna.dropna(subset=["ks_control", "jacks"])

    guides_selected = guides_all.apply(
        lambda v: v["ks_control"] if 0 < v["jacks"] < 2 else v["ks_control"] - 0.5,
        axis=1,
    )
    guides_selected = guides_all.loc[guides_selected.sort_values(ascending=False).index]
    guides_selected = guides_selected.groupby("gene").head(n=2)

    corr_all = replicates_correlation(fc_sgrna, guides=list(guides_all.index))
    corr_selected = replicates_correlation(fc_sgrna, guides=list(guides_selected.index))

    #
    palette = dict(Yes=CrispyPlot.PAL_DBGD[1], No=CrispyPlot.PAL_DBGD[0])

    f, axs = plt.subplots(2, 1, sharex="all", sharey="all", figsize=(2.5, 1.5), dpi=600)

    for i, (n, df) in enumerate([("all", corr_all), ("selected n=2", corr_selected)]):
        ax = axs[i]

        sns.boxplot(
            "corr",
            "replicate",
            orient="h",
            data=df,
            palette=palette,
            saturation=1,
            showcaps=False,
            flierprops=CrispyPlot.FLIERPROPS,
            notch=True,
            ax=ax,
        )
        ax.axvline(0, ls="-", lw=0.05, zorder=0, c="#484848")
        ax.set_ylabel("Replicate")

        ax_twin = ax.twinx()
        ax_twin.set_ylabel(n.capitalize())
        ax_twin.axes.get_yaxis().set_ticks([])

    ax.set_xlabel("Pearson R")

    plt.subplots_adjust(hspace=0.05, wspace=0.05)
    plt.savefig(f"{rpath}/KY_replicates_correlation_boxplots.png", bbox_inches="tight")
    plt.close("all")

    # -
    guides = guides.dropna(subset=["jacks", "ks_control"])

    guides_arocs = pd.concat(
        [
            guides_aroc(fc_sgrna, guide_metric(guides, a, b).groupby("gene").head(n=2))
            .assign(alpha=a)
            .assign(beta=b)
            for a in np.arange(0, 1.1, 0.1)
            for b in np.arange(0, 1.1, 0.1)
        ]
    )

    #
    plot_df = pd.pivot_table(guides_arocs, index="alpha", columns="beta")

    plt.figure(figsize=(2.5, 2.5), dpi=600)

    sns.heatmap(plot_df, annot=True, cbar=False, fmt=".0f", cmap="Greys")

    plt.savefig(f"{rpath}/KY_metrics_heatmap_arocs.png", bbox_inches="tight")
    plt.close("all")
