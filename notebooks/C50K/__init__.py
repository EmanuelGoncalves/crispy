#!/usr/bin/env python
# Copyright (C) 2019 Emanuel Goncalves

import os
import logging
import numpy as np
import pandas as pd
import seaborn as sns
import pkg_resources
import matplotlib.pyplot as plt
from limix.qtl import st_scan
from crispy.Utils import Utils
from crispy.QCPlot import QCplot
from scipy.stats import ks_2samp
from scipy.interpolate import interpn
from crispy.DataImporter import Sample
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.multitest import multipletests

LOG = logging.getLogger("Crispy")

dpath = pkg_resources.resource_filename("crispy", "data/")
rpath = pkg_resources.resource_filename("notebooks", "C50K/reports/")

clib_palette = {
    "Yusa_v1.1": "#3182bd",
    "Yusa_v1": "#6baed6",
    "Sabatini_Lander_AML": "#31a354",
    "Sabatini_Lander_2015": "#74c476",
    "Sabatini_Lander_2013": "#a1d99b",
    "Brunello": "#9467bd",
    "Avana": "#ba9cd4",
    "GeCKOv2": "#ff7f0e",
    "GeCKOv1": "#ffa85b",
    "TKOv3": "#17becf",
    "TKO": "#48dceb",
    "Manjunath_Wu": "#8c564b",
    "minimal": "#d62728",
}

clib_order = [
    "Yusa_v1.1",
    "Yusa_v1",
    "Sabatini_Lander_AML",
    "Sabatini_Lander_2015",
    "Sabatini_Lander_2013",
    "Brunello",
    "Avana",
    "GeCKOv2",
    "GeCKOv1",
    "TKOv3",
    "TKO",
    "Manjunath_Wu",
    "minimal",
]


def define_sgrnas_sets(clib, fc=None, add_controls=True, dataset_name="Yusa_v1"):
    sgrna_sets = dict()

    # sgRNA essential
    sgrnas_essential = Utils.get_essential_genes(return_series=False)
    sgrnas_essential = set(clib[clib["Gene"].isin(sgrnas_essential)].index)
    sgrnas_essential_fc = (
        None
        if add_controls is None
        else fc.reindex(sgrnas_essential).median(1).dropna()
    )

    sgrna_sets["essential"] = dict(
        color="#e6550d", sgrnas=sgrnas_essential, fc=sgrnas_essential_fc
    )

    # sgRNA non-essential
    sgrnas_nonessential = Utils.get_non_essential_genes(return_series=False)
    sgrnas_nonessential = set(clib[clib["Gene"].isin(sgrnas_nonessential)].index)
    sgrnas_nonessential_fc = (
        None
        if add_controls is None
        else fc.reindex(sgrnas_nonessential).median(1).dropna()
    )

    sgrna_sets["nonessential"] = dict(
        color="#3182bd", sgrnas=sgrnas_nonessential, fc=sgrnas_nonessential_fc
    )

    # sgRNA non-targeting
    if add_controls:
        if dataset_name in ["Yusa_v1", "Yusa_v1.1", "Sabatini_Lander_AML"]:
            sgrnas_control = {i for i in clib.index if i.startswith("CTRL0")}
        else:
            sgrnas_control = set(
                clib[[i.startswith("NO_CURRENT_") for i in clib["Gene"]]].index
            )

        sgrnas_control_fc = fc.reindex(sgrnas_control).median(1).dropna()

        sgrna_sets["nontargeting"] = dict(
            color="#31a354",
            sgrnas=sgrnas_control,
            fc=None if add_controls is None else sgrnas_control_fc,
        )

    return sgrna_sets


def estimate_ks(fc, control_guides_fc, verbose=0):
    ks_sgrna = []
    for sgrna in fc.index:
        if verbose > 0:
            LOG.info(f"sgRNA={sgrna}")

        d_ctrl, p_ctrl = ks_2samp(fc.loc[sgrna], control_guides_fc)

        ks_sgrna.append({"sgrna": sgrna, "ks_control": d_ctrl, "pval_control": p_ctrl})

    ks_sgrna = pd.DataFrame(ks_sgrna).set_index("sgrna")
    ks_sgrna["median_fc"] = fc.loc[ks_sgrna.index].median(1).values

    return ks_sgrna


def sgrnas_scores_scatter(
    df_ks,
    x="ks_control",
    y="median_fc",
    z=None,
    z_color="#F2C500",
    add_cbar=True,
    ax=None,
):
    highlight_guides = False

    df = df_ks.dropna(subset=[x, y])

    if z is None:
        data, x_e, y_e = np.histogram2d(df[x], df[y], bins=20)
        z_var = interpn(
            (0.5 * (x_e[1:] + x_e[:-1]), 0.5 * (y_e[1:] + y_e[:-1])),
            data,
            np.vstack([df[x], df[y]]).T,
            method="splinef2d",
            bounds_error=False,
        )

    elif type(z) is str:
        df = df.dropna(subset=[z])
        z_var = df[z]

    elif (type(z) is set) or (type(z) is list):
        highlight_guides = True

    else:
        assert True, f"z type {type(z)} is not supported, user String, Set or List"

    if ax is None:
        plt.figure(figsize=(2.5, 2), dpi=600)
        ax = plt.gca()

    x_var, y_var = df[x], df[y]

    g = ax.scatter(
        x_var,
        y_var,
        c="#E1E1E1" if highlight_guides else z_var,
        marker="o",
        edgecolor="",
        cmap=None if highlight_guides else "Spectral_r",
        s=1,
        alpha=0.85,
    )

    if highlight_guides:
        ax.scatter(
            x_var.reindex(z),
            y_var.reindex(z),
            c=z_color,
            marker="o",
            edgecolor="",
            s=2,
            alpha=0.85,
        )

    if (not highlight_guides) and add_cbar:
        cbar = plt.colorbar(g, spacing="uniform", extend="max")
        cbar.ax.set_ylabel("Density" if z is None else z)

    ax.set_xlabel(x)
    ax.set_ylabel(y)

    return ax


def random_guides_aroc(sgrna_fc, sgrna_metric, n_guides=1, n_iterations=10):
    scores = []

    for i in range(n_iterations):
        sgrnas = sgrna_metric.groupby(sgrna_metric).apply(
            lambda x: x.sample(n=n_guides)
        )
        sgrnas = sgrnas.rename("x").reset_index().drop("x", axis=1).set_index("sgRNA")

        res = guides_aroc(sgrna_fc, sgrnas).assign(n_guides=n_guides)

        scores.append(res)

    scores = pd.concat(scores)

    return scores


def guides_aroc(sgrna_fc, sgrna_metric):
    sg_fc_sgrna = sgrna_fc.loc[sgrna_metric.index]
    sg_fc_genes = sg_fc_sgrna.groupby(sgrna_metric["Gene"]).mean()
    LOG.info(f"#(sgRNAs)={sg_fc_sgrna.shape[0]}; #(Genes)={sg_fc_genes.shape[0]}")

    ess_aroc = dict(auc={s: QCplot.pr_curve(sg_fc_genes[s]) for s in sg_fc_genes})

    res = pd.DataFrame(ess_aroc).reset_index()

    return res


def guides_aroc_benchmark(
    metrics, sgrna_counts, nguides_thres=5, rand_iterations=10, guide_set=None
):
    sgrna_counts_all = sgrna_counts.counts.remove_low_counts(sgrna_counts.plasmids)

    # Define set of guides
    guides = metrics.reindex(sgrna_counts_all.index)["Gene"].dropna()

    if guide_set is not None:
        guides = guides.reindex(guide_set).dropna()

    gene_nguides = guides.value_counts()
    gene_nguides = gene_nguides[gene_nguides >= nguides_thres]

    guides = guides[guides.isin(gene_nguides.index)]
    LOG.info(f"#(sgRNAs)={guides.shape[0]}; #(genes)={gene_nguides.shape[0]}")

    # Benchmark guides: AROC
    aroc_scores = []
    for m in metrics:
        if m != "Gene":
            LOG.info(f"Metric = {m}")

            guides_metric = metrics[[m, "Gene"]].sort_values(m)
            guides_metric = guides_metric.groupby("Gene")

            metric_aroc = []
            for n in range(1, (nguides_thres + 1)):
                guides_metric_topn = guides_metric.head(n=n)

                # Calculate fold-changes on subset
                sgrna_fc = (
                    sgrna_counts_all.loc[guides_metric_topn.index]
                    .norm_rpm()
                    .foldchange(sgrna_counts.plasmids)
                )

                metric_aroc.append(
                    guides_aroc(sgrna_fc, guides_metric_topn).assign(n_guides=n)
                )

            metric_aroc = pd.concat(metric_aroc).assign(dtype=m)

            aroc_scores.append(metric_aroc)

    # Randomised benchmark
    if rand_iterations is not None:
        rand_aroc = []
        for n in range(1, (nguides_thres + 1)):
            rand_aroc.append(
                random_guides_aroc(
                    sgrna_fc, guides, n_guides=n, n_iterations=rand_iterations
                ).assign(n_guides=n)
            )
        rand_aroc = pd.concat(rand_aroc)
        rand_aroc = (
            rand_aroc.groupby(["index", "n_guides"])
            .mean()
            .assign(dtype="random")
            .reset_index()
        )

        aroc_scores.append(rand_aroc)

    # Merge
    aroc_scores = pd.concat(aroc_scores, sort=False)

    return aroc_scores


def project_score_sample_map(lib_version="V1.1"):
    ky_v11_manifest = pd.read_csv(
        f"{dpath}/crispr_manifests/project_score_manifest.csv.gz"
    )

    s_map = []
    for i in ky_v11_manifest.index:
        s_map.append(
            pd.DataFrame(
                dict(
                    model_id=ky_v11_manifest.iloc[i]["model_id"],
                    s_ids=ky_v11_manifest.iloc[i]["library"].split(", "),
                    s_lib=ky_v11_manifest.iloc[i]["experiment_identifier"].split(", "),
                )
            )
        )
    s_map = pd.concat(s_map).set_index("s_lib")
    s_map = s_map.query(f"s_ids == '{lib_version}'")

    return s_map


def ky_v11_calculate_gene_fc(sgrna_fc, guides):
    s_map = project_score_sample_map()

    # Subset sgRNA matrix
    fc_all = sgrna_fc.loc[guides.index, sgrna_fc.columns.isin(s_map.index)]

    # Average guides across genes
    fc_all = fc_all.groupby(guides.loc[fc_all.index, "Gene"]).mean()

    # Average genes across samples
    fc_all = fc_all.groupby(s_map.loc[fc_all.columns]["model_id"], axis=1).mean()

    return fc_all


def assemble_crisprcleanr(dfolder, metric, dtype="fc"):
    files = [f for f in os.listdir(dfolder) if f.endswith(".csv")]
    files = [f for f in files if f.split("_")[1] == metric]

    if dtype == "fc":
        files = [f for f in files if f.split("_")[2] == "corrected"]
        res = pd.concat(
            [
                pd.read_csv(f"{dfolder}/{f}")["correctedFC"].rename(f.split("_")[0])
                for f in files
            ],
            axis=1,
            sort=False,
        )

    return res


def assemble_crispy(dfolder, metric, dtype="fc"):
    files = [f for f in os.listdir(dfolder) if f.endswith(".csv.gz")]
    files = [f for f in files if f.split("_")[1] == metric]

    if dtype == "fc":
        files = [f for f in files if f.split("_")[2] == "corrected"]
        res = pd.concat(
            [
                pd.read_csv(f"{dfolder}/{f}", index_col="sgrna")["corrected"].rename(
                    f.split("_")[0]
                )
                for f in files
            ],
            axis=1,
            sort=False,
        )

    return res


def lm_limix(y, x):
    # Build matrices
    Y = y.dropna()

    Y = pd.DataFrame(
        StandardScaler().fit_transform(Y), index=Y.index, columns=Y.columns
    )

    X = x.loc[Y.index]

    m = Sample().get_covariates().loc[Y.index]
    m["intercept"] = 1

    # Linear Mixed Model
    lmm = st_scan(X, Y, M=m, lik="normal", verbose=False)

    res = pd.DataFrame(
        dict(
            beta=lmm.variant_effsizes.values,
            pval=lmm.variant_pvalues.values,
            gene=X.columns,
        )
    )

    res = res.assign(phenotype=y.columns[0])
    res = res.assign(samples=Y.shape[0])

    return res


def lm_associations(Y, X):
    assocs = pd.concat([lm_limix(Y[[g]], X) for g in Y], sort=False).reset_index(
        drop=True
    )
    assocs = assocs.assign(fdr=multipletests(assocs["pval"], method="fdr_bh")[1])
    return assocs
