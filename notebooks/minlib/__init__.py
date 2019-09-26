#!/usr/bin/env python
# Copyright (C) 2019 Emanuel Goncalves

import os
import logging
import numpy as np
import pandas as pd
import pkg_resources
import matplotlib.pyplot as plt
from limix.qtl import scan
from crispy.Utils import Utils
from crispy.QCPlot import QCplot
from scipy.stats import pearsonr
from scipy.stats import ks_2samp
from scipy.stats import gaussian_kde
from scipy.interpolate import interpn
from crispy.DataImporter import Sample
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.multitest import multipletests


def random_guides_aroc(sgrna_counts, sgrna_metric, n_guides=1, n_iterations=10):
    sgrna_counts_all = sgrna_counts.counts.remove_low_counts(sgrna_counts.plasmids)

    scores = []
    for i in range(n_iterations):
        sgrnas = pd.concat(
            [
                df.sample(n=n_guides) if df.shape[0] > n_guides else df
                for _, df in sgrna_metric.groupby("Gene")
            ]
        )

        sgrna_fc = (
            sgrna_counts_all.loc[sgrnas.index]
            .norm_rpm()
            .foldchange(sgrna_counts.plasmids)
        )

        res = guides_aroc(sgrna_fc, sgrnas).assign(n_guides=n_guides)

        scores.append(res)

    scores = pd.concat(scores)

    return scores


def guides_aroc(sgrna_fc, sgrna_metric):
    sg_fc_sgrna = sgrna_fc.loc[sgrna_metric.index]
    sg_fc_genes = sg_fc_sgrna.groupby(sgrna_metric["Gene"]).mean()
    LOG.info(f"#(sgRNAs)={sg_fc_sgrna.shape[0]}; #(Genes)={sg_fc_genes.shape[0]}")

    ess_aroc = dict(auc={s: QCplot.aroc_threshold(sg_fc_genes[s], fpr_thres=0.05)[0] for s in sg_fc_genes})

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

    LOG.info(f"#(sgRNAs)={guides.shape[0]}")

    # AROC scores
    aroc_scores = []

    # Randomised benchmark
    if rand_iterations is not None:
        rand_aroc = []

        for n in range(1, (nguides_thres + 1)):
            rand_aroc.append(
                random_guides_aroc(
                    sgrna_counts, metrics.loc[guides.index, ["Gene"]], n_guides=n, n_iterations=rand_iterations
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

    # Benchmark guides: AROC
    for m in metrics:
        if m != "Gene":
            LOG.info(f"Metric = {m}")

            guides_metric = metrics.loc[guides.index, [m, "Gene"]].sort_values(m)

            metric_aroc = []
            for n in range(1, (nguides_thres + 1)):
                guides_metric_topn = pd.concat(
                    [
                        df.head(n=n) if df.shape[0] > n else df
                        for _, df in guides_metric.groupby("Gene")
                    ]
                )

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

    # Merge
    aroc_scores = pd.concat(aroc_scores, sort=False)

    return aroc_scores

def ky_v11_calculate_gene_fc(sgrna_fc, guides):
    s_map = project_score_sample_map()

    # Subset sgRNA matrix
    fc_all = sgrna_fc.loc[guides.index, sgrna_fc.columns.isin(s_map.index)]

    # Average guides across genes
    fc_all = fc_all.groupby(guides.loc[fc_all.index, "Gene"]).mean()

    # Average genes across samples
    fc_all = fc_all.groupby(s_map.loc[fc_all.columns]["model_id"], axis=1).mean()

    return fc_all


def assemble_corrected_fc(
    dfolder, metric, n_guides, method="crispy", dtype="corrected_fc", lib=None, fextension=".csv.gz"
):
    LOG.info(f"metric={metric}; n_guides={n_guides}")

    def read_fc(fpath):
        if method == "crisprcleanr":
            return pd.read_csv(fpath, index_col=0)["correctedFC"]
        else:
            return pd.read_csv(fpath, index_col="sgrna")["corrected"]

    files = [f for f in os.listdir(dfolder) if f.endswith(fextension)]
    files = [f for f in files if f"_{metric}_" in f]
    files = [f for f in files if f.endswith(f"{dtype}{fextension}")]

    if metric != "all":
        files = [f for f in files if f"_top{n_guides}_" in f]

    if len(files) == 0:
        LOG.warning(f"No file selected: metric={metric}, n_guides={n_guides}")
        return None

    res = pd.concat(
        [read_fc(f"{dfolder}/{f}").rename(f.split("_")[0]) for f in files],
        axis=1,
        sort=False,
    )

    if lib is not None:
        res = res.groupby(lib.loc[res.index, "Gene"]).mean()

    return res


def lm_limix(y, x):
    # Build matrices
    Y = y.dropna()

    Y = pd.DataFrame(
        StandardScaler().fit_transform(Y), index=Y.index, columns=Y.columns
    )

    X = x.loc[Y.index]

    m = Sample().get_covariates().loc[Y.index]
    m = m.loc[:, m.std() > 0]
    m["intercept"] = 1

    # Linear Mixed Model
    lmm = scan(X, Y, M=m, lik="normal", verbose=False)

    effect_sizes = lmm.effsizes["h2"].query("effect_type == 'candidate'")
    effect_sizes = effect_sizes.set_index("test").drop(
        columns=["trait", "effect_type"]
    )
    pvalues = lmm.stats["pv20"].rename("pval")

    res = (
        pd.concat([effect_sizes, pvalues], axis=1, sort=False)
            .reset_index(drop=True)
    )
    res = res.assign(y_gene=y.columns[0])
    res = res.assign(samples=y.shape[0])
    res = res.assign(ncovariates=m.shape[1])

    return res.sort_values("pval")


def lm_associations(Y, X):
    assocs = pd.concat([lm_limix(Y[[g]], X) for g in Y], sort=False).reset_index(drop=True)
    assocs = assocs.assign(fdr=multipletests(assocs["pval"], method="fdr_bh")[1])
    return assocs