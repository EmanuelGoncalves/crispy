#!/usr/bin/env python
# Copyright (C) 2019 Emanuel Goncalves

import logging
import numpy as np
import pandas as pd
import pkg_resources
from crispy.Utils import Utils
from crispy.QCPlot import QCplot
from scipy.stats import ks_2samp
from scipy.interpolate import interpn
from scipy.stats import gaussian_kde


LOG = logging.getLogger("Crispy")
DPATH = pkg_resources.resource_filename("crispy", "data/")


def downsample_sgrnas(
    counts,
    lib,
    manifest,
    n_guides_thresholds,
    n_iters=10,
    plasmids=None,
    gene_col="Gene",
):
    plasmids = ["CRISPR_C6596666.sample"] if plasmids is None else plasmids

    # Guides library
    glib = (
        lib[lib.index.isin(counts.index)].reset_index()[["sgRNA_ID", gene_col]].dropna()
    )
    glib = glib.groupby(gene_col)

    scores = []
    for n_guides in n_guides_thresholds:
        for iteration in range(n_iters):
            LOG.info(f"Number of sgRNAs: {n_guides}; Iteration: {iteration + 1}")

            # Downsample randomly guides per genes
            sgrnas = pd.concat(
                [d.sample(n=n_guides) if d.shape[0] > n_guides else d for _, d in glib]
            )

            # sgRNAs fold-change
            sgrnas_fc = counts.norm_rpm().foldchange(plasmids)

            # genes fold-change
            genes_fc = sgrnas_fc.groupby(sgrnas.set_index("sgRNA_ID")[gene_col]).mean()
            genes_fc = genes_fc.groupby(manifest["model_id"], axis=1).mean()

            # AROC of 1% FDR
            res = pd.DataFrame(
                [
                    dict(sample=s, aroc=QCplot.aroc_threshold(genes_fc[s])[0])
                    for s in genes_fc
                ]
            ).assign(n_guides=n_guides)
            scores.append(res)

    return pd.concat(scores)


def define_sgrnas_sets(clib, fc=None, add_controls=True, dataset_name="Yusa_v1"):
    sgrna_sets = dict()

    # sgRNA essential
    sgrnas_essential = Utils.get_essential_genes(return_series=False)
    sgrnas_essential = set(clib[clib["Gene"].isin(sgrnas_essential)].index)
    sgrnas_essential_fc = (
        None if fc is None else fc.reindex(sgrnas_essential).median(1).dropna()
    )

    sgrna_sets["essential"] = dict(
        color="#e6550d", sgrnas=sgrnas_essential, fc=sgrnas_essential_fc
    )

    # sgRNA non-essential
    sgrnas_nonessential = Utils.get_non_essential_genes(return_series=False)
    sgrnas_nonessential = set(clib[clib["Gene"].isin(sgrnas_nonessential)].index)
    sgrnas_nonessential_fc = (
        None if fc is None else fc.reindex(sgrnas_nonessential).median(1).dropna()
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
            fc=None if fc is None else sgrnas_control_fc,
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


def project_score_sample_map(lib_version="V1.1"):
    ky_v11_manifest = pd.read_csv(
        f"{DPATH}/crispr_manifests/project_score_manifest.csv.gz"
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


def density_interpolate(xx, yy, dtype="gaussian"):
    if dtype == "gaussian":
        xy = np.vstack([xx, yy])
        zz = gaussian_kde(xy)(xy)

    else:
        data, x_e, y_e = np.histogram2d(xx, yy, bins=20)

        zz = interpn(
            (0.5 * (x_e[1:] + x_e[:-1]), 0.5 * (y_e[1:] + y_e[:-1])),
            data,
            np.vstack([xx, yy]).T,
            method="splinef2d",
            bounds_error=False,
        )

    return zz
