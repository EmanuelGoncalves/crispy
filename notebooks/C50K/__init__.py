#!/usr/bin/env python
# Copyright (C) 2019 Emanuel Goncalves

import logging
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from crispy.Utils import Utils
from scipy.stats import ks_2samp
from scipy.interpolate import interpn

LOG = logging.getLogger("Crispy")

clib_palette = {
    "Yusa_v1.1": "#3182bd",
    "Yusa_v1": "#6baed6",
    "Sabatini_Lander_AML": "#31a354",
    "Sabatini_Lander_2015": "#74c476",
    "Sabatini_Lander_2013": "#a1d99b",
    "Brunello": "#9467bd",
    "Avana": "#ba9cd4",
    "GeCKO_2014": "#ff7f0e",
    "GeCKO_2013": "#ffa85b",
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
    "GeCKO_2014",
    "GeCKO_2013",
    "TKOv3",
    "TKO",
    "Manjunath_Wu",
    "minimal",
]


def define_sgrnas_sets(clib, fc=None, add_controls=True):
    sgrna_sets = dict()

    # sgRNA essential
    sgrnas_essential = Utils.get_essential_genes(return_series=False)
    sgrnas_essential = set(clib[clib["Gene"].isin(sgrnas_essential)].index)
    sgrnas_essential_fc = None if add_controls is None else fc.reindex(sgrnas_essential).median(1).dropna()

    sgrna_sets["essential"] = dict(
        color="#e6550d",
        sgrnas=sgrnas_essential,
        fc=sgrnas_essential_fc,
    )

    # sgRNA non-essential
    sgrnas_nonessential = Utils.get_non_essential_genes(return_series=False)
    sgrnas_nonessential = set(clib[clib["Gene"].isin(sgrnas_nonessential)].index)
    sgrnas_nonessential_fc = None if add_controls is None else fc.reindex(sgrnas_nonessential).median(1).dropna()

    sgrna_sets["nonessential"] = dict(
        color="#3182bd",
        sgrnas=sgrnas_nonessential,
        fc=sgrnas_nonessential_fc,
    )

    # sgRNA non-targeting
    if add_controls:
        sgrnas_control = {i for i in clib.index if i.startswith("CTRL0")}
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


def sgrnas_scores_scatter(df_ks, x="ks_control", y="median_fc", z=None):
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

    else:
        df = df.dropna(subset=[z])
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
