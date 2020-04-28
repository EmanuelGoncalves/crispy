#!/usr/bin/env python
# Copyright (C) 2019 Emanuel Goncalves

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from crispy.CrispyPlot import CrispyPlot


def pc_labels(n):
    return [f"PC{i}" for i in np.arange(1, n + 1)]


def dim_reduction_pca(df, pca_ncomps=10):
    df_pca = PCA(n_components=pca_ncomps)

    df_pcs = df_pca.fit_transform(df.T)
    df_pcs = pd.DataFrame(df_pcs, index=df.T.index, columns=pc_labels(pca_ncomps))

    df_vexp = pd.Series(df_pca.explained_variance_ratio_, index=df_pcs.columns)

    return df_pcs, df_vexp


def dim_reduction(
    df,
    pca_ncomps=50,
    tsne_ncomps=2,
    perplexity=30.0,
    early_exaggeration=12.0,
    learning_rate=200.0,
    n_iter=1000,
):
    # PCA
    df_pca = dim_reduction_pca(df, pca_ncomps)[0]

    # tSNE
    df_tsne = TSNE(
        n_components=tsne_ncomps,
        perplexity=perplexity,
        early_exaggeration=early_exaggeration,
        learning_rate=learning_rate,
        n_iter=n_iter,
    ).fit_transform(df_pca)
    df_tsne = pd.DataFrame(df_tsne, index=df_pca.index, columns=pc_labels(tsne_ncomps))

    return df_tsne, df_pca


def plot_dim_reduction(data, palette=None, ctype="tSNE", hue_filed="tissue"):
    if hue_filed not in data.columns:
        data = data.assign(tissue="All")

    if palette is None:
        palette = dict(All=CrispyPlot.PAL_DBGD[0])

    fig, ax = plt.subplots(1, 1, figsize=(4.0, 4.0), dpi=600)

    for t, df in data.groupby(hue_filed):
        ax.scatter(
            df["PC1"],
            df["PC2"],
            c=palette[t],
            marker="o",
            edgecolor="",
            s=5,
            label=t,
            alpha=0.8,
        )
    ax.set_xlabel("Dimension 1" if ctype == "tSNE" else "PC 1")
    ax.set_ylabel("Dimension 2" if ctype == "tSNE" else "PC 2")
    ax.axis("off" if ctype == "tSNE" else "on")

    if ctype == "pca":
        ax.grid(True, ls="-", lw=0.1, alpha=1.0, zorder=0)

    ax.legend(
        loc="center left",
        bbox_to_anchor=(1, 0.5),
        prop={"size": 4},
        frameon=False,
        title=hue_filed,
    ).get_title().set_fontsize("5")

    return ax
