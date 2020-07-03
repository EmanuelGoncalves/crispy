#!/usr/bin/env python
# Copyright (C) 2019 Emanuel Goncalves

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.patches as mpatches
from natsort import natsorted
from crispy.DataImporter import Sample
from scipy.stats import gaussian_kde
from scipy.interpolate import interpn


class CrispyPlot:
    # PLOTING PROPS
    SNS_RC = {
        "axes.linewidth": 0.3,
        "xtick.major.width": 0.3,
        "ytick.major.width": 0.3,
        "xtick.major.size": 2.5,
        "ytick.major.size": 2.5,
        "xtick.minor.width": 0.3,
        "ytick.minor.width": 0.3,
        "xtick.minor.size": 1.5,
        "ytick.minor.size": 1.5,
        "xtick.direction": "out",
        "ytick.direction": "out",
    }

    # PALETTES
    PAL_DBGD = {0: "#656565", 1: "#F2C500", 2: "#E1E1E1", 3: "#0570b0"}
    PAL_SET1 = sns.color_palette("Set1", n_colors=9).as_hex()
    PAL_SET2 = sns.color_palette("Set2", n_colors=8).as_hex()
    PAL_DTRACE = {
        0: "#E1E1E1",
        1: PAL_SET2[1],
        2: "#656565",
        3: "#2b8cbe",
        4: "#de2d26",
    }

    PAL_YES_NO = {"No": "#E1E1E1", "Yes": PAL_SET2[1]}

    PAL_TISSUE = {
        "Lung": "#c50092",
        "Haematopoietic and Lymphoid": "#00ca5b",
        "Large Intestine": "#c100ce",
        "Central Nervous System": "#c3ce44",
        "Skin": "#d869ff",
        "Breast": "#496b00",
        "Head and Neck": "#ff5ff2",
        "Bone": "#004215",
        "Esophagus": "#fa006f",
        "Ovary": "#28d8fb",
        "Peripheral Nervous System": "#ff2c37",
        "Kidney": "#5ea5ff",
        "Stomach": "#fcbb3b",
        "Bladder": "#3e035f",
        "Pancreas": "#f6bc62",
        "Liver": "#df93ff",
        "Thyroid": "#ae7400",
        "Soft Tissue": "#f5aefa",
        "Cervix": "#008464",
        "Endometrium": "#980018",
        "Biliary Tract": "#535d3c",
        "Prostate": "#ff80b6",
        "Uterus": "#482800",
        "Vulva": "#ff7580",
        "Placenta": "#994800",
        "Testis": "#875960",
        "Small Intestine": "#fb8072",
        "Adrenal Gland": "#ff8155",
        "Eye": "#fa1243",
        "Other": "#000000",
    }

    PAL_TISSUE_2 = dict(
        zip(
            *(
                natsorted(list(PAL_TISSUE)),
                sns.color_palette("tab20c").as_hex()
                + sns.color_palette("tab20b").as_hex(),
            )
        )
    )

    PAL_CANCER_TYPE = dict(
        zip(
            *(
                natsorted(
                    list(Sample().samplesheet["cancer_type"].value_counts().index)
                ),
                sns.color_palette("tab20c").as_hex()
                + sns.color_palette("tab20b").as_hex()
                + sns.light_palette(PAL_SET1[1], n_colors=5).as_hex()[:-1],
            )
        )
    )
    PAL_CANCER_TYPE["Pancancer"] = PAL_SET1[5]

    PAL_MODEL_TYPE = {**PAL_CANCER_TYPE, **PAL_TISSUE_2}

    PAL_GROWTH_CONDITIONS = {
        "Adherent": "#fb8072",
        "Semi-Adherent": "#80b1d3",
        "Suspension": "#fdb462",
        "Unknown": "#d9d9d9",
    }

    PAL_MSS = {"MSS": "#d9d9d9", "MSI": "#fb8072"}

    SV_PALETTE = {
        "tandem-duplication": "#377eb8",
        "deletion": "#e41a1c",
        "translocation": "#984ea3",
        "inversion": "#4daf4a",
        "inversion_h_h": "#4daf4a",
        "inversion_t_t": "#ff7f00",
    }

    PPI_PAL = {
        "T": "#fc8d62",
        "1": "#656565",
        "2": "#7c7c7c",
        "3": "#949494",
        "4": "#ababab",
        "5+": "#c3c3c3",
        "-": "#2b8cbe",
        "X": "#2ca02c",
    }

    PPI_ORDER = ["T", "1", "2", "3", "4", "5+", "-"]

    GENESETS = ["essential", "nonessential", "nontargeting"]
    GENESETS_PAL = dict(
        essential="#e6550d", nonessential="#3182bd", nontargeting="#31a354"
    )

    # BOXPLOT PROPOS
    BOXPROPS = dict(linewidth=1.0)
    WHISKERPROPS = dict(linewidth=1.0)
    MEDIANPROPS = dict(linestyle="-", linewidth=1.0, color="red")
    FLIERPROPS = dict(
        marker="o",
        markerfacecolor="black",
        markersize=2.0,
        linestyle="none",
        markeredgecolor="none",
        alpha=0.6,
    )

    # CORRELATION PLOT PROPS
    ANNOT_KWS = dict(stat="R")
    MARGINAL_KWS = dict(kde=False, hist_kws={"linewidth": 0})

    LINE_KWS = dict(lw=1.0, color=PAL_DBGD[1], alpha=1.0)
    SCATTER_KWS = dict(edgecolor="w", lw=0.3, s=10, alpha=0.6, color=PAL_DBGD[0])
    JOINT_KWS = dict(lowess=True, scatter_kws=SCATTER_KWS, line_kws=LINE_KWS)

    @staticmethod
    def get_palette_continuous(n_colors, color=PAL_DBGD[0]):
        pal = sns.light_palette(color, n_colors=n_colors + 2).as_hex()[2:]
        return pal

    @staticmethod
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

    @classmethod
    def get_palettes(cls, samples, samplesheet):
        samples = set(samples).intersection(samplesheet.index)

        pal_tissue = {
            s: cls.PAL_TISSUE_2[samplesheet.loc[s, "tissue"]] for s in samples
        }

        pal_growth = {
            s: cls.PAL_GROWTH_CONDITIONS[samplesheet.loc[s, "growth_properties"]]
            for s in samples
        }

        palettes = pd.DataFrame(dict(tissue=pal_tissue, media=pal_growth))

        return palettes

    @classmethod
    def triu_plot(cls, x, y, color, label, **kwargs):
        df = pd.DataFrame(dict(x=x, y=y)).dropna()
        ax = plt.gca()
        ax.hexbin(
            df["x"], df["y"], cmap="Spectral_r", gridsize=30, mincnt=1, bins="log", lw=0
        )
        ax.grid(True, ls=":", lw=0.1, alpha=1.0, zorder=0)

    @classmethod
    def triu_scatter_plot(cls, x, y, color, label, **kwargs):
        df = pd.DataFrame(dict(x=x, y=y)).dropna()
        df["z"] = cls.density_interpolate(df["x"], df["y"], dtype="interpolate")
        df = df.sort_values("z")

        ax = plt.gca()

        ax.scatter(
            df["x"],
            df["y"],
            c=df["z"],
            marker="o",
            edgecolor="",
            s=5,
            alpha=0.8,
            cmap="Spectral_r",
        )

        ax.grid(True, ls=":", lw=0.1, alpha=1.0, zorder=0)

    @classmethod
    def diag_plot(cls, x, color, label, **kwargs):
        sns.distplot(
            x[~np.isnan(x)], label=label, color=CrispyPlot.PAL_DBGD[0], kde=False
        )

    @classmethod
    def attenuation_scatter(
        cls,
        x,
        y,
        plot_df,
        z="cluster",
        zorder=None,
        pal=None,
        figsize=(2.5, 2.5),
        plot_reg=True,
        ax_min=None,
        ax_max=None,
    ):
        if ax_min is None:
            ax_min = plot_df[[x, y]].min().min() * 1.1

        if ax_max is None:
            ax_max = plot_df[[x, y]].max().max() * 1.1

        if zorder is None:
            zorder = ["High", "Low"]

        if pal is None:
            pal = dict(High=cls.PAL_DTRACE[1], Low=cls.PAL_DTRACE[0])

        g = sns.jointplot(
            x,
            y,
            plot_df,
            "scatter",
            color=CrispyPlot.PAL_DTRACE[0],
            xlim=[ax_min, ax_max],
            ylim=[ax_min, ax_max],
            space=0,
            s=5,
            edgecolor="w",
            linewidth=0.0,
            marginal_kws={"hist": False, "rug": False},
            stat_func=None,
            alpha=0.1,
        )

        for n in zorder[::-1]:
            df = plot_df.query(f"{z} == '{n}'")
            g.x, g.y = df[x], df[y]

            g.plot_joint(
                sns.regplot,
                color=pal[n],
                fit_reg=False,
                scatter_kws={"s": 3, "alpha": 0.5, "linewidth": 0},
            )

            if plot_reg:
                g.plot_joint(
                    sns.kdeplot,
                    cmap=sns.light_palette(pal[n], as_cmap=True),
                    legend=False,
                    shade=False,
                    shade_lowest=False,
                    n_levels=9,
                    alpha=0.8,
                    lw=0.1,
                )

            g.plot_marginals(sns.kdeplot, color=pal[n], shade=True, legend=False)

        handles = [mpatches.Circle([0, 0], 0.25, facecolor=pal[s], label=s) for s in pal]
        g.ax_joint.legend(
            loc="upper left", handles=handles, title="Protein\nattenuation", frameon=False
        )

        g.ax_joint.grid(True, ls="-", lw=0.1, alpha=1.0, zorder=0)
        g.ax_joint.plot([ax_min, ax_max], [ax_min, ax_max], "k--", lw=0.3)

        plt.gcf().set_size_inches(figsize)

        return g


class MidpointNormalize(colors.Normalize):
    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        colors.Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y))
