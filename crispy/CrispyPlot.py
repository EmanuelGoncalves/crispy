#!/usr/bin/env python
# Copyright (C) 2019 Emanuel Goncalves

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.colors as colors
from natsort import natsorted
from scipy.interpolate import interpn
from scipy.stats import gaussian_kde


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
    PAL_SET2 = sns.color_palette("Set2", n_colors=8).as_hex()
    PAL_DTRACE = {
        0: "#E1E1E1",
        1: PAL_SET2[1],
        2: "#656565",
        3: "#2b8cbe",
        4: "#de2d26",
    }

    PAL_YES_NO = {
        "No": "#E1E1E1",
        "Yes": PAL_SET2[1],
    }

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
        "Small Intestine": "fb8072",
        "Adrenal Gland": "#ff8155",
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

    PAL_GROWTH_CONDITIONS = {
        "Adherent": "#fb8072",
        "Semi-Adherent": "#80b1d3",
        "Suspension": "#fdb462",
        "Unknown": "#d9d9d9",
    }

    PAL_MSS = {
        "MSS": "#d9d9d9",
        "MSI": "#fb8072",
    }

    SV_PALETTE = {
        "tandem-duplication": "#377eb8",
        "deletion": "#e41a1c",
        "translocation": "#984ea3",
        "inversion": "#4daf4a",
        "inversion_h_h": "#4daf4a",
        "inversion_t_t": "#ff7f00",
    }

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


class MidpointNormalize(colors.Normalize):
    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        colors.Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y))
