#!/usr/bin/env python
# Copyright (C) 2019 Emanuel Goncalves

import numpy as np
import seaborn as sns
import matplotlib.colors as colors


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
        "xtick.direction": "in",
        "ytick.direction": "in",
    }

    # PALETTES
    PAL_DBGD = {0: "#656565", 1: "#F2C500", 2: "#E1E1E1"}

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

    @staticmethod
    def get_palette_continuous(n_colors, color=PAL_DBGD[0]):
        pal = sns.light_palette(color, n_colors=n_colors + 2).as_hex()[2:]
        return pal


class MidpointNormalize(colors.Normalize):
    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        colors.Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y))
