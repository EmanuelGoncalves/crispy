#!/usr/bin/env python
# Copyright (C) 2018 Emanuel Goncalves

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


class CrispyPlot:
    # PLOTING PROPS
    SNS_RC = {
        'axes.linewidth': .3,
        'xtick.major.width': .3, 'ytick.major.width': .3,
        'xtick.major.size': 2., 'ytick.major.size': 2.,
        'xtick.direction': 'out', 'ytick.direction': 'out'
    }

    # PALETTES
    PAL_DBGD = {0: '#656565', 1: '#F2C500', 2: '#E1E1E1'}

    SV_PALETTE = {
        'tandem-duplication': '#377eb8',
        'deletion': '#e41a1c',
        'translocation': '#984ea3',
        'inversion': '#4daf4a',
        'inversion_h_h': '#4daf4a',
        'inversion_t_t': '#ff7f00',
    }

    # BOXPLOT PROPOS
    FLIERPROPS = dict(
        marker='o', markerfacecolor='black', markersize=2., linestyle='none', markeredgecolor='none', alpha=.6
    )
    MEDIANPROPS = dict(linestyle='-', linewidth=1., color='red')
    BOXPROPS = dict(linewidth=1.)
    WHISKERPROPS = dict(linewidth=1.)

    @staticmethod
    def get_palette_continuous(n_colors, color=PAL_DBGD[0]):
        pal = sns.light_palette(color, n_colors=n_colors + 2).as_hex()[2:]
        return pal
