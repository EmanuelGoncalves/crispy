#!/usr/bin/env python
# Copyright (C) 2019 Emanuel Goncalves

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from crispy.CrispyPlot import CrispyPlot


class GIPlot(CrispyPlot):
    @classmethod
    def gi_regression(cls, x_gene, y_gene, plot_df):
        grid = sns.JointGrid(x_gene, y_gene, data=plot_df, space=0)

        grid.ax_joint.scatter(
            x=plot_df[x_gene],
            y=plot_df[y_gene],
            edgecolor="w",
            lw=0.1,
            s=3,
            c=cls.PAL_DBGD[0],
            alpha=1.0,
        )

        grid.plot_marginals(
            sns.distplot, kde=False, hist_kws=dict(linewidth=0), color=cls.PAL_DBGD[0]
        )

        grid.ax_joint.grid(axis="both", lw=0.1, color="#e1e1e1", zorder=0)

        plt.gcf().set_size_inches(1.5, 1.5)

        return grid

    @classmethod
    def gi_classification(cls, x_gene, y_gene, plot_df, pal=None, orient="v", stripplot=True):
        grid = sns.JointGrid(x_gene, y_gene, data=plot_df, space=0)

        sns.boxplot(
            x=x_gene,
            y=y_gene,
            notch=True,
            data=plot_df,
            boxprops=dict(linewidth=0.3),
            whiskerprops=dict(linewidth=0.3),
            medianprops=cls.MEDIANPROPS,
            flierprops=cls.FLIERPROPS,
            palette=pal,
            showcaps=False,
            sym="" if stripplot else None,
            ax=grid.ax_joint,
            saturation=1.
        )

        if stripplot:
            sns.stripplot(
                x=x_gene,
                y=y_gene,
                data=plot_df,
                dodge=True,
                jitter=.3,
                size=1.5,
                linewidth=.1,
                edgecolor="white",
                palette=pal,
                ax=grid.ax_joint,
            )

        grid.ax_joint.grid(
            axis="y" if orient == "v" else "x", lw=0.1, color="#e1e1e1", zorder=0
        )

        plt.gcf().set_size_inches(1, 1.5)

        return grid
