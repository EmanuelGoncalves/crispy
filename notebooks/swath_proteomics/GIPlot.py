#!/usr/bin/env python
# Copyright (C) 2019 Emanuel Goncalves

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.stats import spearmanr, pearsonr
from crispy.CrispyPlot import CrispyPlot


class GIPlot(CrispyPlot):
    MARKERS = ["o", "X"]

    @classmethod
    def gi_regression(cls, x_gene, y_gene, plot_df, style=None, lowess=False):
        grid = sns.JointGrid(x_gene, y_gene, data=plot_df, space=0)

        # Joint
        if style is not None:
            for i, (t, df) in enumerate(plot_df.groupby(style)):
                grid.ax_joint.scatter(
                    x=df[x_gene],
                    y=df[y_gene],
                    edgecolor="w",
                    lw=0.1,
                    s=3,
                    c=cls.PAL_DBGD[0],
                    alpha=1.0,
                    marker=cls.MARKERS[i],
                    label=t,
                )
        else:
            grid.ax_joint.scatter(
                x=plot_df[x_gene],
                y=plot_df[y_gene],
                edgecolor="w",
                lw=0.1,
                s=3,
                c=cls.PAL_DBGD[0],
                alpha=1.0,
            )

        grid.plot_joint(
            sns.regplot,
            data=plot_df,
            line_kws=dict(lw=1.0, color=cls.PAL_DBGD[1]),
            marker="",
            lowess=lowess,
            truncate=True,
        )

        grid.plot_marginals(
            sns.distplot, kde=False, hist_kws=dict(linewidth=0), color=cls.PAL_DBGD[0]
        )

        grid.ax_joint.grid(axis="both", lw=0.1, color="#e1e1e1", zorder=0)

        cor, pval = pearsonr(plot_df[x_gene], plot_df[y_gene])
        annot_text = f"R={cor:.2g}, p={pval:.1e}"
        grid.ax_joint.text(
            0.95,
            0.05,
            annot_text,
            fontsize=4,
            transform=grid.ax_joint.transAxes,
            ha="right",
        )

        if style is not None:
            grid.ax_joint.legend(prop=dict(size=4), frameon=False, loc=2)

        plt.gcf().set_size_inches(1.5, 1.5)

        return grid

    @staticmethod
    def _marginal_boxplot(a, xs=None, ys=None, zs=None, vertical=False, **kws):
        if vertical:
            ax = sns.boxplot(x=zs, y=ys, orient="v", **kws)
        else:
            ax = sns.boxplot(x=xs, y=zs, orient="h", **kws)

        ax.set_ylabel("")
        ax.set_xlabel("")

    @classmethod
    def gi_regression_marginal(
        cls,
        x,
        y,
        z,
        style,
        plot_df,
        scatter_kws=None,
        line_kws=None,
        legend_title="",
        discrete_pal=None,
        hue_order=None,
        annot_text=None,
        add_hline=False,
        add_vline=False,
    ):
        # Defaults
        if scatter_kws is None:
            scatter_kws = dict(edgecolor="w", lw=0.3, s=12)

        if line_kws is None:
            line_kws = dict(lw=1.0, color=cls.PAL_DBGD[0])

        #
        grid = sns.JointGrid(x, y, plot_df, space=0, ratio=8)

        grid.plot_marginals(
            cls._marginal_boxplot,
            palette=cls.PAL_DBGD if discrete_pal is None else discrete_pal,
            data=plot_df,
            linewidth=0.3,
            fliersize=1,
            notch=False,
            saturation=1.0,
            xs=x,
            ys=y,
            zs=z,
            showcaps=False,
            boxprops=cls.BOXPROPS,
            whiskerprops=cls.WHISKERPROPS,
            flierprops=cls.FLIERPROPS,
            medianprops=dict(linestyle="-", linewidth=1.0),
        )

        sns.regplot(
            x=x,
            y=y,
            data=plot_df,
            color=cls.PAL_DBGD[0],
            truncate=True,
            fit_reg=True,
            scatter=False,
            line_kws=line_kws,
            ax=grid.ax_joint,
        )

        for feature in [0, 1]:
            for i, (t, df) in enumerate(plot_df[plot_df[z] == feature].groupby(style)):
                sns.regplot(
                    x=x,
                    y=y,
                    data=df,
                    color=cls.PAL_DBGD[feature],
                    fit_reg=False,
                    scatter_kws=scatter_kws,
                    label=t if feature == 0 else None,
                    marker=cls.MARKERS[i],
                    ax=grid.ax_joint,
                )

        grid.ax_joint.legend(prop=dict(size=4), frameon=False, loc=2)

        # Annotation
        if annot_text is None:
            cor, pval = pearsonr(plot_df[x], plot_df[y])
            annot_text = f"R={cor:.2g}, p={pval:.1e}"

        grid.ax_joint.text(
            0.95,
            0.05,
            annot_text,
            fontsize=4,
            transform=grid.ax_joint.transAxes,
            ha="right",
        )

        if add_hline:
            grid.ax_joint.axhline(0, ls="-", lw=0.3, c=cls.PAL_DBGD[0], alpha=0.2)

        if add_vline:
            grid.ax_joint.axvline(0, ls="-", lw=0.3, c=cls.PAL_DBGD[0], alpha=0.2)

        if discrete_pal is None:
            handles = [
                mpatches.Circle(
                    (0.0, 0.0), 0.25, facecolor=cls.PAL_DBGD[t], label="Yes" if t else "No"
                )
                for t in [0, 1]
            ]

        elif hue_order is None:
            handles = [
                mpatches.Circle((0.0, 0.0), 0.25, facecolor=c, label=t)
                for t, c in discrete_pal.items()
            ]

        else:
            handles = [
                mpatches.Circle((0.0, 0.0), 0.25, facecolor=discrete_pal[t], label=t)
                for t in hue_order
            ]

        grid.ax_marg_y.legend(
            handles=handles,
            title=legend_title,
            loc="center left",
            bbox_to_anchor=(1, 0.5),
            frameon=False,
        )

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
