#!/usr/bin/env python
# Copyright (C) 2019 Emanuel Goncalves

import numpy as np
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from natsort import natsorted
from scipy.stats import pearsonr
from adjustText import adjust_text
from crispy.CrispyPlot import CrispyPlot, MidpointNormalize


class GIPlot(CrispyPlot):
    MARKERS = ["o", "X"]

    @classmethod
    def gi_regression(
        cls, x_gene, y_gene, plot_df, style=None, lowess=False, palette=None
    ):
        pal = cls.PAL_DTRACE if palette is None else palette

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
                    c=pal[2],
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
                c=pal[2],
                alpha=1.0,
            )

        grid.plot_joint(
            sns.regplot,
            data=plot_df,
            line_kws=dict(lw=1.0, color=pal[1]),
            marker="",
            lowess=lowess,
            truncate=True,
        )

        grid.plot_marginals(
            sns.distplot, kde=False, hist_kws=dict(linewidth=0), color=pal[2]
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
    def _marginal_boxplot(_, xs=None, ys=None, zs=None, vertical=False, **kws):
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
        plot_df,
        style=None,
        scatter_kws=None,
        line_kws=None,
        legend_title=None,
        discrete_pal=None,
        hue_order=None,
        annot_text=None,
        add_hline=False,
        add_vline=False,
    ):
        # Defaults
        if scatter_kws is None:
            scatter_kws = dict(edgecolor="w", lw=0.3, s=8)

        if line_kws is None:
            line_kws = dict(lw=1.0, color=cls.PAL_DBGD[0])

        if discrete_pal is None:
            discrete_pal = cls.PAL_DTRACE

        if hue_order is None:
            hue_order = natsorted(set(plot_df[z]))

        #
        grid = sns.JointGrid(x, y, plot_df, space=0, ratio=8)

        grid.plot_marginals(
            cls._marginal_boxplot,
            palette=discrete_pal,
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
            color=discrete_pal[0],
            truncate=True,
            fit_reg=True,
            scatter=False,
            line_kws=line_kws,
            ax=grid.ax_joint,
        )

        for feature in hue_order:
            dfs = plot_df[plot_df[z] == feature]
            dfs = (
                dfs.assign(style=1).groupby("style")
                if style is None
                else dfs.groupby(style)
            )

            for i, (mtype, df) in enumerate(dfs):
                sns.regplot(
                    x=x,
                    y=y,
                    data=df,
                    color=discrete_pal[feature],
                    fit_reg=False,
                    scatter_kws=scatter_kws,
                    label=mtype if i > 0 else None,
                    marker=cls.MARKERS[i],
                    ax=grid.ax_joint,
                )

        if style is not None:
            grid.ax_joint.legend(prop=dict(size=4), frameon=False, loc=2)

        # Annotation
        if annot_text is None:
            df_corr = plot_df.dropna(subset=[x, y, z])
            cor, pval = pearsonr(df_corr[x], df_corr[y])
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

        handles = [
            mpatches.Circle(
                (0.0, 0.0),
                0.25,
                facecolor=discrete_pal[t],
                label=f"{t} (N={(plot_df[z] == t).sum()})",
            )
            for t in hue_order
        ]

        grid.ax_marg_y.legend(
            handles=handles,
            title=z if legend_title is None else legend_title,
            loc="center left",
            bbox_to_anchor=(1, 0.5),
            frameon=False,
        )

        grid.ax_joint.grid(axis="both", lw=0.1, color="#e1e1e1", zorder=0)

        plt.gcf().set_size_inches(1.5, 1.5)

        return grid

    @classmethod
    def gi_classification(
        cls, x_gene, y_gene, plot_df, palette=None, orient="v", stripplot=True
    ):
        pal = cls.PAL_DTRACE if palette is None else palette

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
            saturation=1.0,
        )

        if stripplot:
            sns.stripplot(
                x=x_gene,
                y=y_gene,
                data=plot_df,
                dodge=True,
                jitter=0.3,
                size=1.5,
                linewidth=0.1,
                edgecolor="white",
                palette=pal,
                ax=grid.ax_joint,
            )

        grid.ax_joint.grid(
            axis="x" if orient == "v" else "x", lw=0.1, color="#e1e1e1", zorder=0
        )

        plt.gcf().set_size_inches(1, 1.5)

        return grid

    @classmethod
    def gi_tissue_plot(cls, x, y, plot_df):
        fig, ax = plt.subplots(1, 1, figsize=(2, 2), dpi=600)

        for t, df in plot_df.groupby("tissue"):
            ax.scatter(
                df[x],
                df[y],
                c=GIPlot.PAL_TISSUE_2[t],
                marker="o",
                edgecolor="",
                s=5,
                label=t,
                alpha=0.8,
            )

        ax.set_xlabel(x)
        ax.set_ylabel(y)
        ax.grid(True, ls="-", lw=0.1, alpha=1.0, zorder=0)
        ax.legend(
            loc="center left",
            bbox_to_anchor=(1, 0.5),
            prop={"size": 4},
            frameon=False,
            title="Tissue",
        ).get_title().set_fontsize("5")

        return ax

    @classmethod
    def gi_continuous_plot(
        cls,
        x,
        y,
        z,
        plot_df,
        cmap="Spectral_r",
        mid_point_norm=True,
        mid_point=0,
        cbar_label=None,
    ):
        df = plot_df.dropna(subset=[x, y, z])

        fig, ax = plt.subplots(1, 1, figsize=(2.5, 2), dpi=600)

        sc = ax.scatter(
            df[x],
            df[y],
            c=df[z],
            marker="o",
            edgecolor="",
            s=5,
            cmap=cmap,
            alpha=0.8,
            norm=MidpointNormalize(midpoint=mid_point) if mid_point_norm else None,
        )

        cbar = plt.colorbar(sc)
        cbar.ax.set_ylabel(
            z if cbar_label is None else cbar_label, rotation=270, va="bottom"
        )

        ax.set_xlabel(x)
        ax.set_ylabel(y)
        ax.grid(True, ls="-", lw=0.1, alpha=1.0, zorder=0)

        return ax


class MOFAPlot(CrispyPlot):
    @classmethod
    def variance_explained_heatmap(cls, df_r2, row_order=None):
        nrows, ncols = df_r2.shape
        row_order = list(df_r2.index) if row_order is None else row_order

        f, (axh, axb) = plt.subplots(
            1,
            2,
            sharex="col",
            sharey="row",
            figsize=(0.3 * ncols, 0.3 * nrows),
            gridspec_kw=dict(width_ratios=[4, 1]),
        )

        # Heatmap
        g = sns.heatmap(
            df_r2.loc[row_order],
            cmap="Greys",
            annot=True,
            cbar=False,
            fmt=".2f",
            linewidths=0.5,
            ax=axh,
            annot_kws={"fontsize": 5},
        )
        axh.set_xlabel("Factors")
        axh.set_ylabel("Data-sets")
        axh.set_title("Variance explained per factor")
        axh.grid(True, ls="-", lw=0.1, alpha=1.0, zorder=0)
        g.set_yticklabels(g.get_yticklabels(), rotation=0, horizontalalignment="right")

        # Barplot
        i_sums = df_r2.loc[row_order[::-1]].sum(1)

        norm = mpl.colors.Normalize(vmin=0, vmax=i_sums.max())
        colors = [plt.cm.Greys(norm(v)) for v in i_sums]

        axb.barh(np.arange(len(row_order)) + 0.5, i_sums, color=colors, linewidth=0)
        axb.set_xlabel("R2")
        axb.set_ylabel("")
        axb.set_title("Total variance per data-set")
        axb.grid(True, ls="-", lw=0.1, alpha=1.0, zorder=0, axis="x")

        plt.subplots_adjust(hspace=0, wspace=0.05)

    @classmethod
    def factors_weights(cls, factor_weights, dataset, cmap="RdYlGn"):
        nrows, ncols = factor_weights[dataset].shape
        fig = sns.clustermap(
            factor_weights[dataset],
            cmap=cmap,
            center=0,
            figsize=(0.25 * ncols, min(0.01 * nrows, 10)),
        )
        fig.ax_heatmap.set_xlabel("Factors")
        fig.ax_heatmap.set_ylabel(f"{dataset} features")

    @classmethod
    def factor_weights_scatter(cls, weights, dataset, factor, n_features=30, fontsize=4, r2=None):
        df = cls.get_top_features(weights, dataset, factor, n_features).reset_index()

        fig, ax = plt.subplots(1, 1, figsize=(2.5, 2), dpi=600)

        ax.scatter(df.index, df[factor], marker="o", edgecolor="", s=5, alpha=0.8, c=cls.PAL_DTRACE[2])

        texts = [
            plt.text(i, w, g, color="k", fontsize=fontsize) for i, (g, w) in df.iterrows()
        ]
        adjust_text(
            texts, arrowprops=dict(arrowstyle="-", color="k", alpha=0.75, lw=0.3)
        )

        ax.set_ylabel("Loading" if r2 is None else f"Loading ({r2.loc[dataset, factor]:.2f}%)")
        ax.set_xlabel("Rank position")
        ax.set_title(f"Data-set: {dataset}; Factor: {factor}; Top {n_features} features")
        ax.grid(True, ls="-", lw=0.1, alpha=1.0, zorder=0)

        return ax

    @classmethod
    def data_heatmap(cls, df, cmap="RdYlGn", title="", center=None, col_colors=None, row_colors=None, mask=None):
        nrows, ncols = df.shape
        fig = sns.clustermap(
            df,
            mask=mask,
            cmap=cmap,
            center=center,
            col_colors=col_colors,
            row_colors=row_colors,
            xticklabels=False,
            figsize=(min(0.2 * ncols, 15), min(0.2 * nrows, 15)),
        )
        fig.ax_heatmap.set_xlabel("Samples")
        fig.ax_heatmap.set_ylabel(f"Features")
        fig.ax_col_dendrogram.set_title(title)

    @staticmethod
    def get_top_features(weights, dataset, factor, n_features=30):
        df = weights[dataset][factor]
        df = df.loc[df.abs().sort_values(ascending=False).head(n_features).index]
        return df.sort_values()
