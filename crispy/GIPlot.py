#!/usr/bin/env python
# Copyright (C) 2019 Emanuel Goncalves

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from natsort import natsorted
from crispy.Utils import Utils
from adjustText import adjust_text
from matplotlib.lines import Line2D
from scipy.stats import pearsonr, spearmanr
from sklearn.preprocessing import MinMaxScaler
from crispy.CrispyPlot import CrispyPlot, MidpointNormalize


class GIPlot(CrispyPlot):
    MARKERS = ["o", "X", "v", "^"]

    @classmethod
    def gi_regression_no_marginals(
        cls,
        x_gene,
        y_gene,
        plot_df,
        alpha=1.0,
        hue=None,
        style=None,
        lowess=False,
        palette=None,
        plot_reg=True,
        figsize=(3, 3),
        plot_style_legend=True,
        plot_hue_legend=True,
        ax=None,
    ):
        pal = cls.PAL_DTRACE if palette is None else palette

        plot_df = plot_df.dropna(subset=[x_gene, y_gene])

        if ax is None:
            _, ax = plt.subplots(1, 1, figsize=figsize, dpi=600)

        markers_handles = dict()

        # Joint
        for t, df in [("none", plot_df)] if hue is None else plot_df.groupby(hue):
            for i, (n, df_style) in enumerate(
                [("none", df)] if style is None else df.groupby(style)
            ):
                ax.scatter(
                    x=df_style[x_gene],
                    y=df_style[y_gene],
                    edgecolor="w",
                    lw=0.1,
                    s=3,
                    c=pal[2] if palette is None else pal[t],
                    alpha=alpha,
                    marker=cls.MARKERS[i],
                )
                markers_handles[n] = cls.MARKERS[i]

        if plot_reg:
            sns.regplot(
                x_gene,
                y_gene,
                data=plot_df,
                line_kws=dict(lw=1.0, color=cls.PAL_DTRACE[1]),
                marker="",
                lowess=lowess,
                truncate=True,
                ax=ax,
            )

        ax.grid(axis="both", lw=0.1, color="#e1e1e1", zorder=0)

        cor, pval = spearmanr(plot_df[x_gene], plot_df[y_gene])
        annot_text = f"Spearman's R={cor:.2g}, p-value={pval:.1e}"
        ax.text(0.95, 0.05, annot_text, fontsize=4, transform=ax.transAxes, ha="right")

        if plot_hue_legend and (palette is not None):
            hue_handles = [
                Line2D(
                    [0],
                    [0],
                    marker="o",
                    label=t,
                    mew=0,
                    markersize=3,
                    markerfacecolor=c,
                    lw=0,
                )
                for t, c in pal.items()
            ]
            hue_legend = ax.legend(
                handles=hue_handles,
                loc="center left",
                bbox_to_anchor=(1, 0.5),
                prop={"size": 3},
                frameon=False,
            )
            ax.add_artist(hue_legend)

        if plot_style_legend and (style is not None):
            style_handles = [
                Line2D(
                    [0],
                    [0],
                    marker=m,
                    label=n,
                    mew=0,
                    markersize=3,
                    markerfacecolor=cls.PAL_DTRACE[2],
                    lw=0,
                )
                for n, m in markers_handles.items()
            ]
            ax.legend(
                handles=style_handles, loc="upper left", frameon=False, prop={"size": 3}
            )

        return ax

    @classmethod
    def gi_regression(
        cls,
        x_gene,
        y_gene,
        plot_df=None,
        size=None,
        size_range=None,
        size_inverse=False,
        size_legend_loc="best",
        size_legend_title=None,
        hue=None,
        style=None,
        lowess=False,
        palette=None,
        plot_reg=True,
        plot_annot=True,
        hexbin=False,
        color=None,
        label=None,
        a=0.75,
    ):
        pal = cls.PAL_DTRACE if palette is None else palette

        if plot_df is None:
            plot_df = pd.concat([x_gene, y_gene], axis=1)
            x_gene, y_gene = x_gene.name, y_gene.name

        plot_df = plot_df.dropna(subset=[x_gene, y_gene])

        if size is not None:
            plot_df = plot_df.dropna(subset=[size])

            feature_range = [1, 10] if size_range is None else size_range
            s_transform = MinMaxScaler(feature_range=feature_range)
            s_transform = s_transform.fit((plot_df[[size]] * -1) if size_inverse else plot_df[[size]])

        grid = sns.JointGrid(x_gene, y_gene, data=plot_df, space=0)

        # Joint
        if plot_reg:
            grid.plot_joint(
                sns.regplot,
                data=plot_df,
                line_kws=dict(lw=1.0, color=cls.PAL_DTRACE[1]),
                marker="",
                lowess=lowess,
                truncate=True,
            )

        hue_df = plot_df.groupby(hue) if hue is not None else [(None, plot_df)]
        for i, (h, h_df) in enumerate(hue_df):
            style_df = h_df.groupby(style) if style is not None else [(None, h_df)]

            for j, (s, s_df) in enumerate(style_df):
                if hexbin:
                    grid.ax_joint.hexbin(
                        s_df[x_gene],
                        s_df[y_gene],
                        cmap="Spectral_r",
                        gridsize=100,
                        mincnt=1,
                        bins="log",
                        lw=0,
                        alpha=1,
                    )

                else:
                    if size is None:
                        s = 3
                    elif size_inverse:
                        s = s_transform.transform(s_df[[size]] * -1)
                    else:
                        s = s_transform.transform(s_df[[size]])

                    sc = grid.ax_joint.scatter(
                        x=s_df[x_gene],
                        y=s_df[y_gene],
                        edgecolor="w",
                        lw=0.1,
                        s=s,
                        c=pal[2] if h is None else pal[h],
                        alpha=a,
                        marker=cls.MARKERS[0] if s is None else cls.MARKERS[j],
                        label=s,
                    )

                grid.x = s_df[x_gene].rename("")
                grid.y = s_df[y_gene].rename("")
                grid.plot_marginals(
                    sns.kdeplot,
                    # hist_kws=dict(linewidth=0, alpha=a),
                    cut=0,
                    legend=False,
                    shade=True,
                    color=pal[2] if h is None else pal[h],
                    label=h,
                )

        grid.ax_joint.grid(axis="both", lw=0.1, color="#e1e1e1", zorder=0)

        if plot_annot:
            cor, pval = spearmanr(plot_df[x_gene], plot_df[y_gene])
            annot_text = f"Spearman's R={cor:.2g}, p-value={pval:.1e}"
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

        if hue is not None:
            grid.ax_marg_y.legend(
                prop=dict(size=4),
                frameon=False,
                loc="center left",
                bbox_to_anchor=(1, 0.5),
            )

        if size is not None:
            def inverse_transform_func(x):
                arg_x = np.array(x).reshape(-1, 1)
                res = s_transform.inverse_transform(arg_x)[:, 0]
                return res

            handles, labels = sc.legend_elements(
                prop="sizes",
                num=8,
                func=inverse_transform_func,
            )
            grid.ax_joint.legend(
                handles,
                labels,
                title=size_legend_title,
                loc=size_legend_loc,
                frameon=False,
                prop={"size": 3},
            ).get_title().set_fontsize("3")

        plt.gcf().set_size_inches(2, 2)

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
        plot_reg=True,
        plot_annot=True,
        marginal_notch=False,
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
            notch=marginal_notch,
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

        if plot_reg:
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
        if plot_annot:
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
        cls,
        x_gene,
        y_gene,
        plot_df,
        hue=None,
        palette=None,
        orient="v",
        stripplot=True,
        notch=True,
        order=None,
        hue_order=None,
        plot_legend=True,
        legend_kws=None,
        ax=None,
    ):
        pal = cls.PAL_DTRACE if palette is None else palette

        if ax is None and orient == "v":
            figsize = (0.2 * len(set(plot_df[x_gene])), 2)

        elif ax is None and orient == "h":
            figsize = (2, 0.2 * len(set(plot_df[y_gene])))

        else:
            figsize = None

        if ax is None:
            _, ax = plt.subplots(1, 1, figsize=figsize, dpi=600)

        if stripplot:
            sns.stripplot(
                x=x_gene,
                y=y_gene,
                order=order,
                hue=hue,
                hue_order=hue_order,
                data=plot_df,
                dodge=True,
                orient=orient,
                jitter=0.3,
                size=1.5,
                linewidth=0.1,
                alpha=0.5,
                edgecolor="white",
                palette=pal,
                ax=ax,
                zorder=0,
            )

        bp = sns.boxplot(
            x=x_gene,
            y=y_gene,
            order=order,
            hue=hue,
            hue_order=hue_order,
            data=plot_df,
            orient=orient,
            notch=notch,
            boxprops=dict(linewidth=0.3),
            whiskerprops=dict(linewidth=0.3),
            medianprops=cls.MEDIANPROPS,
            flierprops=cls.FLIERPROPS,
            palette=pal,
            showcaps=False,
            sym="" if stripplot else None,
            saturation=1.0,
            ax=ax,
        )

        for patch in bp.artists:
            r, g, b, a = patch.get_facecolor()
            patch.set_facecolor((r, g, b, 0.1))

        ax.grid(
            True,
            axis="y" if orient == "v" else "x",
            ls="-",
            lw=0.1,
            alpha=1.0,
            zorder=0,
        )

        if plot_legend and (hue is not None):
            hue_nfeatures = len(set(plot_df[hue]))
            handles, labels = bp.get_legend_handles_labels()
            ax.legend(
                handles[: (hue_nfeatures - 1)],
                labels[: (hue_nfeatures - 1)],
                frameon=False,
                **legend_kws,
            )

        elif ax.get_legend() is not None:
            ax.get_legend().remove()

        return ax

    @classmethod
    def gi_tissue_plot(
        cls,
        x,
        y,
        plot_df=None,
        hue="tissue",
        pal=CrispyPlot.PAL_TISSUE_2,
        plot_reg=True,
        annot=True,
        lowess=False,
        figsize=(2, 2),
    ):
        if plot_df is None:
            plot_df = pd.concat([x, y], axis=1)
            x, y = x.name, y.name

        plot_df = plot_df.dropna(subset=[x, y])

        fig, ax = plt.subplots(1, 1, figsize=figsize, dpi=600)

        for t, df in plot_df.groupby(hue):
            ax.scatter(
                df[x],
                df[y],
                c=pal[t],
                marker="o",
                edgecolor="",
                s=5,
                label=t,
                alpha=0.8,
            )

        if plot_reg:
            sns.regplot(
                x,
                y,
                data=plot_df,
                line_kws=dict(lw=1.0, color=cls.PAL_DTRACE[1]),
                marker="",
                lowess=lowess,
                truncate=True,
                ax=ax,
            )

        if annot:
            cor, pval = spearmanr(plot_df[x], plot_df[y])
            annot_text = f"Spearman's R={cor:.2g}, p-value={pval:.1e}"
            ax.text(
                0.95, 0.05, annot_text, fontsize=4, transform=ax.transAxes, ha="right"
            )

        ax.set_xlabel(x)
        ax.set_ylabel(y)
        ax.grid(True, ls="-", lw=0.1, alpha=1.0, zorder=0)
        ax.legend(
            loc="center left",
            bbox_to_anchor=(1, 0.5),
            prop={"size": 3},
            frameon=False,
            title=hue,
        ).get_title().set_fontsize("3")

        return ax

    @classmethod
    def gi_continuous_plot(
        cls,
        x,
        y,
        z,
        plot_df,
        cmap="Spectral_r",
        joint_alpha=0.8,
        mid_point_norm=True,
        mid_point=0,
        cbar_label=None,
        lowess=False,
        plot_reg=False,
        corr_annotation=True,
        ax=None,
    ):
        df = plot_df.dropna(subset=[x, y, z])

        if ax is None:
            _, ax = plt.subplots(1, 1, figsize=(2.5, 2), dpi=600)

        sc = ax.scatter(
            df[x],
            df[y],
            c=df[z],
            marker="o",
            edgecolor="",
            s=3,
            cmap=cmap,
            alpha=joint_alpha,
            norm=MidpointNormalize(midpoint=mid_point) if mid_point_norm else None,
        )

        cbar = plt.colorbar(sc)
        cbar.ax.set_ylabel(
            z if cbar_label is None else cbar_label, rotation=270, va="bottom"
        )

        if plot_reg:
            sns.regplot(
                x,
                y,
                data=plot_df,
                line_kws=dict(lw=1.0, color=cls.PAL_DTRACE[1]),
                marker="",
                lowess=lowess,
                truncate=True,
                ax=ax,
            )

        if corr_annotation:
            cor, pval = spearmanr(df[x], df[y])
            annot_text = f"R={cor:.2g}, p={pval:.1e}"
            ax.text(
                0.95, 0.05, annot_text, fontsize=4, transform=ax.transAxes, ha="right"
            )

        ax.set_xlabel(x)
        ax.set_ylabel(y)
        ax.grid(True, ls="-", lw=0.1, alpha=1.0, zorder=0)

        return ax

    @classmethod
    def gi_manhattan(cls, assoc_df):
        _, axs = plt.subplots(1, len(Utils.CHR_ORDER), figsize=(len(Utils.CHR_ORDER), 3), dpi=600, sharey="row")

        for i, ctype in enumerate(Utils.CHR_ORDER):
            ax = axs[i]

            for ttype in ["ppi != 'T'", "ppi == 'T'"]:
                t_df = assoc_df.query(f"(chr == '{ctype}') & (fdr < .1) & ({ttype})")
                t_df["log_pval"] = -np.log10(t_df["pval"])

                def calculate_score(pval, beta):
                    s = np.log10(pval)
                    if beta > 0:
                        s *= -1
                    return s

                t_df["score"] = [
                    calculate_score(p, b) for p, b in t_df[["pval", "beta"]].values
                ]

                ax.scatter(
                    t_df["chr_pos"],
                    t_df["score"],
                    s=5,
                    alpha=.7,
                    c=CrispyPlot.PAL_DTRACE[0],
                    lw=.5 if ttype == "ppi == 'T'" else 0,
                    edgecolors=CrispyPlot.PAL_DTRACE[1] if ttype == "ppi == 'T'" else None,
                )

                labels = [
                    ax.text(
                        row["chr_pos"],
                        row["score"],
                        f"{row['y_id'].split(';')[1] if ';' in row['y_id'] else row['y_id']} ~ {row['x_id']}",
                        color="k",
                        fontsize=4,
                        zorder=3,
                    )
                    for _, row in t_df.query(f"log_pval > 10").head(5).iterrows()
                ]
                adjust_text(
                    labels,
                    arrowprops=dict(arrowstyle="-", color="k", alpha=0.75, lw=0.3),
                    ax=ax,
                )

            ax.set_xlabel(f"Chr {ctype}")
            ax.set_ylabel(f"-log10(p-value)" if i == 0 else None)
            ax.grid(axis="y", lw=0.1, color="#e1e1e1", zorder=0)

        plt.subplots_adjust(hspace=0, wspace=0.05)