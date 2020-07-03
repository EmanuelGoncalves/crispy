#!/usr/bin/env python
# Copyright (C) 2019 Emanuel Goncalves

import os
import h5py
import gseapy
import logging
import matplotlib
import numpy as np
import pandas as pd
import pkg_resources
import seaborn as sns
import matplotlib.pyplot as plt
from adjustText import adjust_text
from crispy.Enrichment import Enrichment
from crispy.CrispyPlot import CrispyPlot
from mofapy2.run.entry_point import entry_point
from sklearn.linear_model import LinearRegression

LOG = logging.getLogger("Crispy")
DPATH = pkg_resources.resource_filename("crispy", "data")


class MOFA:
    def __init__(
        self,
        views=None,
        groupby=None,
        likelihoods=None,
        factors_n=10,
        covariates=None,
        fit_intercept=True,
        scale_views=True,
        scale_groups=True,
        iterations=1000,
        convergence_mode="slow",
        use_overlap=True,
        startELBO=1,
        freqELBO=1,
        dropR2=None,
        verbose=1,
        from_file=None,
    ):
        """
        This is a wrapper of MOFA to perform multi-omics integrations of the GDSC data-sets. 
        
            - Multiple groups are NOT supported
            - Only samples part of all views are considered
        
        :param views: dict(str: pandas.DataFrame)
        """

        self.verbose = verbose
        self.from_file = from_file

        self.factors_n = factors_n
        self.factors_labels = [f"F{i + 1}" for i in range(factors_n)]

        self.likelihoods = likelihoods

        self.views = views
        self.scale_views = scale_views
        self.scale_groups = scale_groups
        self.views_labels = list(self.views)

        self.use_overlap = use_overlap

        self.iterations = iterations
        self.convergence_mode = convergence_mode
        self.startELBO = startELBO
        self.freqELBO = freqELBO
        self.dropR2 = dropR2

        # Covariates
        self.covariates = covariates
        self.fit_intercept = fit_intercept

        # Samples
        self.samples = set.intersection(
            *[set(self.views[v]) for v in self.views_labels]
        )

        if self.covariates is not None:
            LOG.info(f"Covariates provided N={len(self.covariates)}")
            for k, v in self.covariates.items():
                self.samples = self.samples.intersection(set(v.index))

        self.samples = list(self.samples)
        LOG.info(f"Overlaping samples: {len(self.samples)}")

        # Reduce to overlaping samples
        if self.use_overlap:
            for k, df in self.views.items():
                self.views[k] = df[self.samples]

        # Info
        for k, df in self.views.items():
            LOG.info(f"View {k}: {df.shape}")

        # Regress-out covariates
        if self.covariates is not None:
            self.views = self.regress_out_covariates(fit_intercept=self.fit_intercept)

        # RUN MOFA
        # Prepare data, melt & tidy
        self.data = []

        for k in self.views_labels:
            df = self.views[k].copy()
            df.index.name = "feature"
            df.columns.name = "sample"

            df = df.unstack().rename("value").reset_index()

            if groupby is not None:
                df["group"] = groupby.reindex(df["sample"]).values

            else:
                df = df.assign(group="gdsc")

            self.data.append(df.assign(view=k))

        self.data = pd.concat(self.data, ignore_index=True)
        self.data = self.data[["sample", "group", "feature", "value", "view"]].dropna()

        # Initialise entry point
        self.ep = entry_point()

        # Set data options
        self.ep.set_data_options(
            scale_groups=self.scale_groups, scale_views=self.scale_views
        )
        self.ep.set_data_df(self.data, likelihoods=self.likelihoods)

        # Set model options
        self.ep.set_model_options(factors=self.factors_n)

        # Set training options
        self.ep.set_train_options(
            iter=self.iterations,
            convergence_mode=self.convergence_mode,
            startELBO=self.startELBO,
            freqELBO=self.freqELBO,
            dropR2=self.dropR2,
            verbose=verbose > 0,
        )

        # Run MOFA
        self.ep.build()

        if not os.path.isfile(self.from_file):
            self.ep.run()
            self.save_hdf5(self.from_file)

        self.mofa_file = h5py.File(self.from_file, "r")

        self.factors = self.get_factors(self.mofa_file)
        self.weights = self.get_weights(self.mofa_file)
        self.rsquare = self.get_rsquare(self.mofa_file)

    @staticmethod
    def get_factors(mofa_hdf5):
        factors_n = int(mofa_hdf5["training_stats"]["number_factors"][0])
        factors_labels = [f"F{i + 1}" for i in range(factors_n)]

        z = mofa_hdf5["expectations"]["Z"]
        factors = pd.concat(
            [
                pd.DataFrame(
                    df,
                    columns=[s.decode("utf-8") for s in mofa_hdf5["samples"][k]],
                    index=factors_labels,
                ).T
                for k, df in z.items()
            ]
        )
        return factors

    @staticmethod
    def get_weights(mofa_hdf5):
        factors_n = int(mofa_hdf5["training_stats"]["number_factors"][0])
        factors_labels = [f"F{i + 1}" for i in range(factors_n)]

        w = mofa_hdf5["expectations"]["W"]
        weights = {
            n: pd.DataFrame(
                w[n].value.T,
                index=[f.decode("utf-8") for f in mofa_hdf5["features"][n]],
                columns=factors_labels,
            )
            for n, df in w.items()
        }
        return weights

    @staticmethod
    def get_rsquare(mofa_hdf5):
        factors_n = int(mofa_hdf5["training_stats"]["number_factors"][0])
        factors_labels = [f"F{i + 1}" for i in range(factors_n)]

        r2 = mofa_hdf5["variance_explained"]["r2_per_factor"]
        rsquare = {
            k: pd.DataFrame(
                df,
                index=[s.decode("utf-8") for s in mofa_hdf5["views"]["views"]],
                columns=factors_labels,
            )
            for k, df in r2.items()
        }
        return rsquare

    @classmethod
    def read_mofa_hdf5(cls, hdf5_file):
        mofa_hdf5 = h5py.File(hdf5_file, "r")

        factors = cls.get_factors(mofa_hdf5)
        weights = cls.get_weights(mofa_hdf5)
        rsquare = cls.get_rsquare(mofa_hdf5)

        return factors, weights, rsquare

    @staticmethod
    def lm_residuals(y, x, fit_intercept=True, add_intercept=False):
        # Prepare input matrices
        ys = y.dropna()
        xs = x.loc[ys.index]
        xs = xs.loc[:, xs.std() > 0]

        if ys.shape[0] <= xs.shape[1]:
            return None

        # Linear regression models
        lm = LinearRegression(fit_intercept=fit_intercept).fit(xs, ys)

        # Calculate residuals
        residuals = ys - lm.predict(xs) - lm.intercept_

        # Add intercept
        if add_intercept:
            residuals += lm.intercept_

        return residuals

    def regress_out_covariates(self, fit_intercept=True, add_intercept=True):
        views_ = {}

        for k in self.views:
            if k in self.covariates:
                cov = self.covariates[k]

                # Confirm type is continuous
                k_dtype = self.views[k].dtypes[0]

                if k_dtype in [int]:
                    LOG.warning(
                        f"View {k} is of type {k_dtype}; covariates not regressed-out"
                    )
                    views_[k] = self.views[k].copy()
                    continue

                if self.verbose > 0:
                    LOG.info(
                        f"Regressing-out covariates (N={cov.shape}) from {k} view"
                    )

                # Regress-out
                views_[k] = pd.DataFrame(
                    {
                        i: self.lm_residuals(
                            self.views[k].loc[i],
                            cov,
                            fit_intercept=fit_intercept,
                            add_intercept=add_intercept,
                        )
                        for i in self.views[k].index
                    }
                ).T[self.samples]

            else:
                views_[k] = self.views[k].copy()

        return views_

    def pathway_enrichment(
        self, factor, views=None, genesets=None, nprocesses=4, permutation_num=0
    ):
        if genesets is None:
            genesets = [
                "c6.all.v7.1.symbols.gmt",
                "c5.all.v7.1.symbols.gmt",
                "h.all.v7.1.symbols.gmt",
                "c2.all.v7.1.symbols.gmt",
            ]

        if views is None:
            views = ["methylation", "transcriptomics", "proteomics"]

        df = pd.concat(
            [
                gseapy.ssgsea(
                    self.weights[v][factor],
                    processes=nprocesses,
                    permutation_num=permutation_num,
                    gene_sets=Enrichment.read_gmt(f"{DPATH}/pathways/{g}"),
                    no_plot=True,
                )
                .res2d.assign(geneset=g)
                .assign(view=v)
                .reset_index()
                for v in views
                for g in genesets
            ],
            ignore_index=True,
        )
        df = df.rename(columns={"sample1": "nes"}).sort_values("nes")
        return df

    def get_top_features(self, view, factor, n_features=30):
        df = self.weights[view][factor]
        df = df.loc[df.abs().sort_values(ascending=False).head(n_features).index]
        return df.sort_values()

    def get_factor_view_weights(self, factor):
        return pd.DataFrame({v: self.weights[v][factor] for v in self.weights})

    def save_hdf5(self, outfile):
        self.ep.save(outfile)


class MOFAPlot(CrispyPlot):
    @classmethod
    def factors_corr_clustermap(cls, mofa_obj, method="pearson", cmap="RdBu_r"):
        plot_df = mofa_obj.factors.corr(method=method)

        nrows, ncols = plot_df.shape
        fig = sns.clustermap(
            plot_df,
            cmap=cmap,
            center=0,
            annot=True,
            fmt=".2f",
            annot_kws={"size": 5},
            figsize=(min(0.3 * ncols, 10), min(0.3 * nrows, 10)),
        )
        fig.ax_heatmap.set_xlabel("Factors")
        fig.ax_heatmap.set_ylabel("Factors")

        fig.ax_heatmap.set_yticklabels(fig.ax_heatmap.get_yticklabels(), rotation=0)

    @classmethod
    def variance_explained_heatmap(cls, mofa_obj, row_order=None, rotate_ylabels=True):
        n_heatmaps = len(mofa_obj.rsquare)
        nrows, ncols = list(mofa_obj.rsquare.values())[0].shape
        row_order = (
            list(list(mofa_obj.rsquare.values())[0].index)
            if row_order is None
            else row_order
        )

        f, axs = plt.subplots(
            n_heatmaps,
            2,
            sharex="col",
            sharey="row",
            figsize=(0.3 * ncols, 0.5 * nrows),
            gridspec_kw=dict(width_ratios=[4, 1]),
        )

        for i, k in enumerate(mofa_obj.rsquare):
            axh, axb = axs[i][0] if n_heatmaps > 1 else axs[0], axs[i][1]

            df = mofa_obj.rsquare[k]

            # Heatmap
            g = sns.heatmap(
                df.loc[row_order],
                cmap="Greys",
                annot=True,
                cbar=False,
                fmt=".2f",
                linewidths=0.5,
                ax=axh,
                annot_kws={"fontsize": 5},
            )
            axh.set_xlabel("Factors" if i == (n_heatmaps - 1) else None)
            axh.set_ylabel(k)
            axh.set_title("Variance explained per factor" if i == 0 else None)
            g.set_yticklabels(
                g.get_yticklabels(), rotation=0 if rotate_ylabels else 90, horizontalalignment="right"
            )

            # Barplot
            i_sums = df.loc[row_order[::-1]].sum(1)

            norm = matplotlib.colors.Normalize(vmin=0, vmax=i_sums.max())
            colors = [plt.cm.Greys(norm(v)) for v in i_sums]

            axb.barh(np.arange(len(row_order)) + 0.5, i_sums, color=colors, linewidth=0)
            axb.set_xlabel("R2" if i == (n_heatmaps - 1) else None)
            axb.set_ylabel("")
            axb.set_title("Total variance per view" if i == 0 else None)
            axb.grid(True, ls="-", lw=0.1, alpha=1.0, zorder=0, axis="x")

        plt.subplots_adjust(hspace=0, wspace=0.05)

    @classmethod
    def factors_weights(cls, mofa_obj, view, cmap="RdYlGn"):
        nrows, ncols = mofa_obj.weights[view].shape
        fig = sns.clustermap(
            mofa_obj.weights[view],
            cmap=cmap,
            center=0,
            figsize=(0.25 * ncols, min(0.01 * nrows, 10)),
        )
        fig.ax_heatmap.set_xlabel("Factors")
        fig.ax_heatmap.set_ylabel(f"{view} features")

    @classmethod
    def factor_weights_scatter(
        cls, mofa_obj, view, factor, n_features=30, label_features=True, fontsize=4, r2=None
    ):
        df = mofa_obj.get_top_features(
            view, factor, n_features=n_features
        ).reset_index()

        fig, ax = plt.subplots(1, 1, figsize=(2.5, 2), dpi=600)

        ax.scatter(
            df.index,
            df[factor],
            marker="o",
            edgecolor="",
            s=5,
            alpha=0.8,
            c=cls.PAL_DTRACE[2],
        )

        if label_features:
            texts = [
                plt.text(i, w, g, color="k", fontsize=fontsize)
                for i, (g, w) in df.iterrows()
            ]
            adjust_text(
                texts, arrowprops=dict(arrowstyle="-", color="k", alpha=0.75, lw=0.3)
            )

        ax.set_ylabel(
            "Loading"
            if r2 is None
            else f"Loading ({mofa_obj.rsquare.loc[view, factor]:.2f}%)"
        )
        ax.set_xlabel("Rank position")
        ax.set_title(f"View: {view}; Factor: {factor}; Top {n_features} features")
        ax.grid(True, ls="-", lw=0.1, alpha=1.0, zorder=0)

        return ax

    @classmethod
    def view_heatmap(
        cls,
        mofa_obj,
        view,
        factor,
        df=None,
        cmap="Spectral",
        title="",
        center=None,
        col_colors=None,
        row_colors=None,
        mask=None,
        n_features=30,
        standard_scale=None,
    ):
        if df is None:
            df = mofa_obj.views[view]
            df = df.loc[
                mofa_obj.get_top_features(view, factor, n_features=n_features).index
            ]

        if mask is None:
            mask = df.isnull()

        if center is None:
            center = 0.5 if view == "methylation" else 0

        elif not center:
            center = None

        df = df.T.fillna(df.T.mean()).T

        nrows, ncols = mofa_obj.views[view].shape
        fig = sns.clustermap(
            df,
            mask=mask,
            cmap=cmap,
            center=center,
            standard_scale=standard_scale,
            col_colors=col_colors,
            row_colors=row_colors,
            xticklabels=False,
            figsize=(min(0.1 * ncols, 7), min(0.05 * nrows, 7)),
        )
        fig.ax_heatmap.set_xlabel("Samples")
        fig.ax_heatmap.set_ylabel(f"Features")
        fig.ax_col_dendrogram.set_title(title)

    @classmethod
    def covariates_heatmap(cls, covs, mofa_obj, agg):
        # Dendogram heatmap
        n_heatmaps = len(mofa_obj.rsquare)
        nrows, ncols = list(mofa_obj.rsquare.values())[0].shape

        row_order = list(list(mofa_obj.rsquare.values())[0].index)
        col_order = list(list(mofa_obj.rsquare.values())[0].columns)

        f, axs = plt.subplots(
            1,
            n_heatmaps + 2,
            sharex="none",
            sharey="none",

            gridspec_kw={"width_ratios": [0.4] * n_heatmaps + [9, 4]},
            figsize=(0.3 * n_heatmaps + 0.3 * covs.shape[0] + 0.3 * len(set(agg)), 0.3 * ncols),
        )

        vmax = np.max([mofa_obj.rsquare[k].max().max() for k in mofa_obj.rsquare])

        # Factors
        for i, k in enumerate(mofa_obj.rsquare):
            axh = axs[i]

            df = mofa_obj.rsquare[k]

            # Heatmap
            g = sns.heatmap(
                df.loc[row_order, col_order].T,
                cmap="Greys",
                annot=True,
                cbar=False,
                fmt=".2f",
                linewidths=0.5,
                ax=axh,
                vmin=0,
                vmax=vmax,
                annot_kws={"fontsize": 5},
            )
            axh.set_title(f"{k} cell lines")
            g.set_xticklabels(
                labels=row_order, rotation=90, horizontalalignment="center"
            )
            g.set_yticklabels(
                g.get_yticklabels() if i == 0 else [], rotation=0, horizontalalignment="right"
            )

            # # Barplot
            # i_sums = df.loc[row_order[::-1]].sum(1)
            #
            # norm = matplotlib.colors.Normalize(vmin=0, vmax=i_sums.max())
            # colors = [plt.cm.Greys(norm(v)) for v in i_sums]
            #
            # axb.barh(np.arange(len(row_order)) + 0.5, i_sums, color=colors, linewidth=0)
            # axb.set_xlabel("R2" if i == (n_heatmaps - 1) else None)
            # axb.set_ylabel("")
            # axb.set_title("Total variance per view" if i == 0 else None)
            # axb.grid(True, ls="-", lw=0.1, alpha=1.0, zorder=0, axis="x")

        # Covariates
        ax = axs[n_heatmaps]

        g = sns.heatmap(
            covs.T.loc[col_order],
            cmap="Spectral",
            center=0,
            annot=True,
            cbar=False,
            fmt=".2f",
            linewidths=0.5,
            ax=ax,
            annot_kws={"fontsize": 5},
        )
        g.set_yticklabels([])
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.set_title(f"Potential related factors")

        # Tissue contribution
        ax = axs[-1]

        g = sns.heatmap(
            mofa_obj.factors.groupby(agg).mean().T.loc[col_order],
            cmap=sns.diverging_palette(220, 20, sep=20, as_cmap=True),
            center=0,
            annot=True,
            cbar=False,
            fmt=".1f",
            linewidths=0.5,
            ax=ax,
            annot_kws={"fontsize": 5},
        )
        ax.yaxis.tick_right()
        g.set_yticklabels(
            g.get_yticklabels(), rotation=0, horizontalalignment="left"
        )
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.set_title(f"{agg.name} contribution")

        plt.subplots_adjust(wspace=0.01)

        # plt.suptitle("Variance explained per factor" if i == 0 else None)
        # axs[-1][1].axis('off')

        return axs
