#!/usr/bin/env python
# Copyright (C) 2019 Emanuel Goncalves

import os
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
        views,
        likelihoods=None,
        factors_n=10,
        covariates=None,
        fit_intercept=True,
        scale_views=True,
        iterations=1000,
        convergence_mode="slow",
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

        self.likelihoods = likelihoods

        self.factors_n = factors_n
        self.factors_labels = [f"F{i+1}" for i in range(factors_n)]

        self.views = views
        self.scale_views = scale_views
        self.views_labels = list(self.views)

        self.iterations = iterations
        self.convergence_mode = convergence_mode
        self.startELBO = startELBO
        self.freqELBO = freqELBO
        self.dropR2 = dropR2

        self.from_file = from_file

        # Covariates
        self.covariates = covariates
        self.fit_intercept = fit_intercept

        # Samples
        self.samples = set.intersection(
            *[set(self.views[v]) for v in self.views_labels]
        )

        if self.covariates is not None:
            LOG.info(f"Covariates provided: {self.covariates.shape}")
            self.samples = self.samples.intersection(set(self.covariates.index))

        self.samples = list(self.samples)
        LOG.info(f"Samples: {len(self.samples)}")

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
            df = self.views[k][self.samples].copy()
            df.index.name = "feature"
            df.columns.name = "sample"

            df = df.unstack().rename("value").reset_index()

            self.data.append(df.assign(view=k))

        self.data = pd.concat(self.data, ignore_index=True).assign(group="gdsc")
        self.data = self.data[["sample", "feature", "view", "group", "value"]].dropna()

        # Initialise entry point
        self.ep = entry_point()

        # Set data options
        self.ep.set_data_options(scale_groups=False, scale_views=self.scale_views)
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

        if (self.from_file is not None) and os.path.isfile(self.from_file):
            import h5py

            f = h5py.File(self.from_file, "r")

            self.factors = pd.DataFrame(
                f["expectations"]["Z"]["gdsc"].value.T,
                index=[s.decode("utf-8") for s in f["samples"]["gdsc"].value],
                columns=self.factors_labels,
            )

            self.weights = {
                n: pd.DataFrame(
                    f["expectations"]["W"][n].value.T,
                    index=self.views[n].index,
                    columns=self.factors_labels,
                )
                for n, df in f["expectations"]["W"].items()
            }

            self.rsquare = pd.DataFrame(
                f["variance_explained"]["r2_per_factor"]["gdsc"].value,
                index=self.views_labels,
                columns=self.factors_labels,
            )

        else:
            self.ep.run()

            self.factors = self.get_factors()
            self.weights = self.get_weights()
            self.rsquare = self.get_rsquare()

    @staticmethod
    def lm_residuals(y, x, fit_intercept=True, add_intercept=True):
        # Prepare input matrices
        ys = y.dropna()
        xs = x.loc[ys.index]

        # Linear regression models
        lm = LinearRegression(fit_intercept=fit_intercept).fit(xs, ys)

        # Calculate residuals
        residuals = ys - lm.predict(xs)

        # Add intercept
        if add_intercept:
            residuals += lm.intercept_

        return residuals

    def regress_out_covariates(self, fit_intercept=True, add_intercept=True):
        views_ = {}

        for k in self.views_labels:
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
                    f"Regressing-out covariates (N={self.covariates.shape[1]}) from {k} view"
                )

            # Regress-out
            views_[k] = pd.DataFrame(
                {
                    i: self.lm_residuals(
                        self.views[k].loc[i],
                        self.covariates,
                        fit_intercept=fit_intercept,
                        add_intercept=add_intercept,
                    )
                    for i in self.views[k].index
                }
            ).T[self.samples]

        return views_

    def get_factors(self):
        return pd.DataFrame(
            self.ep.model.nodes["Z"].getExpectation(),
            index=self.samples,
            columns=self.factors_labels,
        )

    def get_weights(self):
        return {
            k: pd.DataFrame(
                self.ep.model.nodes["W"].getExpectation()[i],
                index=self.views[k].index,
                columns=self.factors_labels,
            )
            for i, k in enumerate(self.views_labels)
        }

    def get_rsquare(self):
        return pd.DataFrame(
            self.ep.model.calculate_variance_explained()[0],
            index=self.views_labels,
            columns=self.factors_labels,
        )

    def pathway_enrichment(self, factor, views=None, genesets=None, nprocesses=4, permutation_num=0):
        if genesets is None:
            genesets = [
                "c6.all.v7.0.symbols.gmt",
                "c5.bp.v7.0.symbols.gmt",
                "h.all.v7.0.symbols.gmt",
                "c2.cp.kegg.v7.0.symbols.gmt",
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
                ).res2d.assign(geneset=g)
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
    def variance_explained_heatmap(cls, mofa_obj, row_order=None):
        nrows, ncols = mofa_obj.rsquare.shape
        row_order = list(mofa_obj.rsquare.index) if row_order is None else row_order

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
            mofa_obj.rsquare.loc[row_order],
            cmap="Greys",
            annot=True,
            cbar=False,
            fmt=".2f",
            linewidths=0.5,
            ax=axh,
            annot_kws={"fontsize": 5},
        )
        axh.set_xlabel("Factors")
        axh.set_ylabel("Views")
        axh.set_title("Variance explained per factor")
        axh.grid(True, ls="-", lw=0.1, alpha=1.0, zorder=0)
        g.set_yticklabels(g.get_yticklabels(), rotation=0, horizontalalignment="right")

        # Barplot
        i_sums = mofa_obj.rsquare.loc[row_order[::-1]].sum(1)

        norm = matplotlib.colors.Normalize(vmin=0, vmax=i_sums.max())
        colors = [plt.cm.Greys(norm(v)) for v in i_sums]

        axb.barh(np.arange(len(row_order)) + 0.5, i_sums, color=colors, linewidth=0)
        axb.set_xlabel("R2")
        axb.set_ylabel("")
        axb.set_title("Total variance per view")
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
        cls, mofa_obj, view, factor, n_features=30, fontsize=4, r2=None
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
        cmap="RdYlGn",
        title="",
        center=None,
        col_colors=None,
        row_colors=None,
        mask=None,
        n_features=30,
    ):
        df = mofa_obj.views[view]
        df = df.loc[
            mofa_obj.get_top_features(view, factor, n_features=n_features).index
        ]

        if mask is None:
            mask = df.isnull()

        if center is None:
            center = 0.5 if view == "methylation" else 0

        df = df.replace(np.nan, center)

        nrows, ncols = mofa_obj.views[view].shape
        fig = sns.clustermap(
            df,
            mask=mask,
            cmap=cmap,
            center=center,
            col_colors=col_colors,
            row_colors=row_colors,
            xticklabels=False,
            figsize=(min(0.1 * ncols, 7), min(0.05 * nrows, 7)),
        )
        fig.ax_heatmap.set_xlabel("Samples")
        fig.ax_heatmap.set_ylabel(f"Features")
        fig.ax_col_dendrogram.set_title(title)

    @classmethod
    def covariates_heatmap(cls, covs, mofa_obj):
        nrows, ncols = covs.shape

        fig = sns.clustermap(
            covs,
            col_dendrogram_set_axis_off=False,
            cmap="Spectral",
            center=0,
            annot=True,
            fmt=".2f",
            cbar=False,
            col_cluster=False,
            annot_kws={"size": 5},
            figsize=(0.3 * ncols, 0.25 * nrows),
        )
        fig.ax_heatmap.set_xlabel("Factors")
        fig.ax_heatmap.set_ylabel(f"Features")

        sns.heatmap(
            mofa_obj.rsquare,
            cmap="Greys",
            annot=True,
            cbar=False,
            fmt=".2f",
            linewidths=0.5,
            ax=fig.ax_col_dendrogram,
            annot_kws={"fontsize": 5},
        )
        fig.ax_col_dendrogram.set_yticklabels(
            mofa_obj.rsquare.index, rotation=0, horizontalalignment="right"
        )
        fig.ax_col_dendrogram.set_title("Variance explained per factor")
        fig.ax_col_dendrogram.set_ylabel("Views")

        return fig
