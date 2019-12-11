#!/usr/bin/env python
# Copyright (C) 2019 Emanuel Goncalves

import logging
import numpy as np
import pandas as pd
import pkg_resources
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from natsort import natsorted
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.model_selection import StratifiedShuffleSplit

LOG = logging.getLogger("Crispy")
DPATH = pkg_resources.resource_filename("crispy", "data/")


class GExp:
    GEXP_GENESETS_FILE = f"{DPATH}/GExp_genesets_20191126.csv"

    def __init__(self, genesets=None):
        if genesets is None:
            self.genesets = (
                pd.read_csv(self.GEXP_GENESETS_FILE)
                .groupby("gtype")["gene"]
                .agg(set)
                .to_dict()
            )

        else:
            self.genesets = genesets

        LOG.info(
            f"N Genes low/high: {len(self.genesets['low'])}/{len(self.genesets['high'])}"
        )

        self.palette = {"low": "#fc8d62", "high": "#2b8cbe", "all": "#656565"}

    def plot_histogram(self, x, figsize=(2.5, 1.5)):
        _, ax = plt.subplots(1, 1, figsize=figsize)
        for n, s in [
            ("low", x.reindex(self.genesets["low"])),
            ("high", x.reindex(self.genesets["high"])),
            ("all", x),
        ]:
            sns.distplot(
                s,
                hist=False,
                label=n,
                kde_kws={"cut": 0, "shade": True},
                color=self.palette[n],
                ax=ax,
            )
        ax.grid(True, ls="-", lw=0.1, alpha=1.0, zorder=0, axis="x")
        ax.legend(frameon=False, prop={"size": 5})
        ax.set_title(x.name)
        return ax

    @staticmethod
    def benchmark(y_true, y_score, max_fpr, return_curve=False):
        fpr, tpr, thres = roc_curve(y_true, y_score)

        auc = roc_auc_score(y_true, y_score)

        fpr_thres = min(thres[fpr <= max_fpr])

        return (auc, fpr_thres, fpr, tpr, thres) if return_curve else (auc, fpr_thres)

    def discretise(
        self,
        x,
        n_splits=100,
        train_size=0.7,
        decimal_places=2,
        max_fpr=0.05,
        verbose=0,
    ):
        # Round x values
        x = x.round(decimal_places)

        # Gene-expression values range
        x_range = list(natsorted(set(x)))

        # x values labels
        x_labels = pd.Series([0] * len(x), index=x.index)
        x_labels.loc[x_labels.index.isin(self.genesets["low"])] = -1
        x_labels.loc[x_labels.index.isin(self.genesets["high"])] = 1

        # Cross-validation
        x_lr, x_stats = [], []

        cv = StratifiedShuffleSplit(n_splits, train_size=train_size)
        for i, (idx_train, idx_test) in enumerate(cv.split(x.to_frame(), x_labels)):
            # Train and Test set
            x_train, x_test = x.iloc[idx_train], x.iloc[idx_test]

            # High and low gene-expression values
            x_train_low = x_train.reindex(self.genesets["low"]).dropna()
            x_train_high = x_train.reindex(self.genesets["high"]).dropna()

            if verbose > 0:
                LOG.info(
                    f"[Iteration {i+1}] N Genes (low/high): {len(x_train_low)}/{len(x_train_high)}"
                )

            # Density fitting
            x_train_low_k = stats.gaussian_kde(x_train_low)
            x_train_high_k = stats.gaussian_kde(x_train_high)

            # Estimate log-ratios for values range
            lr = pd.Series(
                np.log2(
                    x_train_high_k.evaluate(x_range) / x_train_low_k.evaluate(x_range)
                ),
                index=x_range,
            )

            # Assign x_test log-ratios
            x_test_lr = lr[x_test.values]
            x_test_lr.index = x_test.index
            x_lr.append(x_test_lr)

            # Benchmark
            for gset in self.genesets:
                y_test_score = x_test_lr if gset == "high" else -x_test_lr
                y_test_true = y_test_score.index.isin(self.genesets[gset]).astype(int)

                y_test_auc, y_test_thres = self.benchmark(
                    y_test_true, y_test_score, max_fpr=max_fpr
                )
                y_test_thres = y_test_thres if gset == "high" else -y_test_thres

                x_stats.append(dict(auc=y_test_auc, thres=y_test_thres, gtype=gset))

        # Aggregate
        x_lr = pd.concat(x_lr, axis=1, sort=False)
        x_stats = pd.DataFrame(x_stats)

        # Returning data-frame
        res = pd.concat(
            [
                x_lr.mean(1).rename("lr_mean"),
                x_lr.std(1).rename("lr_std"),
                x_lr.count(1).rename("lr_count"),
                x_labels.rename("gtype"),
            ],
            axis=1,
            sort=False,
        ).sort_values("lr_mean")

        for gset, gset_df in x_stats.groupby("gtype"):
            res[f"{gset}_thres_std"] = gset_df["thres"].std()
            res[f"{gset}_thres_mean"] = gset_df["thres"].mean()

            if gset == "high":
                res[gset] = (res["lr_mean"] > gset_df["thres"].mean()).astype(int)

            else:
                res[gset] = (res["lr_mean"] < gset_df["thres"].mean()).astype(int)

        if (len(x) - len(x_lr)) != 0:
            LOG.warning(f"Genes {len(x)} / {len(res)} cross-validated")

        return res
