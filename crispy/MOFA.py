#!/usr/bin/env python
# Copyright (C) 2019 Emanuel Goncalves

import io
import gseapy
import logging
import numpy as np
import pandas as pd
import pkg_resources
import seaborn as sns
import matplotlib.pyplot as plt
from crispy.QCPlot import QCplot
from crispy.Enrichment import Enrichment
from crispy.CrispyPlot import CrispyPlot
from swath_proteomics.LMModels import LMModels
from mofapy2.run.entry_point import entry_point
from swath_proteomics.GIPlot import GIPlot, MOFAPlot
from sklearn.metrics import roc_auc_score, roc_curve


class MOFA:
    def __init__(
        self,
        views,
        factors_n=10,
        scale_views=True,
        iterations=1000,
        convergence_mode="slow",
        startELBO=1,
        freqELBO=1,
        dropR2=None,
        verbose=1,
    ):
        """""
        This is a wrapper of MOFA to perform multi-omics integrations of the GDSC data-sets. Multiple groups are NOT
        supported.
        
        :param views: dict(str: pandas.DataFrame)
        """""

        self.verbose = verbose

        self.views = views

        self.factors_n = factors_n
        self.factors_labels = [f"F{i+1}" for i in range(factors_n)]

        self.scale_views = scale_views

        self.iterations = iterations
        self.convergence_mode = convergence_mode
        self.startELBO = startELBO
        self.freqELBO = freqELBO
        self.dropR2 = dropR2

        # Prepare data in melt tidy format
        self.data = []
        for k, df in self.views.items():
            df.columns.name = "sample"
            df.index.name = "feature"
            self.data.append(df.unstack().rename("value").reset_index().assign(view=k))
        self.data = (
            pd.concat(self.data, ignore_index=True).assign(group="gdsc").dropna()
        )

        # Initialise entry point
        self.ep = entry_point()

        # Set data options
        self.ep.set_data_options(scale_groups=False, scale_views=self.scale_views)
        self.ep.set_data_df(self.data)

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

        # Build and run MOFA
        self.ep.build()
        self.ep.run()
