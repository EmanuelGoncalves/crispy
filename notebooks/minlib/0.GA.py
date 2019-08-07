# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.3'
#       jupytext_version: 1.0.0
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %matplotlib inline
# %autosave 0
# %load_ext autoreload
# %autoreload 2

import random
import logging
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from math import sqrt
from itertools import groupby
from crispy.QCPlot import QCplot
from scipy.stats import spearmanr
from scipy.interpolate import interpn
from crispy.CrispyPlot import CrispyPlot
from crispy.DataImporter import PipelineResults, Sample
from crispy.CRISPRData import CRISPRDataSet, ReadCounts, Library
from sklearn.metrics import mean_squared_error, matthews_corrcoef
from minlib import dpath, rpath, LOG, sgrnas_scores_scatter, project_score_sample_map
from deap import base, creator, tools, algorithms


# Project Score

ky = CRISPRDataSet("Yusa_v1.1")
ky_smap = project_score_sample_map()


# sgRNAs lib

lib = ky.lib.dropna(subset=["Gene"])


# GA
# Params
N_GEN = 10
POP_SIZE = 20
CXPB = .5
MUTPB = .2
INDPB = .15

# Types
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

# Initialisation
toolbox = base.Toolbox()
toolbox.register("attribute", np.random.randint, low=2)

# Operators
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutFlipBit, indpb=INDPB)
toolbox.register("select", tools.selTournament, tournsize=int(POP_SIZE * .25))


# Optimisation

lib_opt = []
for gene, gene_lib in lib.groupby("Gene"):
    sgrnas = np.array(gene_lib.index)

    if len(sgrnas) <= 1:
        continue

    def evaluate(individual):
        count_all = ky.counts.loc[sgrnas].mean()
        count_individual = ky.counts.loc[sgrnas[np.array(individual) == 1]].mean()

        corr = spearmanr(count_all, count_individual)[0]
        guid = sum(individual) / len(individual)

        return (corr) * (1 - guid),

    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attribute, n=len(sgrnas))
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    toolbox.register("evaluate", evaluate)

    # Statistics
    stats = tools.Statistics(key=lambda ind: ind.fitness.values)
    stats.register("median", np.nanmedian)
    stats.register("gene", lambda v: gene)

    pop = toolbox.population(n=POP_SIZE)
    pop = algorithms.eaSimple(pop, toolbox, cxpb=CXPB, mutpb=MUTPB, ngen=N_GEN, stats=stats)[0]

    lib_opt.append(pd.DataFrame(pop, columns=sgrnas).mean())

lib_opt = pd.concat(lib_opt)
lib_opt = pd.concat([lib, lib_opt.rename("ga")], axis=1, sort=False)


#

minimal_lib = lib_opt.query("ga > .75")


#

ky.counts.reindex(minimal_lib.index).dropna().to_csv(
    f"{rpath}/KosukeYusa_v1.1_sgrna_counts_minimal_ga.csv.gz",
    compression="gzip",
)

minimal_lib.to_csv(f"{dpath}/crispr_libs/minimal_ga.csv.gz", compression="gzip", index=False)
