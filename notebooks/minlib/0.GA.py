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
import datetime
import itertools
import numpy as np
import pandas as pd
import seaborn as sns
import multiprocessing
import matplotlib.pyplot as plt
from math import sqrt
from itertools import groupby
from natsort import natsorted
from crispy.QCPlot import QCplot
from scipy.stats import spearmanr
from scipy.interpolate import interpn
from crispy.CrispyPlot import CrispyPlot
from crispy.DataImporter import PipelineResults, Sample
from crispy.CRISPRData import CRISPRDataSet, ReadCounts, Library
from sklearn.metrics import mean_squared_error, matthews_corrcoef, recall_score, roc_auc_score, roc_curve
from minlib import dpath, rpath, LOG, sgrnas_scores_scatter, project_score_sample_map
from deap import base, creator, tools, algorithms
from crispy.Utils import Utils


# Project Score

KY = CRISPRDataSet("Yusa_v1.1")
KY_SMAP = project_score_sample_map()
KY_COUNTS = KY.counts.remove_low_counts(KY.plasmids)


# sgRNAs lib

mlib = (
    Library.load_library("master.csv.gz", set_index=False)
    .query("lib == 'KosukeYusa'")
    .dropna(subset=["Gene"])
    .set_index("sgRNA_ID")
)
mlib = mlib[mlib.index.isin(KY_COUNTS.index)]


# sgRNAs lib filtered

JACKS_THRES = 1.0

ky_v11_wge = pd.read_csv(
    f"{dpath}/crispr_libs/Yusa_v1.1_WGE_map.txt", sep="\t", index_col=0
)
ky_v11_wge = ky_v11_wge[~ky_v11_wge["WGE_sequence"].isna()]


mlib_min = mlib.query(f"jacks_min < {JACKS_THRES}").sort_values("ks_control_min")
mlib_min = mlib_min[
    ~mlib_min.index.isin(ky_v11_wge[ky_v11_wge["Assembly"] == "Grch19"].index)
]

mlib_min_dict = {g: list(df.index) for g, df in mlib_min.groupby("Gene")}


# Missed dependencies

top_missed = pd.read_csv(f"{rpath}/dep_recap_per_genes.csv", index_col=0)
top_missed = top_missed.eval("all - minimal_top_2")
top_missed


# Genes

GENES = natsorted(set(mlib_min["Gene"]))
LOG.info(f"Genes: {len(GENES)}")

TRUE_SET = Utils.get_essential_genes(return_series=False)
FALSE_SET = Utils.get_non_essential_genes(return_series=False)
INDEX_SET = TRUE_SET.union(FALSE_SET)

RANK_SET = [i in INDEX_SET for i in GENES]
RANK_Y_TRUE = [int(i in TRUE_SET) for i in GENES if i in INDEX_SET]


# Evaluation functions

def individual_lib(ind):
    ind = dict(zip(*(GENES, ind)))
    ind_sgrnas = list(
        itertools.chain.from_iterable(
            [mlib_min_dict[g][: ind[g]] for g in mlib_min_dict]
        )
    )
    return ind_sgrnas


def calculate_gene_fc(sgrnas):
    return KY_SGRNA_FC.groupby(mlib.loc[sgrnas, "Gene"]).mean()


def binarise_fc(fcs, fpr_thres=0.01):
    fcs = fcs.loc[GENES]

    fcs_thres = []
    for i, _ in enumerate(fcs):
        fpr, tpr, thres = roc_curve(RANK_Y_TRUE, -fcs.iloc[RANK_SET, i])
        fcs_thres.append(-min(thres[fpr < fpr_thres]))

    fcs_bin = (fcs < fcs_thres).astype(int)
    return fcs_bin


def recap_dependencies(ind_bin):
    ind_recalls = [
        recall_score(KY_BIN.iloc[:, i], ind_bin.iloc[:, i])
        for i, _ in enumerate(SAMPLES)
    ]
    return ind_recalls


def evaluate(ind):
    # LOG.info(ind)

    ind_sgrnas = individual_lib(ind)
    # LOG.info("Library")

    ind_fc = calculate_gene_fc(ind_sgrnas)
    # LOG.info("Fold-change")

    ind_bin = binarise_fc(ind_fc)
    # LOG.info("Binary")

    ind_recall = recap_dependencies(ind_bin)
    # LOG.info("Recall")

    ind_size = sum(ind) / len(mlib_min)
    ind_fitness = sum(np.array(ind_recall) > 0.9) / len(ind_recall)

    multi_fitness = ind_fitness / ind_size
    # LOG.info("Fitness")

    return multi_fitness,


# All sgRNAs fold-changes

KY_SGRNA_FC = KY_COUNTS.norm_rpm().foldchange(KY.plasmids)
KY_SGRNA_FC = KY_SGRNA_FC.groupby(KY_SMAP["model_id"], axis=1).mean()

SAMPLES = natsorted(set(KY_SGRNA_FC))
LOG.info(f"Samples: {len(SAMPLES)}")

KY_FC = calculate_gene_fc(mlib.index)
KY_BIN = binarise_fc(KY_FC)
KY_BIN = KY_BIN.loc[GENES, SAMPLES]


# GA Params

N_GEN = 500
POP_SIZE = 50
CXPB = 0.5
MUTPB = 0.20
INDPB = 0.10
IND_SIZE = len(GENES)


# Types

creator.create("Fitness", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.Fitness)


# Initialisation

toolbox = base.Toolbox()

pool = multiprocessing.Pool()
toolbox.register("map", pool.map)

toolbox.register("attribute", np.random.randint, low=2, high=4)
toolbox.register(
    "individual", tools.initRepeat, creator.Individual, toolbox.attribute, n=IND_SIZE
)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)


# Operators

toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutUniformInt, low=2, up=4, indpb=INDPB)
toolbox.register("select", tools.selTournament, tournsize=POP_SIZE)
toolbox.register("evaluate", evaluate)


# Statistics

stats = tools.Statistics(key=lambda ind: ind.fitness.values)
stats.register("avg", np.mean)
stats.register("std", np.std)
stats.register("min", np.min)
stats.register("max", np.max)


# Optimisation

pop = toolbox.population(n=POP_SIZE)
# pop = [creator.Individual([2] * IND_SIZE) for i in range(POP_SIZE)]
pop, logbook = algorithms.eaSimple(pop, toolbox, cxpb=CXPB, mutpb=MUTPB, ngen=N_GEN, stats=stats, verbose=True)


# K Best solution

kbest = tools.selBest(pop, k=10)
kbest = pd.DataFrame(kbest, columns=GENES).T
kbest = kbest.loc[kbest.mean(1).sort_values(ascending=False).index]
kbest.to_csv(f"{rpath}/ga_kbest_{datetime.datetime.now().strftime('%Y%m%d_%H:%M:%S')}.csv")

#
ind = list(kbest.median(1).astype(int).loc[GENES])
