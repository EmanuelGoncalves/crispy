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
import itertools
import numpy as np
import pandas as pd
import seaborn as sns
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
from sklearn.metrics import mean_squared_error, matthews_corrcoef, recall_score
from minlib import dpath, rpath, LOG, sgrnas_scores_scatter, project_score_sample_map
from deap import base, creator, tools, algorithms


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

JACKS_THRES = 1.

ky_v11_wge = pd.read_csv(
    f"{dpath}/crispr_libs/Yusa_v1.1_WGE_map.txt", sep="\t", index_col=0
)
ky_v11_wge = ky_v11_wge[~ky_v11_wge["WGE_sequence"].isna()]


mlib_min = mlib.query(f"jacks_min < {JACKS_THRES}").sort_values("ks_control_min")
mlib_min = mlib_min[
    ~mlib_min.index.isin(ky_v11_wge[ky_v11_wge["Assembly"] == "Grch19"].index)
]

mlib_min_dict = {g: df for g, df in mlib_min.groupby("Gene")}


# Genes

GENES = natsorted(set(mlib_min["Gene"]))
LOG.info(f"Genes: {len(GENES)}")


# All sgRNAs fold-changes

def calculate_gene_fc(sgrnas):
    fc = KY_COUNTS.loc[sgrnas].norm_rpm().foldchange(KY.plasmids)
    fc_gene = fc.groupby(mlib.loc[fc.index, "Gene"]).mean()
    fc_gene_avg = fc_gene.groupby(KY_SMAP["model_id"], axis=1).mean()
    return fc_gene_avg


def binarise_fc(fcs):
    fcs_thres = [QCplot.aroc_threshold(fcs.loc[GENES, s], fpr_thres=0.01)[1] for s in fcs]
    fcs_bin = (fcs < fcs_thres).astype(int)
    return fcs_bin


KY_FC = calculate_gene_fc(mlib.index)
KY_BIN = binarise_fc(KY_FC)


# Samples

SAMPLES = natsorted(set(KY_BIN))
LOG.info(f"Samples: {len(SAMPLES)}")


# Individual subset library

def individual_lib(ind):
    ind = dict(zip(*(GENES, ind)))
    ind_sgrnas = list(
        itertools.chain.from_iterable(
            [mlib_min_dict[g].head(ind[g]).index for g in mlib_min_dict]
        )
    )
    return ind_sgrnas


# Recall dependencies

def recap_dependencies(all_bin, ind_bin):
    all_bin, ind_bin = all_bin.loc[GENES, SAMPLES], ind_bin.loc[GENES, SAMPLES]
    ind_recalls = [recall_score(all_bin.iloc[:, i], ind_bin.iloc[:, i]) for i, _ in enumerate(SAMPLES)]
    return ind_recalls


# GA Params

N_GEN = 500
POP_SIZE = 10
CXPB = 0.5
MUTPB = 0.25
INDPB = 0.3
IND_SIZE = len(GENES)


# Types

creator.create("Fitness", base.Fitness, weights=(-1.0, 1.0))
creator.create("Individual", list, fitness=creator.Fitness)


# Initialisation

toolbox = base.Toolbox()
toolbox.register("attribute", np.random.randint, low=2, high=4)
toolbox.register(
    "individual", tools.initRepeat, creator.Individual, toolbox.attribute, n=IND_SIZE
)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)


# Operators

def evaluate(ind):
    ind_sgrnas = individual_lib(ind)
    ind_fc = calculate_gene_fc(ind_sgrnas)
    ind_bin = binarise_fc(ind_fc)
    ind_recall = recap_dependencies(KY_BIN, ind_bin)
    ind_recall_90 = sum(np.array(ind_recall) > 0.9) / len(ind_recall)

    return sum(ind), np.round(ind_recall_90, 2)


toolbox.register("mate", tools.cxTwoPoint)

toolbox.register("mutate", tools.mutUniformInt, low=2, up=4, indpb=INDPB)

# toolbox.register("select", tools.selTournament, tournsize=10)
toolbox.register("select", tools.selNSGA2)

toolbox.register("evaluate", evaluate)


# Statistics

stats = tools.Statistics(key=lambda ind: ind.fitness.values)
stats.register("avg", np.mean, axis=0)
stats.register("std", np.std, axis=0)
stats.register("min", np.min, axis=0)
stats.register("max", np.max, axis=0)


# Optimisation

pop = toolbox.population(n=POP_SIZE)
pop = algorithms.eaSimple(
    pop, toolbox, cxpb=CXPB, mutpb=MUTPB, ngen=N_GEN, stats=stats
)
