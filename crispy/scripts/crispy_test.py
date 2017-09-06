#!/usr/bin/env python
# Copyright (C) 2017 Emanuel Goncalves

import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy.stats import pearsonr
from sklearn.metrics.ranking import roc_auc_score

from scripts.crispy_bagel import fold_changes, bagel_scores

# Define data location
data_dir = "./data/crispr_raw_counts/"

# Calculate fold-changes
fold_changes(data_dir)


# Calculate crispy scores
fcs_file = "%s/gdsc_crispr.foldchanges.tsv" % os.path.dirname(data_dir)
gdsc_bg = bagel_scores(fcs_file, ['OVR4_c905R1', 'OVR4_c905R2'])
# gdsc_bg.to_csv("%s/gdsc_crispr.foldchanges.crispy.tsv" % os.path.dirname(data_dir), float_format="%.6f", sep="\t")
gdsc_bg = pd.read_csv("%s/gdsc_crispr.foldchanges.crispy.tsv" % os.path.dirname(data_dir), index_col=0, sep="\t")

# Import original crispy scores
gdsc_bagel = pd.read_csv("%s/gdsc_crispr.foldchanges.crispy.tsv" % os.path.dirname(data_dir), sep="\t", index_col=0)

# Import Crispr project scores
gdsc_crispr = pd.read_csv("/Users/eg14/Data/sanger/18_BAGEL-R_output/OVCAR-4_GeneLevelBF.csv", sep=",", index_col=0)

# Plot
plot_df = pd.DataFrame({
	"crispy": gdsc_bagel["BF"],
	"crispy": gdsc_bg["mean"],
	"crispr": gdsc_crispr["x"],
	"fc": -gdsc_fc[["OVR4_c905R1", "OVR4_c905R2"]].mean(1).reset_index().groupby("gene").mean()[0]
}).dropna()


def corrfunc(x, y, **kws):
	r, p = pearsonr(x, y)
	ax = plt.gca()
	ax.annotate(
		'R={:.2f}\n{:s}'.format(r, ('p<1e-4' if p < 1e-4 else ('p=%.1e' % p))),
		xy=(.5, .5), xycoords=ax.transAxes, horizontalalignment='center', verticalalignment='center', fontsize=7
	)

sns.set(style='ticks', context='paper', font_scale=0.75, rc={'axes.linewidth': .3, 'xtick.major.width': .3, 'ytick.major.width': .3, 'xtick.major.size': 2.5, 'ytick.major.size': 2.5, 'xtick.direction': 'out', 'ytick.direction': 'out'})
g = sns.PairGrid(plot_df, despine=False, palette=['#34495e'])
g.map_upper(sns.regplot, color='#34495e', truncate=True, scatter_kws={'s': 5, 'edgecolor': 'w', 'linewidth': .3, 'alpha': .8}, line_kws={'linewidth': .5})
g.map_diag(sns.distplot, kde=False, bins=30, hist_kws={'color': '#34495e'})
g.map_lower(corrfunc)
plt.gcf().set_size_inches(2., 2., forward=True)
plt.savefig("./reports/bagel_comparison.png", bbox_inches="tight", dpi=600)
plt.close("all")


# Reference sets
genes_ess = set(pd.read_csv("./resources/curated_BAGEL_essential.csv")["gene"])
genes_non_ess = set(pd.read_csv("./resources/curated_BAGEL_nonEssential.csv")["gene"])

plot_df["ess"] = [int(i in genes_ess) for i in plot_df.index]
plot_df["ness"] = [int(i in genes_non_ess) for i in plot_df.index]

pd.DataFrame({s: {f: roc_auc_score(plot_df[s], (-1 if s == "ness" else 1) * plot_df[f]) for f in ["crispy", "crispr", "crispy", "fc"]} for s in ["ess", "ness"]})
