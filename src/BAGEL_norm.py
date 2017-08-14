#!/usr/bin/env python
# Copyright (C) 2017 Emanuel Goncalves

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy_sugar.stats import quantile_gaussianize
from sklearn.metrics.ranking import roc_curve, auc

cdir = "/Users/eg14/Data/crispr_drug/RawData/Lib1.1/"

# cfiles = [
# 	"HT29_c907R2_D8.read_count_foldchange.tsv", "HT29_c907R2_D10_Dabraf.read_count_foldchange.tsv", "HT29_c907R2_D10_DMSO.read_count_foldchange.tsv",
# 	"HT29_c907R2_D14_Dabraf.read_count_foldchange.tsv", "HT29_c907R2_D14_DMSO.read_count_foldchange.tsv", "HT29_c907R2_D18_Dabraf.read_count_foldchange.tsv",
# 	"HT29_c907R2_D18_DMSO.read_count_foldchange.tsv", "HT29_c907R2_D21_Dabraf.read_count_foldchange.tsv", "HT29_c907R2_D21_DMSO.read_count_foldchange.tsv"
# ]

coreEss = set(pd.read_csv("/Users/eg14/Data/sanger/curated_BAGEL_essential.csv")["Gene"])

cfiles = [
    "HT29_c907R2_D8.read_count_foldchange_bf.tsv",
    "HT29_c907R2_D10_Dabraf.read_count_foldchange_bf.tsv", "HT29_c907R2_D10_DMSO.read_count_foldchange_bf.tsv",
    "HT29_c907R2_D14_Dabraf.read_count_foldchange_bf.tsv", "HT29_c907R2_D14_DMSO.read_count_foldchange_bf.tsv",
    "HT29_c907R2_D18_Dabraf.read_count_foldchange_bf.tsv", "HT29_c907R2_D18_DMSO.read_count_foldchange_bf.tsv",
    "HT29_c907R2_D21_Dabraf.read_count_foldchange_bf.tsv", "HT29_c907R2_D21_DMSO.read_count_foldchange_bf.tsv"
]

# - Assemble BFs
bf_scores = pd.DataFrame(
    {f.split(".")[0]: pd.read_csv("%s/%s" % (cdir, f), index_col=1)["BF"].drop("NON-TARGETING") for f in cfiles})

aucs_pre = []
# c = "HT29_c907R2_D21_Dabraf"
for c in bf_scores:
    df = pd.DataFrame({"score": bf_scores[c]})
    df["ess"] = [int(i in coreEss) for i in df.index]

    curve_fpr, curve_tpr, _ = roc_curve(df["ess"], df["score"])

    aucs_pre.append(auc(curve_fpr, curve_tpr))
print(np.mean(aucs_pre))

# - Normalise
bf_scores = pd.DataFrame({c: quantile_gaussianize(bf_scores[c].values) for c in bf_scores}, index=bf_scores.index)

aucs_pre = []
# c = "HT29_c907R2_D21_Dabraf"
for c in bf_scores:
    df = pd.DataFrame({"score": bf_scores[c]})
    df["ess"] = [int(i in coreEss) for i in df.index]

    curve_fpr, curve_tpr, _ = roc_curve(df["ess"], df["score"])

    aucs_pre.append(auc(curve_fpr, curve_tpr))
print(np.mean(aucs_pre))

bf_scores.to_csv("/Users/eg14/Data/crispr_drug/ht29_bagel.csv")

plot_df = bf_scores[['EGFR' in i for i in bf_scores.index]]
plot_df = plot_df.unstack().reset_index()
plot_df.columns = ["condition", "gene", "score"]
plot_df = plot_df[[not i.endswith("_D8") for i in plot_df["condition"]]]
plot_df["treatment"] = ["DMSO" if "DMSO" in i else "Dabraf" for i in plot_df["condition"]]

pal = {"Dabraf": "#e48066", "DMSO": "#cccccc"}

sns.set(style='ticks', context='paper',
        rc={'axes.linewidth': .3, 'xtick.major.width': .3, 'ytick.major.width': .3, 'xtick.major.size': 2.5,
            'ytick.major.size': 2.5}, font_scale=0.75)
sns.boxplot("treatment", "score", data=plot_df, linewidth=.3, sym='', palette=pal)
sns.stripplot('treatment', 'score', data=plot_df, linewidth=.3, edgecolor='white', jitter=.2, size=1.5, palette=pal)
plt.axhline(0, ls='-', lw=0.3, c='#95a5a6', alpha=.5)
plt.xlabel("EGFR")
plt.ylabel("Bagel essentiality score")
plt.gcf().set_size_inches(1., 2)
plt.savefig('./reports/crispr_dabraf.png', bbox_inches='tight', dpi=600)
plt.close('all')
