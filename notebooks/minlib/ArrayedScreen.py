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

import ast
import logging
import numpy as np
import pandas as pd
import pkg_resources
import seaborn as sns
import matplotlib.pyplot as plt
from Bio.Seq import Seq
from Bio.Alphabet import generic_dna
from DataImporter import Sample, GeneExpression
from crispy.CRISPRData import Library, CRISPRDataSet
from minlib.Utils import define_sgrnas_sets, project_score_sample_map


LOG = logging.getLogger("Crispy")
DPATH = pkg_resources.resource_filename("crispy", "data/")
RPATH = pkg_resources.resource_filename("notebooks", "minlib/reports/")


# Project Score samples acquired with Kosuke_Yusa v1.1 library
#

ss = Sample().samplesheet

scoress = pd.read_excel(f"{DPATH}/score_samplesheet.xlsx", index_col=0)
scoress["transduction_efficiencty"] = scoress["transduction_efficiencty"].apply(lambda v: np.mean([float(i) if i != "NA" else np.nan for i in str(v).split(", ")]))
scoress["cas9_activity"] = scoress["cas9_activity"].apply(lambda v: np.mean([float(i) if i != "NA" else np.nan for i in str(v).split(", ")]))

ky = CRISPRDataSet("Yusa_v1.1")

ky_smap = project_score_sample_map()

ky_counts = ky.counts.remove_low_counts(ky.plasmids)

ky_fc = ky_counts.norm_rpm().foldchange(ky.plasmids)
ky_fc = ky_fc.groupby(ky_smap["model_id"], axis=1).mean()

ky_gsets = define_sgrnas_sets(ky.lib, ky_fc, add_controls=True)

ky_delta = (
    ky_fc.loc[ky_gsets["nontargeting"]["sgrnas"]].median()
    - ky_fc.loc[ky_gsets["nonessential"]["sgrnas"]].median()
)
ky_delta = (
    pd.concat(
        [
            ky_delta.rename("delta"),
            ss[
                [
                    "model_name",
                    "cancer_type",
                    "tissue",
                    "msi_status",
                    "growth",
                    "ploidy",
                ]
            ],
            scoress[["transduction_efficiencty", "cas9_activity"]],
        ],
        axis=1,
        sort=False,
    )
    .dropna()
    .sort_values("delta", ascending=False)
)


# sgRNA sets histograms
#

for dtype in ["head", "tail"]:
    df = ky_delta.head(10) if dtype == "head" else ky_delta.tail(5)

    for s in df.index:
        plt.figure(figsize=(2.5, 1.5))

        for c in ky_gsets:
            LOG.info(f"{dtype} - {s} - {c}")

            plot_df = ky_fc[s].reindex(ky_gsets[c]["sgrnas"]).dropna()

            sns.distplot(
                plot_df,
                hist=False,
                label=c,
                kde_kws={"cut": 0, "shade": True},
                color=ky_gsets[c]["color"],
            )

        plt.grid(True, ls=":", lw=0.1, alpha=1.0, zorder=0, axis="x")
        plt.xlabel("sgRNAs fold-change")
        plt.ylabel("Density")
        plt.title(f"{ss.loc[s, 'model_name']} - {s} - {ss.loc[s, 'tissue']}")
        plt.legend(frameon=False)
        plt.savefig(
            f"{RPATH}/arrayed_distributions_{dtype}_{s}.pdf",
            bbox_inches="tight",
            transparent=True,
        )
        plt.close("all")


# Gene-expression
#

gexp = GeneExpression().readcount


# Master library (KosukeYusa v1.1 + Avana + Brunello + TKOv3)
#

master_lib = Library.load_library("MasterLib_v1.csv.gz", set_index=False).dropna(
    subset=["WGE_Sequence"]
)

# Consider only sgRNAs from KosukeYusa lib and mapping to GRCh38
mlib = master_lib.query("Library == 'KosukeYusa'").query("Assembly == 'Human (GRCh38)'")

# Remove sgRNAs that match to multiple genes
sg_count = mlib.groupby("WGE_Sequence")["Approved_Symbol"].agg(lambda v: len(set(v)))
sg_count = sg_count[sg_count != 1]
mlib = mlib[~mlib["WGE_Sequence"].isin(sg_count.index)]

# Remove sgRNAs with no alignment to GRCh38
mlib = mlib.dropna(subset=["Approved_Symbol", "Off_Target"])

# Parse off target summaries
mlib["Off_Target"] = mlib["Off_Target"].apply(ast.literal_eval).values

# Calculate absolute distance of JACKS scores to 1 (similar to gene sgRNAs mean)
mlib["JACKS_min"] = abs(mlib["JACKS"] - 1).values

# Sort guides according to KS scores
mlib = mlib.sort_values(["KS", "JACKS_min", "RuleSet2"], ascending=[False, True, False])


# Samples
#

samples = ["SIDM01072", "SIDM00264"]
ness_ks = mlib[mlib["sgRNA_ID"].isin(ky_gsets["nonessential"]["sgrnas"])]["KS"].mean()


# Positive controls
#

pcontrols = [1038831324, 1038835085, 1038832451, 1142347159]


# Negative controls
#

ncontrols = [1033379490, 1033382343]


stop_guides = mlib[["TTTT" in i for i in mlib["WGE_Sequence"]]].dropna(subset=["KS"]).set_index("sgRNA_ID")
stop_guides = stop_guides[[np.all([ot[i] == j for i, j in enumerate([1, 0, 0])]) for ot in stop_guides["Off_Target"]]]
stop_guides = pd.concat([
    stop_guides,
    ky_fc.reindex(stop_guides.index)[samples],
], axis=1).sort_values("KS")

ncontrols = ncontrols + list(stop_guides.head(5)["WGE_ID"].astype(int))


# Test guides
#

tguides_df = gexp[samples].copy()
tguides_df = tguides_df[(tguides_df <= 0).sum(1) == 2]
tguides_df = mlib[mlib["Approved_Symbol"].isin(tguides_df.index)].dropna(subset=["KS"]).set_index("sgRNA_ID")
tguides_df = pd.concat([tguides_df, ky_fc.reindex(tguides_df.index)[samples]], axis=1)
tguides_df = tguides_df[[np.all([ot[i] == j for i, j in enumerate([1, 0, 0])]) for ot in tguides_df["Off_Target"]]]
tguides_df = tguides_df.sort_values("KS")

low_ks = list(tguides_df.head(40)["WGE_ID"].astype(int))
mid_ks = list(tguides_df.loc[abs(tguides_df["KS"] - ness_ks).sort_values().index].head(40)["WGE_ID"].astype(int))


# Assemble
#

guide_list = [("positive", pcontrols), ("negative", ncontrols), ("low_ks", low_ks), ("mid_ks", mid_ks)]
guide_list = pd.concat([mlib[mlib["WGE_ID"].isin(gl)].assign(gtype=n) for n, gl in guide_list], ignore_index=True)
guide_list = pd.concat([guide_list, master_lib.query(f"(WGE_ID == 1038832451) & (Library == 'Avana')").assign(gtype="positive")], ignore_index=False, sort=False)
guide_list = pd.concat([guide_list.set_index("sgRNA_ID"), ky_fc[samples].reindex(guide_list["sgRNA_ID"])], axis=1)

guide_list.groupby("gtype")[samples + ["KS", "FORECAST", "RuleSet2"]].mean()

guide_list.sort_values(["gtype", "KS"]).to_excel(f"{RPATH}/arrayed_screen_guide_list.xlsx")
