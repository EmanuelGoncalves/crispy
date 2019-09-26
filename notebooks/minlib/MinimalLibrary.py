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
from minlib.Utils import define_sgrnas_sets
from crispy.CRISPRData import Library, CRISPRDataSet


LOG = logging.getLogger("Crispy")
DPATH = pkg_resources.resource_filename("crispy", "data/")
RPATH = pkg_resources.resource_filename("notebooks", "minlib/reports/")


# Master library (KosukeYusa v1.1 + Avana + Brunello)
#

master_lib = Library.load_library("MasterLib_v1.csv.gz", set_index=False)

# Drop sgRNAs that do not align to GRCh38
master_lib = master_lib.query("Assembly == 'Human (GRCh38)'")

# Drop sgRNAs with no alignment to GRCh38
master_lib = master_lib.dropna(subset=["Approved_Symbol", "Off_Target"])

# Parse off target summaries
master_lib["Off_Target"] = master_lib["Off_Target"].apply(ast.literal_eval).values

# Calculate absolute distance of JACKS scores to 1 (similar to gene sgRNAs mean)
master_lib["JACKS_min"] = abs(master_lib["JACKS"] - 1).values

# Sort guides according to KS scores
master_lib = master_lib.sort_values(
    ["KS", "JACKS_min", "RuleSet2"], ascending=[False, True, False]
)


# List of genes found to have disconcordant fold-changes between minimal and original library
#

discordant_genes = pd.read_excel(f"{RPATH}/MinimalLib_top2_benchmark_gene_concordance.xlsx", index_col=0)
discordant_genes = set(discordant_genes.query("match == 'Disconcordant'").index)


# HGNC protein coding genes list
#

pgenes = pd.read_csv(
    f"{DPATH}/gene_sets/gene_with_protein_product.txt", sep="\t"
).query("status == 'Approved'")


# Defining a minimal library
#

NGUIDES, JACKS_THRES, RULESET2_THRES, REMOVE_DISCORDANT = 2, 1.0, 0.4, True


def select_guides(
    gene,
    offtarget,
    n_guides,
    qflag,
    library=None,
    sortby=None,
    ascending=None,
    query=None,
    dropna=None,
    sgrnas_exclude=None
):
    # Select gene sgRNAs gene
    gguides = master_lib.query(f"Approved_Symbol == '{gene}'")

    # Subset to library
    if library is not None:
        gguides = gguides.query(f"Library == '{library}'")

    # List of sgRNAs to exclude
    if sgrnas_exclude is not None:
        gguides = gguides[~gguides["sgRNA_ID"].isin(sgrnas_exclude)]

    # Drop NaNs
    if dropna is not None:
        gguides = gguides.dropna(subset=dropna)

    # Query
    if query is not None:
        gguides = gguides.query(query)

    # Sort guides
    if (sortby is not None) and (ascending is not None):
        gguides = gguides.sort_values(sortby, ascending=ascending)

    # Off-target filter
    gguides = gguides.loc[
        [
            np.all([ot[i] <= v for i, v in enumerate(offtarget)])
            for ot in gguides["Off_Target"]
        ]
    ]

    # Pick top n sgRNAs
    gguides = gguides.head(n_guides)

    # Quality flag
    gguides = gguides.assign(Confidence=qflag)

    return gguides


minimal_lib = []
for g in pgenes["symbol"]:
    # - Stringest sgRNA selection round
    # KosukeYusa v1.1 sgRNAs
    g_guides = select_guides(
        library="KosukeYusa",
        gene=g,
        offtarget=[1, 0],
        n_guides=0 if REMOVE_DISCORDANT and g in discordant_genes else NGUIDES,
        qflag="Green",
        query=f"JACKS_min < {JACKS_THRES}",
        dropna=["KS"],
    )

    # Top-up with Avana guides
    if len(g_guides) < NGUIDES:
        g_guides_avana = select_guides(
            library="Avana",
            gene=g,
            offtarget=[1, 0],
            n_guides=NGUIDES - g_guides.shape[0],
            qflag="Green",
            query=f"JACKS_min < {JACKS_THRES}",
            dropna=["KS"],
        )
        g_guides = pd.concat([g_guides, g_guides_avana], sort=False)

    # Top-up with Brunello guides
    if len(g_guides) < NGUIDES:
        g_guides_brunello = select_guides(
            library="Brunello",
            gene=g,
            offtarget=[1, 0],
            n_guides=NGUIDES - g_guides.shape[0],
            sortby="RuleSet2",
            ascending=False,
            qflag="Green",
            query=f"RuleSet2 > {RULESET2_THRES}",
        )
        g_guides = pd.concat([g_guides, g_guides_brunello], sort=False)

    # Top-up with TKOv3 guides
    if len(g_guides) < NGUIDES:
        g_guides_tkov3 = select_guides(
            library="TKOv3",
            gene=g,
            offtarget=[1, 0],
            n_guides=NGUIDES - g_guides.shape[0],
            qflag="Green",
        )
        g_guides = pd.concat([g_guides, g_guides_tkov3], sort=False)

    # - Middle sgRNA selection round
    # Top-up with TKOv3 guides
    if len(g_guides) < NGUIDES:
        g_guides_kosuke = select_guides(
            library="KosukeYusa",
            gene=g,
            offtarget=[1],
            n_guides=0 if REMOVE_DISCORDANT and g in discordant_genes else (NGUIDES - g_guides.shape[0]),
            qflag="Amber",
            dropna=["KS"],
            sgrnas_exclude=set(g_guides["sgRNA_ID"])
        )
        g_guides = pd.concat([g_guides, g_guides_kosuke], sort=False)

    # Top-up with Avana guides
    if len(g_guides) < NGUIDES:
        g_guides_avana = select_guides(
            library="Avana",
            gene=g,
            offtarget=[1],
            n_guides=NGUIDES - g_guides.shape[0],
            qflag="Amber",
            dropna=["KS"],
            sgrnas_exclude=set(g_guides["sgRNA_ID"])
        )
        g_guides = pd.concat([g_guides, g_guides_avana], sort=False)

    # Top-up with Brunello guides
    if len(g_guides) < NGUIDES:
        g_guides_brunello = select_guides(
            library="Brunello",
            gene=g,
            offtarget=[1],
            sortby="RuleSet2",
            ascending=False,
            n_guides=NGUIDES - g_guides.shape[0],
            qflag="Amber",
            sgrnas_exclude=set(g_guides["sgRNA_ID"])
        )
        g_guides = pd.concat([g_guides, g_guides_brunello], sort=False)

    # Top-up with TKOv3 guides
    if len(g_guides) < NGUIDES:
        g_guides_tkov3 = select_guides(
            library="TKOv3",
            gene=g,
            offtarget=[1],
            n_guides=NGUIDES - g_guides.shape[0],
            qflag="Amber",
            sgrnas_exclude=set(g_guides["sgRNA_ID"])
        )
        g_guides = pd.concat([g_guides, g_guides_tkov3], sort=False)

    # - Lose sgRNA selection round
    # Pick top 2 guides
    if len(g_guides) < NGUIDES:
        g_guides_red = select_guides(
            gene=g,
            offtarget=[3],
            n_guides=NGUIDES - g_guides.shape[0],
            qflag="Red",
            sgrnas_exclude=set(g_guides["sgRNA_ID"])
        )
        g_guides = pd.concat([g_guides, g_guides_red], sort=False)

    # - Flags
    # Throw warning if no guides have been found
    if g not in master_lib["Approved_Symbol"].values:
        LOG.warning(f"Gene not included: {g}")
        continue

    # Throw warning if no guides passing the QC have been found
    if len(g_guides) < NGUIDES:
        LOG.warning(f"No sgRNA: {g}")
        continue

    # - Append guides
    minimal_lib.append(g_guides)

minimal_lib = pd.concat(minimal_lib)
LOG.info(f"Genes: {minimal_lib['Approved_Symbol'].value_counts().shape[0]}")
LOG.info(minimal_lib["Library"].value_counts())


# Cancer drivers: gene set overlap
#

drive_genes = pd.read_csv(f"{DPATH}/gene_sets/driver_genes_2018-09-13_1322.csv")
drive_genes_miss = drive_genes[
    ~drive_genes["gene_symbol"].isin(minimal_lib["Approved_Symbol"])
]
LOG.warning("Missed driver genes: " + "; ".join(drive_genes_miss["gene_symbol"]))


# Project score genes
#

score_genes = pd.read_csv(f"{DPATH}/gene_sets/essential_project_score.tsv", sep="\t")
score_genes_miss = score_genes[
    ~score_genes["Approved symbol"].isin(minimal_lib["Approved_Symbol"])
]
score_genes_miss = score_genes_miss.dropna(subset=["Approved symbol"])
LOG.warning(
    "Missed Project Score genes: " + "; ".join(score_genes_miss["Approved symbol"])
)


# Add non-targeting sgRNAs
#

ky = CRISPRDataSet("Yusa_v1.1")
ky_counts = ky.counts.remove_low_counts(ky.plasmids)
ky_fc = ky_counts.norm_rpm().norm_rpm().foldchange(ky.plasmids)
ky_gsets = define_sgrnas_sets(ky.lib, ky_fc, add_controls=True)

ctrl_mean = ky_fc.loc[ky_gsets["nontargeting"]["sgrnas"]].mean(1).dropna()
ctrl_mean = abs(ctrl_mean - ctrl_mean.mean())

ctrl_std = ky_fc.loc[ky_gsets["nontargeting"]["sgrnas"]].std(1).dropna()

ctrl_guides = ky.lib.loc[ctrl_mean.sort_values().head(200).index].dropna(axis=1)
ctrl_guides = ctrl_guides.rename(columns={"sgRNA": "WGE_Sequence"})
ctrl_guides = ctrl_guides.reset_index().rename(columns={"sgRNA": "sgRNA_ID"})

minimal_lib = pd.concat([minimal_lib, ctrl_guides], sort=False)
LOG.info(minimal_lib.shape)

# Export
#

minimal_lib.to_csv(
    f"{DPATH}/crispr_libs/MinimalLib_top{NGUIDES}{'_disconcordant' if REMOVE_DISCORDANT else ''}.csv.gz",
    compression="gzip",
    index=False,
)

minimal_lib.to_excel(
    f"{RPATH}/MinimalLib_top{NGUIDES}{'_disconcordant' if REMOVE_DISCORDANT else ''}.xlsx",
    index=False,
)
