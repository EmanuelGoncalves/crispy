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

import re
import numpy as np
import pandas as pd
from minlib import LOG, rpath, dpath
from crispy.CRISPRData import CRISPRDataSet, Library


# Master library (KosukeYusa v1.1 + Avana + Brunello)

mlib = Library.load_library("master.csv.gz", set_index=False)
mlib = mlib.sort_values(
    ["ks_control_min", "jacks_min", "doenchroot"], ascending=[True, True, False]
)


# Protein coding genes
pgenes = pd.read_csv(f"{dpath}/hgnc_protein_coding_set.txt", sep="\t")


# Defining a minimal library

n_guides = 2

minimal_lib = []

for g in pgenes["symbol"]:
    # KosukeYusa v1.1 sgRNAs
    g_guides = mlib.query(f"(Gene == '{g}') & (lib == 'KosukeYusa')")
    g_guides = g_guides[g_guides["jacks_min"] < 1.5].head(n_guides)

    # Top-up with Avana guides
    if len(g_guides) < n_guides:
        g_guides_avana = mlib.query(f"(Gene == '{g}') & (lib == 'Avana')")
        g_guides_avana = g_guides_avana[g_guides_avana["jacks_min"] < 1.5].head(n_guides)
        g_guides = pd.concat([g_guides, g_guides_avana.head(n_guides - g_guides.shape[0])], sort=False)

    # Top-up with Brunello guides
    if len(g_guides) < n_guides:
        g_guides_brunello = mlib.query(f"(Gene == '{g}') & (lib == 'Brunello')")
        g_guides_brunello = g_guides_brunello.query("doenchroot > .4")
        g_guides_brunello = g_guides_brunello.sort_values("doenchroot")
        g_guides = pd.concat([g_guides, g_guides_brunello.head(n_guides - g_guides.shape[0])], sort=False)

    # Top-up with TKOv3 guides
    if len(g_guides) < n_guides:
        g_guides_tkov3 = mlib.query(f"(Gene == '{g}') & (lib == 'TKOv3')")
        g_guides = pd.concat([g_guides, g_guides_tkov3.head(n_guides - g_guides.shape[0])], sort=False)

    if g not in mlib["Gene"].values:
        # Throw warning if no guides have been found
        LOG.warning(f"Gene not included: {g}")

    elif len(g_guides) < n_guides:
        # Throw warning if no guides passing the QC have been found
        LOG.warning(f"No sgRNA: {g}")

    minimal_lib.append(g_guides)

minimal_lib = pd.concat(minimal_lib)
minimal_lib["lib"] = [
    "LanderSabatini" if g[:2] == "sg" else l for g, l in minimal_lib[["sgRNA_ID", "lib"]].values
]
LOG.info(minimal_lib["lib"].value_counts())


# Gencode

sym_check = pd.read_csv(f"{dpath}/gencode.v30.annotation.gtf.hgnc.map.csv.gz")
sym_check = sym_check.sort_values(["Input", "Match type"])
sym_check = sym_check.groupby("Input").first()

gencode = pd.read_csv(f"{dpath}/gencode.v30.annotation.gtf.gz", sep="\t", skiprows=5, header=None)
gencode = gencode[gencode[8].apply(lambda v: 'gene_type "protein_coding"' in v)]
gencode = gencode[gencode[2] == "gene"]
gencode = gencode[gencode[0] != "chrM"]
gencode["gene_name"] = gencode[8].apply(lambda v: re.search('gene_name "(.*?)";', v).group(1)).values
gencode["Approved symbol"] = sym_check.reindex(gencode["gene_name"])["Approved symbol"].values

gencode_miss = gencode[~gencode["gene_name"].isin(minimal_lib["Gene"].values)]
gencode_miss = gencode_miss[gencode_miss[8].apply(lambda v: 'tag "overlapping_locus"' not in v)]
gencode_miss = gencode_miss[gencode_miss[8].apply(lambda v: ' level 3;' not in v)]
gencode_miss = gencode_miss[gencode_miss["Approved symbol"].isin(pgenes["symbol"].values)]


# Export

minimal_lib.to_csv(f"{dpath}/crispr_libs/minimal.csv.gz", compression="gzip", index=False)
