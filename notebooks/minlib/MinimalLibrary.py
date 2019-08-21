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
import ast
import logging
import pandas as pd
import pkg_resources
from crispy.CRISPRData import CRISPRDataSet, Library


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

# Drop sgRNAs with more than 1 perfect match and any match with 1 sequence mismatch
master_lib["Off_Target"] = master_lib["Off_Target"].apply(ast.literal_eval).values
master_lib = master_lib[[(i[0] == 1) and (i[1] == 0) for i in master_lib["Off_Target"]]]

# Calculate absolute distance of JACKS scores to 1 (similar to gene sgRNAs mean)
master_lib["JACKS_min"] = abs(master_lib["JACKS"] - 1)

# Sort guides according to KS scores
master_lib = master_lib.sort_values(
    ["KS", "JACKS_min", "RuleSet2"], ascending=[False, True, False]
)


# HGNC protein coding genes list
#

pgenes = pd.read_csv(
    f"{DPATH}/gene_sets/gene_with_protein_product.txt", sep="\t"
).query("status == 'Approved'")


# Defining a minimal library
#

NGUIDES, JACKS_THRES, RULESET2_THRES = 2, 0.5, 0.4

minimal_lib = []
for g in pgenes["symbol"]:
    # KosukeYusa v1.1 sgRNAs
    g_guides = master_lib.query(
        f"(Approved_Symbol == '{g}') & (Library == 'KosukeYusa')"
    ).dropna(subset=["KS"])
    g_guides = g_guides.query(f"JACKS_min < {JACKS_THRES}")
    g_guides = g_guides.head(NGUIDES)

    # Top-up with Avana guides
    if len(g_guides) < NGUIDES:
        g_guides_avana = master_lib.query(
            f"(Approved_Symbol == '{g}') & (Library == 'Avana')"
        ).dropna(subset=["KS"])
        g_guides_avana = g_guides_avana.query(f"JACKS_min < {JACKS_THRES}").head(
            NGUIDES
        )
        g_guides = pd.concat(
            [g_guides, g_guides_avana.head(NGUIDES - g_guides.shape[0])], sort=False
        )

    # Top-up with Brunello guides
    if len(g_guides) < NGUIDES:
        g_guides_brunello = master_lib.query(
            f"(Approved_Symbol == '{g}') & (Library == 'Brunello')"
        )
        g_guides_brunello = g_guides_brunello.query(f"RuleSet2 > {RULESET2_THRES}")
        g_guides_brunello = g_guides_brunello.sort_values("RuleSet2")
        g_guides = pd.concat(
            [g_guides, g_guides_brunello.head(NGUIDES - g_guides.shape[0])], sort=False
        )

    # Top-up with TKOv3 guides
    if len(g_guides) < NGUIDES:
        g_guides_tkov3 = master_lib.query(
            f"(Approved_Symbol == '{g}') & (Library == 'TKOv3')"
        )
        g_guides = pd.concat(
            [g_guides, g_guides_tkov3.head(NGUIDES - g_guides.shape[0])], sort=False
        )

    # Throw warning if no guides have been found
    if g not in master_lib["Approved_Symbol"].values:
        LOG.warning(f"Gene not included: {g}")
        continue

    # Throw warning if no guides passing the QC have been found
    if len(g_guides) < NGUIDES:
        LOG.warning(f"No sgRNA: {g}")
        continue

    minimal_lib.append(g_guides)

minimal_lib = pd.concat(minimal_lib)
LOG.info(f"Genes: {minimal_lib['Approved_Symbol'].value_counts().shape[0]}")
LOG.info(minimal_lib["Library"].value_counts())


# Gencode: gene set overlap
#

sym_check = pd.read_csv(f"{DPATH}/gencode.v30.annotation.gtf.hgnc.map.csv.gz")
sym_check = sym_check.sort_values(["Input", "Match type"])
sym_check = sym_check.groupby("Input").first()

gencode = pd.read_csv(
    f"{DPATH}/gencode.v30.annotation.gtf.gz", sep="\t", skiprows=5, header=None
)
gencode = gencode[gencode[8].apply(lambda v: 'gene_type "protein_coding"' in v)]
gencode = gencode[gencode[2] == "gene"]
gencode = gencode[gencode[0] != "chrM"]
gencode["gene_name"] = (
    gencode[8].apply(lambda v: re.search('gene_name "(.*?)";', v).group(1)).values
)
gencode["Approved_Symbol"] = sym_check.reindex(gencode["gene_name"])[
    "Approved symbol"
].values

gencode_miss = gencode[
    ~gencode["gene_name"].isin(minimal_lib["Approved_Symbol"].values)
]
gencode_miss = gencode_miss[
    gencode_miss[8].apply(lambda v: 'tag "overlapping_locus"' not in v)
]
gencode_miss = gencode_miss[gencode_miss[8].apply(lambda v: " level 3;" not in v)]
gencode_miss = gencode_miss[
    gencode_miss["Approved_Symbol"].isin(pgenes["symbol"].values)
]


# Cancer drivers: gene set overlap
#

drive_genes = pd.read_csv(f"{DPATH}/gene_sets/driver_genes_2018-09-13_1322.csv")
drive_genes_miss = drive_genes[
    ~drive_genes["gene_symbol"].isin(minimal_lib["Approved_Symbol"])
]
LOG.warning("Missed driver genes: " + "; ".join(drive_genes_miss["gene_symbol"]))


# Export
#

minimal_lib.to_csv(
    f"{DPATH}/crispr_libs/MinimalLib_top{NGUIDES}.csv.gz",
    compression="gzip",
    index=False,
)
