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

import numpy as np
import pandas as pd
import pkg_resources
import dtrace.DataImporter as DataImporter
from crispy import logger as LOG
from crispy.GuideSelection import GuideSelection

DPATH = pkg_resources.resource_filename("data", "dualguide/")

LIBNAME = "2gCRISPR_Pilot_library_v2.0.0.xlsx"

LINKER = "GCAGAG"
TRNA = "GCATTGGTGGTTCAGTGGTAGAATTCTCGCCTCCCACGCGGGAGACCCGGGTTCAATTCCCGGCCAATGCA"
SCAFFOLDS = pd.read_excel(f"{DPATH}/{LIBNAME}", sheet_name="Scaffolds", index_col=0)["Sequence"].to_dict()
CLONING_ARMS = pd.read_excel(f"{DPATH}/{LIBNAME}", sheet_name="Cloning Arms", index_col=0)["sequence"].to_dict()

LIB_COLUMNS = [
    "WGE_ID",
    "WGE_Sequence",
    "Library",
    "Approved_Symbol",
    "Off_Target",
    "Chr",
    "Start",
    "End",
    "Strand",
    "Confidence",
]
LIB_COLUMNS_EXP = [f"{p}_{i}" for p in ["sgRNA1", "sgRNA2"] for i in LIB_COLUMNS]


def get_nontargeting_sgrnas():
    return pd.read_excel(f"{DPATH}/{LIBNAME}", sheet_name="Non-targeting sgRNAs").loc[:, LIB_COLUMNS]


def get_intergenic_sgrnas():
    sgrnas = pd.read_excel(f"{DPATH}/{LIBNAME}", sheet_name="Intergenic sgRNAs").loc[:, LIB_COLUMNS]
    sgrnas = sgrnas[["TTTT" not in g for g in sgrnas["WGE_Sequence"]]]
    return sgrnas


def get_nonessential_sgrnas(n_guides=2):
    genes = set(pd.read_excel(f"{DPATH}/{LIBNAME}", sheet_name="Non-essential Genes")["Genes"])

    sgrnas = []
    for g in genes:
        LOG.info(f"Non-essential genes sgRNAs: {g}")
        g_sgrnas = gselection.selection_rounds(g, n_guides=n_guides)[LIB_COLUMNS]

        if g_sgrnas.shape[0] == n_guides:
            sgrnas.append(g_sgrnas)

    sgrnas = pd.concat(sgrnas)[LIB_COLUMNS].reset_index(drop=True)

    return sgrnas


def generate_guides_gi(n_guides=5):
    """
    Vectors for gene interaction pairs.

    :param n_guides:
    :return:
    """
    # Original gene interaction pairs
    gi_pairs = pd.read_excel(f"{DPATH}/{LIBNAME}", sheet_name="Gene Interaction Pairs")

    # Build sgRNAs
    gi_sgrnas = []

    for g1, g2 in gi_pairs[["Gene1", "Gene2"]].values:
        LOG.info(f"Gene-interations sgRNAs: {g1}, {g2}")

        g1_sgrnas = (
            gselection.selection_rounds(g1, n_guides=n_guides)[LIB_COLUMNS]
            .add_prefix("sgRNA1_")
            .assign(dummy=1)
        )
        g2_sgrnas = (
            gselection.selection_rounds(g2, n_guides=n_guides)[LIB_COLUMNS]
            .add_prefix("sgRNA2_")
            .assign(dummy=1)
        )

        gi_sgrnas.append(
            g1_sgrnas.merge(g2_sgrnas, how="outer", on="dummy")[LIB_COLUMNS_EXP]
        )

    gi_sgrnas = pd.concat(gi_sgrnas)[LIB_COLUMNS_EXP].reset_index(drop=True)

    return gi_sgrnas


def generate_negative_controls(n_guides=2):
    """
    Vectors for negative controls.

    :return:
    """

    # Non-essential vs Non-essential
    noess = get_nonessential_sgrnas(n_guides=n_guides)
    noess_sgrnas = pd.concat(
        [
            noess.add_prefix("sgRNA1_"),
            noess.iloc[::-1].add_prefix("sgRNA2_").reset_index(drop=True),
        ],
        axis=1,
    )[LIB_COLUMNS_EXP]

    # Non-targeting vs Non-targeting
    ntarg = get_nontargeting_sgrnas()
    ntarg_sgrnas = pd.concat(
        [
            ntarg.add_prefix("sgRNA1_"),
            ntarg.iloc[::-1].add_prefix("sgRNA2_").reset_index(drop=True),
        ],
        axis=1,
    )[LIB_COLUMNS_EXP]

    # Intergenic vs Intergenic
    inter = get_intergenic_sgrnas()
    inter_sgrnas = pd.concat(
        [
            inter.add_prefix("sgRNA1_"),
            inter.iloc[::-1].add_prefix("sgRNA2_").reset_index(drop=True),
        ],
        axis=1,
    )[LIB_COLUMNS_EXP]

    # Concat sgRNAs
    ng_sgrnas = pd.concat(
        [noess_sgrnas, ntarg_sgrnas, inter_sgrnas], axis=0, sort=False
    ).reset_index(drop=True)[LIB_COLUMNS_EXP]

    return ng_sgrnas


def generate_copy_number(cn_thres=6, n_guides=2):
    """
    Vectors to target copy-number amplified regions.

    :param cn_thres:
    :param n_guides:
    :return:
    """

    # HT-29 copy number
    cn = DataImporter.CopyNumber().filter(subset=["SIDM00136"]).iloc[:, 0]

    # Identify set of copy nubmer amplified genes
    hgnc = pd.read_csv(f"{DPATH}/protein-coding_gene.txt", sep="\t", index_col="symbol")

    genes = pd.concat([hgnc["location"], cn.rename("cn")], axis=1, sort=False)
    genes = genes.query(f"cn >= {cn_thres}").dropna().sort_values("cn", ascending=False)

    # sgRNAs targeting copy number amplified genes
    cn_sgrnas = []
    for g in genes.index:
        LOG.info(f"Copy number amplified: {g}")
        g_sgrnas = gselection.selection_rounds(g, n_guides=n_guides)

        if g_sgrnas.shape[0] == n_guides:
            cn_sgrnas.append(g_sgrnas)
    cn_sgrnas = pd.concat(cn_sgrnas)[LIB_COLUMNS].reset_index(drop=True)

    pindex = np.arange(int(len(cn_sgrnas) / n_guides)).repeat(n_guides)

    # copy number vs copy number
    cn_cn_sgrnas = pd.concat(
        [
            cn_sgrnas.add_prefix("sgRNA1_"),
            cn_sgrnas.iloc[::-1].add_prefix("sgRNA2_").reset_index(drop=True),
        ],
        axis=1,
    )[LIB_COLUMNS_EXP]

    # copy number vs non-essential
    noess = get_nonessential_sgrnas(n_guides=n_guides)
    cn_noess_sgrnas = pd.concat(
        [
            cn_sgrnas.add_prefix("sgRNA1_"),
            noess.iloc[pindex].add_prefix("sgRNA2_").reset_index(drop=True),
        ],
        axis=1,
    )[LIB_COLUMNS_EXP]

    # copy number vs non-targeting
    ntarg = get_nontargeting_sgrnas()
    cn_ntarg_sgrnas = pd.concat(
        [
            cn_sgrnas.add_prefix("sgRNA1_"),
            ntarg.iloc[pindex].add_prefix("sgRNA2_").reset_index(drop=True),
        ],
        axis=1,
    )[LIB_COLUMNS_EXP]

    # copy number vs intergenic
    inter = get_intergenic_sgrnas()
    cn_inter_sgrnas = pd.concat(
        [
            cn_sgrnas.add_prefix("sgRNA1_"),
            inter.iloc[pindex].add_prefix("sgRNA2_").reset_index(drop=True),
        ],
        axis=1,
    )[LIB_COLUMNS_EXP]

    # Concat sgRNAs
    cn_sgrnas = pd.concat(
        [cn_cn_sgrnas, cn_noess_sgrnas, cn_ntarg_sgrnas, cn_inter_sgrnas],
        axis=0,
        sort=False,
    ).reset_index(drop=True)[LIB_COLUMNS_EXP]

    return cn_sgrnas


def generate_vector_position(n_guides=2):
    """
    Vectors to test scaffold position.

    :param n_guides:
    :return:
    """

    # Read loss of fitness genes used on previous library iteration
    genes = set(pd.read_excel(f"{DPATH}/{LIBNAME}", sheet_name="Essential Genes")["Genes"])

    # Get sgRNAs
    vpos_sgrnas = []
    for g in genes:
        LOG.info(f"Backbone position: {g}")
        g_sgrnas = gselection.selection_rounds(g, n_guides=n_guides)

        if g_sgrnas.shape[0] == n_guides:
            vpos_sgrnas.append(g_sgrnas)
    vpos_sgrnas = pd.concat(vpos_sgrnas)[LIB_COLUMNS].reset_index(drop=True)

    pindex = np.arange(int(len(vpos_sgrnas) / n_guides)).repeat(n_guides)

    # Controls = non-targeting + intergenic
    control = pd.concat(
        [get_nontargeting_sgrnas(), get_intergenic_sgrnas()], axis=0, sort=False
    )[LIB_COLUMNS].reset_index(drop=True)

    # loss of fitness VS control
    vpos_sgrna1 = pd.concat(
        [
            vpos_sgrnas.add_prefix("sgRNA1_"),
            control.iloc[pindex].add_prefix("sgRNA2_").reset_index(drop=True),
        ],
        axis=1,
    )[LIB_COLUMNS_EXP]

    # control VS loss of fitness
    vpos_sgrna2 = pd.concat(
        [
            control.iloc[pindex].add_prefix("sgRNA1_").reset_index(drop=True),
            vpos_sgrnas.add_prefix("sgRNA2_"),
        ],
        axis=1,
    )[LIB_COLUMNS_EXP]

    # Concat sgRNAs
    vpos_sgrnas = pd.concat([vpos_sgrna1, vpos_sgrna2], axis=0, sort=False).reset_index(
        drop=True
    )[LIB_COLUMNS_EXP]

    return vpos_sgrnas


def generate_anchors(n_guides=4):
    """
    GI interaction library anchor sgRNAs.

    :param n_guides:
    :return:
    """

    # Read anchor genes
    genes = set(pd.read_excel(f"{DPATH}/{LIBNAME}", sheet_name="Anchors")["Genes"])

    # Get sgRNAs
    anchors = []
    for g in genes:
        LOG.info(f"Anchors: {g}")
        g_sgrnas = gselection.selection_rounds(g, n_guides=n_guides)[LIB_COLUMNS]

        if g_sgrnas.shape[0] == n_guides:
            anchors.append(g_sgrnas)
        else:
            LOG.warn(f"Not enough sgRNAs: Anchor={g}")
    anchors = pd.concat(anchors)[LIB_COLUMNS].reset_index(drop=True)

    pindex = np.arange(int(len(anchors) / n_guides)).repeat(n_guides)

    # Non-targeting VS Anchor
    ntarg = get_nontargeting_sgrnas()
    ntarg_sgrnas = pd.concat(
        [
            ntarg.iloc[pindex].add_prefix("sgRNA1_").reset_index(drop=True),
            anchors.add_prefix("sgRNA2_"),
        ],
        axis=1,
    )[LIB_COLUMNS_EXP]

    # Intergenic VS Anchor
    inter = get_intergenic_sgrnas()
    inter_sgrnas = pd.concat(
        [
            inter.iloc[pindex].add_prefix("sgRNA1_").reset_index(drop=True),
            anchors.add_prefix("sgRNA2_"),
        ],
        axis=1,
    )[LIB_COLUMNS_EXP]

    # Concat sgRNAs
    anchors_sgrnas = pd.concat(
        [ntarg_sgrnas, inter_sgrnas], axis=0, sort=False
    ).reset_index(drop=True)[LIB_COLUMNS_EXP]

    return anchors_sgrnas


def generate_cut_distance():
    # Read cut distance vectors from previous library iteration
    sgrnas = pd.read_excel(f"{DPATH}/{LIBNAME}", sheet_name="Cut Distance Vectors")
    sgrnas = sgrnas[["TTTT" not in g for g in sgrnas["sgRNA1_WGE_Sequence"]]]
    sgrnas = sgrnas[["TTTT" not in g for g in sgrnas["sgRNA2_WGE_Sequence"]]]
    return sgrnas


def generate_oligo(sgrna1, scaffold, linker, trna, sgrna2):
    sgrna1 = sgrna1[1:-3] if len(sgrna1) == 23 else sgrna1
    sgrna2 = sgrna2[1:-3] if len(sgrna2) == 23 else sgrna2

    oligo = f"{CLONING_ARMS['left']}G{sgrna1}{scaffold}{linker}{trna}G{sgrna2}{CLONING_ARMS['right']}"

    return oligo


if __name__ == "__main__":
    # Guide selection
    #
    gselection = GuideSelection()

    # Pilot library
    #
    lib = pd.concat(
        [
            generate_guides_gi(n_guides=4).assign(Notes="GIpairs"),
            generate_anchors(n_guides=4).assign(Notes="Anchors"),
            generate_copy_number(n_guides=2, cn_thres=6).assign(Notes="CopyNumber"),
            generate_vector_position(n_guides=2).assign(Notes="ScaffoldPosition"),
            generate_negative_controls(n_guides=2).assign(Notes="NegControls"),
            generate_cut_distance().assign(Notes="DistanceCut"),
        ],
        axis=0,
        sort=False,
    )

    # Add different scaffolds
    lib_scaffolds = []

    for s in SCAFFOLDS:
        if s == "Wildtype":
            lib_s = lib.copy()

        else:
            lib_s = lib[lib["Notes"].isin(["ScaffoldPosition", "NegControls"])]

        lib_s = lib_s.assign(Scaffold_Sequence=SCAFFOLDS[s]).assign(Scaffold=s)
        lib_scaffolds.append(lib_s)

    lib = pd.concat(lib_scaffolds, axis=0, sort=False).reset_index(drop=True)

    # Add Linker and tRNA sequence
    lib = lib.assign(Linker=LINKER)
    lib = lib.assign(tRNA=TRNA)

    # Drop duplicates
    vids = ["sgRNA1_WGE_Sequence", "Scaffold_Sequence", "Linker", "tRNA", "sgRNA2_WGE_Sequence"]
    lib = lib.drop_duplicates(subset=vids)

    # Add 2gCRISPR ids to each vector
    lib["ID"] = [f"DG{i:07d}" for i in np.arange(1, len(lib) + 1)]

    # Order columns
    lib = lib[["ID", "Notes", "Scaffold"] + LIB_COLUMNS_EXP + ["Scaffold_Sequence", "Linker", "tRNA"]]

    # Oligo sequences
    lib["Oligo_Sequence"] = [generate_oligo(sg1, s, l, t, sg2) for sg1, s, l, t, sg2 in lib[vids].values]

    # Export
    lib.to_excel(f"{DPATH}/{LIBNAME[:-5]}_guides_only.xlsx", index=False)
