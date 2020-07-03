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
from crispy.Utils import Utils
from crispy import logger as LOG
from crispy.GuideSelection import GuideSelection
from crispy.CRISPRData import CRISPRDataSet, ReadCounts

DPATH = pkg_resources.resource_filename("data", "dualguide/")

GILIB1NAME = "ENCORE_GI_Library_1.xlsx"
PILOTLIBNAME = "2gCRISPR_Pilot_library_v2.0.0.xlsx"

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

LINKER = "GCAGAG"
TRNA = "GCATTGGTGGTTCAGTGGTAGAATTCTCGCCTCCCACGCGGGAGACCCGGGTTCAATTCCCGGCCAATGCA"
SCAFFOLD = pd.read_excel(
    f"{DPATH}/{PILOTLIBNAME}", sheet_name="Scaffolds", index_col=0
).loc["Wildtype", "Sequence"]
CLONING_ARMS = pd.read_excel(f"{DPATH}/{PILOTLIBNAME}", sheet_name="Cloning Arms", index_col=0)["sequence"].to_dict()


def genes_master_check(geneset):
    master_genes = set(gselection.masterlib["Approved_Symbol"])
    genes_in_master = geneset.intersection(master_genes)

    if len(genes_in_master) != len(geneset):
        genes_not_master = geneset.difference(master_genes)
        assert False, f"Genes not covered in master library: {genes_not_master}"


def generate_guides(geneset, n_guides=2, verbose=1):
    # Selection rounds of sgRNAs
    gguides = []
    for g in geneset:
        g_sgrnas = gselection.selection_rounds(g, n_guides=n_guides)[LIB_COLUMNS]

        if g_sgrnas.shape[0] == n_guides:
            gguides.append(g_sgrnas)

        else:
            LOG.warn(f"Not enough sgRNAs ({n_guides}) for {g}")

    gguides = pd.concat(gguides)[LIB_COLUMNS].reset_index(drop=True)

    if verbose > 0:
        gconf_str = ";".join(
            [f"{t}={n}" for t, n in gguides["Confidence"].value_counts().iteritems()]
        )
        LOG.info(f"sgRNAs quality: {gconf_str}")

    return gguides


def pair_sgrnas(sgrnas_pos1, sgrnas_pos2):
    sgrnas1 = sgrnas_pos1[LIB_COLUMNS].add_prefix("sgRNA1_").assign(dummy=1)
    sgrnas2 = sgrnas_pos2[LIB_COLUMNS].add_prefix("sgRNA2_").assign(dummy=1)
    sgrnas = sgrnas1.merge(sgrnas2, how="outer", on="dummy")[LIB_COLUMNS_EXP]
    return sgrnas


def project_score_data(sgrnas, subset=None):
    ddir = pkg_resources.resource_filename("crispy", "data/")

    score_manifest = pd.read_csv(
        f"{ddir}/crispr_manifests/project_score_manifest.csv.gz"
    )

    s_map = []
    for i in score_manifest.index:
        s_map.append(
            pd.DataFrame(
                dict(
                    model_id=score_manifest.iloc[i]["model_id"],
                    s_ids=score_manifest.iloc[i]["library"].split(", "),
                    s_lib=score_manifest.iloc[i]["experiment_identifier"].split(", "),
                )
            )
        )
    s_map = pd.concat(s_map).set_index("s_lib")

    if subset is not None:
        s_map = s_map[s_map["model_id"].isin(subset)]

    score_v1 = CRISPRDataSet("Yusa_v1")
    score_v1_fc = score_v1.counts.norm_rpm().foldchange(score_v1.plasmids)
    score_v1_fc = score_v1_fc.groupby(s_map["model_id"], axis=1).mean()

    score_v11 = CRISPRDataSet("Yusa_v1.1")
    score_v11_fc = score_v11.counts.norm_rpm().foldchange(score_v11.plasmids)
    score_v11_fc = score_v11_fc.groupby(s_map["model_id"], axis=1).mean()

    ess = set(
        score_v1.lib[score_v1.lib["Gene"].isin(Utils.get_essential_genes())].index
    )
    ness = set(
        score_v1.lib[score_v1.lib["Gene"].isin(Utils.get_non_essential_genes())].index
    )
    score_v1_fc = ReadCounts(score_v1_fc).scale(essential=ess, non_essential=ness)
    score_v11_fc = ReadCounts(score_v11_fc).scale(essential=ess, non_essential=ness)

    score_fc = pd.concat(
        [score_v1_fc.loc[sgrnas], score_v11_fc.loc[sgrnas]], axis=1
    ).dropna()

    return score_fc


def define_controls(
    n_genes=3,
    cancer_type="Colorectal Carcinoma",
    cn_min=1,
    cn_max=5,
    crisp_min=-0.10,
    jacks_thres=0.25,
    offtarget=[1, 0, 0],
):
    # Samples
    samples = set(
        DataImporter.Sample().samplesheet.query(f"cancer_type == '{cancer_type}'").index
    )

    # Non-essential genes
    ness = Utils.get_non_essential_genes(return_series=False)
    ness = ness - set(Utils.get_sanger_essential()["Gene"])

    # Non-essential genes sgRNAs
    ness_sgrnas = pd.concat(
        [
            gselection.select_sgrnas(
                g, 2, jacks_thres=jacks_thres, offtarget=offtarget
            ).assign(gene=g)
            for g in ness
        ],
        ignore_index=True,
    ).query("Library == 'KosukeYusa'")

    ness_sgrnas_fc = project_score_data(ness_sgrnas["sgRNA_ID"], samples)

    ness_sgrnas_fc_ds = ness_sgrnas_fc.T.describe().T.dropna()
    ness_sgrnas_fc_ds = ness_sgrnas_fc_ds[ness_sgrnas_fc_ds["25%"] >= crisp_min]
    ness_sgrnas_fc_ds["Approved_Symbol"] = (
        ness_sgrnas.set_index("sgRNA_ID")
        .loc[ness_sgrnas_fc_ds.index, "Approved_Symbol"]
        .values
    )

    ness_sgrnas = ness_sgrnas.set_index("sgRNA_ID").loc[ness_sgrnas_fc_ds.index]

    # Import different levels of information
    ddir = pkg_resources.resource_filename("crispy", "data/")

    cn = DataImporter.CopyNumber(
        f"{ddir}/copy_number/cnv_abs_copy_number_picnic_20191101.csv.gz"
    ).filter(subset=samples)

    hgnc = pd.read_csv(f"{DPATH}/protein-coding_gene.txt", sep="\t", index_col=1)

    # Control genes
    controls = ness_sgrnas.groupby("Approved_Symbol")["Library"].count()
    controls = list(controls[controls == 2].index)
    controls = pd.concat(
        [
            cn.reindex(controls).dropna().T.describe().T,
            hgnc.reindex(controls)["location"],
        ],
        axis=1,
        sort=False,
    ).dropna()
    controls = controls.query(f"(min >= {cn_min}) and (max <= {cn_max})")
    controls = controls.reset_index().rename(columns={"index": "Approved_Symbol"})
    controls = controls.merge(
        ness_sgrnas_fc_ds.reset_index(),
        on="Approved_Symbol",
        suffixes=("_cn", "_crispr"),
    )

    control_genes = list(
        controls.groupby("Approved_Symbol")["min_crispr"]
        .mean()
        .sort_values(ascending=False)[:n_genes]
        .index
    )
    controls = controls[controls["Approved_Symbol"].isin(control_genes)]
    controls["location"] = hgnc.loc[controls["Approved_Symbol"], "location"].values

    control_guides = gselection.masterlib[
        gselection.masterlib["sgRNA_ID"].isin(controls["sgRNA"])
    ].assign(Confidence="Control")[LIB_COLUMNS]

    return control_guides.sort_values("Approved_Symbol")


def generate_oligo(sgrna1, scaffold, linker, trna, sgrna2):
    sgrna1 = sgrna1[1:-3] if len(sgrna1) == 23 else sgrna1
    sgrna2 = sgrna2[1:-3] if len(sgrna2) == 23 else sgrna2

    oligo = f"{CLONING_ARMS['left']}G{sgrna1}{scaffold}{linker}{trna}G{sgrna2}{CLONING_ARMS['right']}"

    return oligo


if __name__ == "__main__":
    # Guide selection
    #
    gselection = GuideSelection()

    # Control guides
    #
    control_sgrnas = define_controls(n_genes=3)

    # Anchor guides
    #
    anchors = set(
        pd.read_excel(f"{DPATH}/{GILIB1NAME}", sheet_name="Anchor Genes")["Genes"]
    )
    genes_master_check(anchors)
    LOG.info(f"Anchor genes = {len(anchors)}")

    anchors_sgrnas = {n: generate_guides(anchors, n_guides=n) for n in [2, 4]}
    LOG.info(f"Anchor sgRNAs = {len(anchors_sgrnas)}")

    # Library guides
    #
    libraries_nguides = 2

    libraries = set(
        pd.read_excel(f"{DPATH}/{GILIB1NAME}", sheet_name="Library Genes")["Genes"]
    )
    libraries = libraries - anchors
    genes_master_check(libraries)
    LOG.info(f"Library genes = {len(libraries)}")

    libraries_sgrnas = generate_guides(libraries, n_guides=libraries_nguides)
    LOG.info(f"Library sgRNAs = {len(libraries_sgrnas)}")

    # GI library
    #
    pilot_lib = pd.read_excel(f"{DPATH}/{PILOTLIBNAME}", sheet_name="Library").query(
        "Scaffold == 'Wildtype'"
    )

    # Singletons
    anchors_sing = pair_sgrnas(control_sgrnas, anchors_sgrnas[4]).assign(
        Notes="AnchorSingletons"
    )
    libraries_sing = pair_sgrnas(libraries_sgrnas, control_sgrnas).assign(
        Notes="LibrarySingletons"
    )

    # Combinations
    anchors_comb = pair_sgrnas(anchors_sgrnas[4], anchors_sgrnas[4]).assign(
        Notes="AnchorCombinations"
    )
    libraries_comb = pair_sgrnas(libraries_sgrnas, anchors_sgrnas[2]).assign(
        Notes="LibraryCombinations"
    )

    # Negative controls
    negative_controls = pilot_lib.query("Notes == 'NegControls'")[
        LIB_COLUMNS_EXP
    ].assign(Notes="NegativeControls")

    # Positive controls
    positive_controls = pilot_lib.query("Notes == 'ScaffoldPosition'")[
        LIB_COLUMNS_EXP
    ].assign(Notes="PositiveControls")

    # GI controls
    gi_controls = pilot_lib.query("Notes == 'GIpairs'")[LIB_COLUMNS_EXP].assign(
        Notes="GIControls"
    )

    # Assemble
    gi_lib = pd.concat(
        [
            anchors_sing,
            libraries_sing,
            anchors_comb,
            libraries_comb,
            negative_controls,
            positive_controls,
            gi_controls,
        ],
        axis=0,
        sort=False,
        ignore_index=True,
    )

    # Add vector components
    gi_lib = gi_lib.assign(Scaffold=SCAFFOLD).assign(Linker=LINKER).assign(tRNA=TRNA)

    # Drop duplicates
    vids = ["sgRNA1_WGE_Sequence", "Scaffold", "Linker", "tRNA", "sgRNA2_WGE_Sequence"]
    gi_lib = gi_lib.drop_duplicates(subset=vids)

    # Add 2gCRISPR ids to each vector
    gi_lib["ID"] = [f"GI{i:09d}" for i in np.arange(1, len(gi_lib) + 1)]

    # Order columns
    gi_lib = gi_lib[["ID", "Notes"] + LIB_COLUMNS_EXP + ["Scaffold", "Linker", "tRNA"]]

    # Oligo sequences
    gi_lib["Oligo_Sequence"] = [generate_oligo(sg1, s, l, t, sg2) for sg1, s, l, t, sg2 in gi_lib[vids].values]

    # Export
    gi_lib.to_excel(f"{DPATH}/{GILIB1NAME[:-5]}_guides_only.xlsx", index=False)
