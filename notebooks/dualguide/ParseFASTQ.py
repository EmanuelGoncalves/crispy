import gzip
import pandas as pd
import pkg_resources
from Bio import SeqIO
from crispy import logger as LOG
from dualguide import read_gi_library

DPATH = pkg_resources.resource_filename("data", "dualguide/")
RPATH = pkg_resources.resource_filename("notebooks", "dualguide/reports/")

STYPES = ["sgRNA1", "sgRNA2", "Scaffold_Sequence", "tRNA", "backbone"]

STYPES_POS_R1 = dict(
    sgRNA1=(0, 19), Scaffold_Sequence=(19, 95), tRNA=(101, 172)
)
STYPES_POS_R2 = dict(sgRNA2=(0, 20), tRNA=(20, 91))

SELEMS = [
    ("sgRNA1", "R1"),
    ("sgRNA2", "R2"),
    ("Scaffold_Sequence", "R1"),
    ("tRNA", "R2"),
    ("backbone", "R1"),
]

TRNA = "GCATTGG"
BACKBONE = "GTTTAAG"
SCAFFOLDS_MOTIFS = dict(
    scaffold_wt_pos1="TTTTAGA",
    scaffold_wt_pos2="GTTAAAA",
    scaffold_k_pos1="TTTAAGA",
    scaffold_k_pos2="GTTTAAA",
    scaffold_mt_pos1="TCTCAGA",
    scaffold_mt_pos2="GTTGAGA",
)


def parse_seq(values, stype, rtype):
    assert stype in STYPES, f"sequence type '{stype}' not supported"

    if stype == "backbone":
        return seq_position(values, BACKBONE)

    elif stype == "tRNA":
        return seq_position(values, TRNA)

    elif stype in SCAFFOLDS_MOTIFS:
        return seq_position(values, SCAFFOLDS_MOTIFS[stype])

    else:
        pos = STYPES_POS_R1[stype] if rtype == "R1" else STYPES_POS_R2[stype]

        if rtype == "R1":
            seq = values[pos[0]:pos[1]]

        else:
            seq = values[pos[0]:pos[1]].reverse_complement()

        if stype == "sgRNA1":
            seq = seq.__radd__("G")

        return seq


def annotate_read(r, library_dict):
    r_dict = dict()

    # sgRNA ID
    r_dict["id"] = library_dict["lib_ids"].get(tuple(r[n].seq._data for n in id_info))

    # sgRNAs at all in library
    for e in ["sgRNA1", "sgRNA2"]:
        r_dict[e] = r[e].seq._data
        r_dict[f"{e}_exists_inlib"] = int(r_dict[e] in library_dict["lib_sgrnas"])

    # Scaffold
    r_dict["scaffold"] = library_dict["lib_scaffold"].get(r["Scaffold_Sequence"].seq._data)

    # Pair exists
    r_dict["sgRNA_pair_exists"] = int(
        (r_dict["sgRNA1"], r_dict["sgRNA2"]) in library_dict["lib_pairs"]
    )

    # Constructs exist
    for e in id_info:
        r_dict[f"{e}_exists"] = int(r[e].seq._data in library_dict["lib_sets"][e])

    # Bool combination of sgRMAs in lib
    r_dict["sgRNAs_both_inlib"] = (
        r_dict["sgRNA1_exists_inlib"] and r_dict["sgRNA2_exists_inlib"]
    )
    r_dict["sgRNAs_any_inlib"] = (
        r_dict["sgRNA1_exists_inlib"] or r_dict["sgRNA2_exists_inlib"]
    )

    # Swapping
    r_dict["swap"] = int(
        (1 - r_dict["sgRNA_pair_exists"]) and r_dict["sgRNAs_both_inlib"]
    )

    # Backbone starting position
    r_dict["backbone_pos"] = r["backbone"]

    # tRNA starting position
    r_dict["trna_pos"] = r["tRNA"]

    return r_dict


def seq_position(read, motif):
    try:
        pos = read.seq._data.index(motif)

    except ValueError:
        pos = -1

    return pos


def build_lib_dict(library):
    library_scaffold = pd.read_excel(
        f"{DPATH}/gi_libraries/2gCRISPR_Pilot_library_v2.0.0.xlsx",
        sheet_name="Scaffolds",
        index_col=1,
    )["Scaffold"].to_dict()

    library_dict = dict(
        lib_ids=library.reset_index().groupby(id_info)["ID"].first().to_dict(),
        lib_sets={c: set(library[c]) for c in id_info},
        lib_sgrnas=set(library["sgRNA1"]).union(library["sgRNA2"]),
        lib_pairs={(sg1, sg2) for sg1, sg2 in library[["sgRNA1", "sgRNA2"]].values},
        lib_scaffold=library_scaffold,
    )
    return library_dict


def parse_fastq_file(filename):
    with gzip.open(filename, "rt") as f:
        reads = {r.id: r for r in SeqIO.parse(f, "fastq")}
    return reads


if __name__ == "__main__":
    # Library & Samplesheet
    #
    lib_ss = pd.read_excel(f"{DPATH}/gi_samplesheet.xlsx")
    lib_ss = lib_ss.query("library == '2gCRISPR_Pilot_library_v2.0.0.xlsx'")

    # Parse reads
    #
    # n, lib_ss_df = list(lib_ss.groupby("name"))[1]
    for n, lib_ss_df in lib_ss.groupby("name"):
        LOG.info(f"{n}")

        # Fields to make up the sgRNA IDs
        id_info = ["sgRNA1", "sgRNA2"]

        if "_PILOT_" in n:
            id_info += ["Scaffold_Sequence"]

        lib = read_gi_library(lib_ss_df["library"].iloc[0])
        lib_dict = build_lib_dict(lib)

        # Read fastq (R1 and R2)
        reads = {}
        for idx in lib_ss_df.index:
            fastq_file = f"{DPATH}/gi_fastqs/{lib_ss.loc[idx, 'directory']}/{lib_ss.loc[idx, 'file']}"
            reads[lib_ss.loc[idx, "read"]] = parse_fastq_file(fastq_file)
            LOG.info(
                f"{n} ({lib_ss.loc[idx, 'read']}) reads parsed: {len(reads[lib_ss.loc[idx, 'read']])}"
            )

        # Retrieve sequence elements
        reads = {
            r_id: {s: parse_seq(reads[r][r_id], s, r) for s, r in SELEMS}
            for r_id in reads["R1"]
        }
        LOG.info(f"Number of reads: {len(reads)}")

        # Annotate reads
        reads = pd.DataFrame({r_id: annotate_read(reads[r_id], lib_dict) for r_id in reads}).T
        LOG.info(f"Reads parsed: {reads.shape}")

        # Export
        reads.to_csv(f"{DPATH}/gi_counts/{n}_parsed_reads.csv.gz", compression="gzip")
