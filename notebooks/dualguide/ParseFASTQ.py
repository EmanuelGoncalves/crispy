import gzip
import pandas as pd
import pkg_resources
from Bio import SeqIO
from crispy import logger as LOG

DPATH = pkg_resources.resource_filename("data", "dualguide/")
RPATH = pkg_resources.resource_filename("notebooks", "dualguide/reports/")
FASTQ_DIR = (
    "MiSeq_walk_up_227-142397264/FASTQ_Generation_2019-10-11_10_04_51Z-194053941/"
)

BACKBONE = "GTTTAAG"
TRNA = "GCATTGG"

STYPES = [
    "sgRNA1",
    "sgRNA2",
    "scaffold",
    "linker",
    "trna",
    "backbone",
    "scaffold_wt_pos1",
    "scaffold_wt_pos2",
    "scaffold_k_pos1",
    "scaffold_k_pos2",
    "scaffold_mt_pos1",
    "scaffold_mt_pos2",
]

STYPES_POS_R1 = dict(
    sgRNA1=(0, 20),
    scaffold=(20, 96),
    linker=(96, 102),
    trna=(102, 173),
    sgRNA2=(173, 193),
)
STYPES_POS_R2 = dict(sgRNA2=(0, 20), trna=(20, 91))
SELEMS = [
    ("sgRNA1", "R1"),
    ("scaffold", "R1"),
    ("linker", "R1"),
    ("sgRNA2", "R2"),
    ("backbone", "R1"),
    ("trna", "R2"),
    ("scaffold_wt_pos1", "R1"),
    ("scaffold_wt_pos2", "R1"),
    ("scaffold_k_pos1", "R1"),
    ("scaffold_k_pos2", "R1"),
    ("scaffold_mt_pos1", "R1"),
    ("scaffold_mt_pos2", "R1"),
]

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

    elif stype == "trna":
        return seq_position(values, TRNA)

    elif stype in SCAFFOLDS_MOTIFS:
        return seq_position(values, SCAFFOLDS_MOTIFS[stype])

    else:
        pos = STYPES_POS_R1[stype] if rtype == "R1" else STYPES_POS_R2[stype]
        seq = (
            values[pos[0] : pos[1]]
            if rtype == "R1"
            else values[pos[0] : pos[1]].reverse_complement()
        )
        return seq


def parse_fastq_file(filename):
    with gzip.open(filename, "rt") as f:
        reads = {r.id: r for r in SeqIO.parse(f, "fastq")}
    return reads


def annotate_read(r):
    r_dict = {}

    # sgRNA ID
    r_dict["id"] = lib_dict["lib_ids"].get(tuple(r[n].seq._data for n in id_info))

    # sgRNAs at all in library
    for e in ["sgRNA1", "sgRNA2"]:
        r_dict[e] = r[e].seq._data
        r_dict[f"{e}_exists_inlib"] = int(r_dict[e] in lib_dict["lib_sgrnas"])
        r_dict[f"status{e[-1]}"] = lib_dict["lib_status"].get(r_dict[e])

    # Scaffold
    r_dict["scaffold"] = lib_dict["lib_scaffold"].get(r["scaffold"].seq._data)

    # Linker
    r_dict["linker"] = lib_dict["lib_linker"].get(r["linker"].seq._data)

    # Pair exists
    r_dict["sgRNA_pair_exists"] = int(
        (r_dict["sgRNA1"], r_dict["sgRNA2"]) in lib_dict["lib_pairs"]
    )

    # Constructs exist
    for e in id_info:
        r_dict[f"{e}_exists"] = int(r[e].seq._data in lib_dict["lib_sets"][e])

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
    r_dict["trna_pos"] = r["trna"]

    # Scaffold positions
    for stype in SCAFFOLDS_MOTIFS:
        r_dict[stype] = r[stype]

    return r_dict


def seq_position(read, motif):
    try:
        pos = read.seq._data.index(motif)

    except ValueError:
        pos = -1

    return pos


if __name__ == "__main__":
    # Library & Samplesheet
    #
    id_info = ["sgRNA1", "scaffold", "linker", "sgRNA2"]

    lib = pd.read_excel(f"{DPATH}/DualGuidePilot_v1.1_annotated.xlsx")

    lib_dict = dict(
        lib_ids=lib.groupby(id_info)["sgRNA_ID"].agg(lambda v: ";".join(v)).to_dict(),
        lib_sets={c: set(lib[c]) for c in id_info},
        lib_sgrnas=set(lib["sgRNA1"]).union(lib["sgRNA2"]),
        lib_linker=lib.groupby("linker")["linker_type"].first().to_dict(),
        lib_pairs={(sg1, sg2) for sg1, sg2 in lib[["sgRNA1", "sgRNA2"]].values},
        lib_scaffold=lib.groupby("scaffold")["scaffold_type"].first().to_dict(),
        lib_status={
            **lib.groupby(f"sgRNA1")[f"status1"].first().to_dict(),
            **lib.groupby(f"sgRNA2")[f"status2"].first().to_dict(),
        },
    )

    lib_ss = pd.read_excel(f"{DPATH}/samplesheet.xlsx")

    # Parse reads
    #
    for n, lib_ss_df in lib_ss.groupby("name"):
        LOG.info(f"{n}")

        # Read fastq (R1 and R2)
        reads = {}
        for idx in lib_ss_df.index:
            fastq_file = f"{DPATH}/{FASTQ_DIR}/{lib_ss.loc[idx, 'directory']}/{lib_ss.loc[idx, 'file']}"
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
        reads = pd.DataFrame(
            {r_id: annotate_read(reads[r_id]) for r_id in reads}
        ).T
        LOG.info(f"Reads parsed: {reads.shape}")

        # Export
        reads.to_csv(
            f"{RPATH}/fastq_parsed_run{lib_ss_df.iloc[0]['run']}_{n}.csv.gz",
            compression="gzip",
        )
