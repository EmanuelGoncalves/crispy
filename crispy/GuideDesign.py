#!/usr/bin/env python
# Copyright (C) 2019 Emanuel Goncalves

import logging
import numpy as np
import pandas as pd
import pkg_resources
import seaborn as sns
import matplotlib.pyplot as plt

# # sgRNAs library selection
#
#   Defining rules:
#       1) Protein-coding transcripts
#           + [Avana] CCDS: shortest transcript selected
#           + [Brunello] GENCODE: (i) APPRIS annotation; (ii) transcript confidence; (iii) support in GENCODE
#               + gene_type = "protein_coding"
#               + level = 1
#               + transcript_support_level = "1"
#               + tag appris_principal_ = "appris_principal_1"
#               (run pyensembl install --release 94 --species homo_sapiens)
#
#       2) Genes with multiple CCDS IDs, the shortest transcript was chosen
#
#       3) NGG protospacer adjacent motifs (PAMs) on both strands are annotated
#
#       4) sgRNA exclusion rules:
#           - Restriction sites
#               + [Avana + TKOv3] No BsmBI site: CGTCTC
#               + [TKOv3] No AgeI site: ACCGGT
#               + [TKOv3] No KpnI site: GGTACC
#               + [TKOv3] No BveI/BspMI site: ACCTGC
#               + [TKOv3] No BsmI site: GAATGC
#           - GC content
#               + [TKOv3] <= 40% or >= 75%
#           - Homopolymers
#               + [TKOv3] Any homopolymers length >= 4
#               + [Avana] T homopolymers length >= 4
#           - Germline SNP
#               + [TKOv3] No common SNP (db138) in PAM or sgRNA
#
#       5) Location in the protein coding sequence:
#           + [Avana] 0-25%: 57% of the guides; 25-50%: 43% of the guides
#
#       6) Off-target effect:
#           + [Avana] PAM-proximal (0) unique nts: -13 (84%); -17 (13%); -20 (0.2%); not unique (3%)
#           + [Yusa] sgRNAs with more than 1 perfect exonic match were excluded;
#           + [Yusa] N12NGG
#
#       7) On-target effect:
#           + [Avana] Rule-set 1:
#
#       8) Multi-guides:
#           + [Brunello] sgRNAs need to target a location at least 5% away from other guides from a protein-coding
#           standpoint.


class GuideDesign:
    BUILD = "Grch38"
    EXON_SEARCH = "https://www.sanger.ac.uk/htgt/wge/api/exon_search"
    CRISPR_SEARCH = "http://www.sanger.ac.uk/htgt/wge/api/crispr_search"

    RESTRICTION_SITES = dict(
        BsmBI="CGTCTC",
        AgeI="ACCGGT",
        KpnI="GGTACC",
        BveI="ACCTGC",
        BspMI="ACCTGC",
        BsmI="GAATGC",
    )

    def __init__(self):
        self.log = logging.getLogger("Crispy")

        self.dpath = pkg_resources.resource_filename("crispy", "data/")

        # GENCODE: evidence based annotation of the human genome (GRCh38), version 29 (Ensembl 94)
        self.gencode = pd.read_csv(
            f"{self.dpath}/gencode.v29.annotation.gtf.gz",
            skiprows=5,
            sep="\t",
            header=None,
        )
        self.log.info(f"GENCODE annotation loaded: {self.gencode.shape}")

    @classmethod
    def get_wes_transcripts(cls, gene_symbol):
        transcripts = pd.read_json(
            f"{cls.EXON_SEARCH}?marker_symbol={gene_symbol}&species={cls.BUILD}"
        )
        return transcripts

    @classmethod
    def get_wes_sgrnas(cls, exons):
        exons_query = "&".join(map(lambda v: f"exon_id[]={v}", exons))
        sgrnas = pd.read_json(
            f"{cls.CRISPR_SEARCH}?&species={cls.BUILD}&{exons_query}", typ="series"
        )
        return sgrnas

    @staticmethod
    def reverse_complement(sequence, reverse=True):
        code = dict(A="T", T="A", C="G", G="C")

        complement_seq = "".join([code[b] for b in sequence])

        if reverse:
            complement_seq = complement_seq[::-1]

        return complement_seq
