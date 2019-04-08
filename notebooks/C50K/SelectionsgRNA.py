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
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from crispy.CrispyPlot import CrispyPlot
from crispy.CRISPRData import CRISPRDataSet


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


BUILD = "Grch38"


gene = "FH"

pd.read_json(f"https://www.sanger.ac.uk/htgt/wge/api/exon_search?marker_symbol={gene}&species={BUILD}")


exons = ["ENSE00001442032", "ENSE00003687860"]

pd.read_json(f"http://www.sanger.ac.uk/htgt/wge/api/crispr_search?&species={BUILD}&{'&'.join(map(lambda v: f'exon_id[]={v}', exons))}", typ="series")