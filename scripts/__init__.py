#!/usr/bin/env python
# Copyright (C) 2018 Emanuel Goncalves

# META DATA PATHS
MANIFEST = 'data/gdsc/crispr/manifest.csv'
SAMPLESHEET = 'data/gdsc/samplesheet.csv'

# CRISPR LIBRARY
LIBRARY = 'data/crispr_libs/KY_Library_v1.1_updated.csv'

QC_FAILED_SGRNAS = [
    'DHRSX_CCDS35195.1_ex1_X:2161152-2161175:+_3-1',
    'DHRSX_CCDS35195.1_ex6_Y:2368915-2368938:+_3-3',
    'DHRSX_CCDS35195.1_ex4_X:2326777-2326800:+_3-2',
    'sgPOLR2K_1'
]

LOW_COUNT_THRES = 30

# CRISPR PATHS
CRISPR_RAW_COUNTS = 'data/gdsc/crispr/gdsc_crispr_rawcounts.csv'
CRISPR_SGRNA_FC = 'data/crispr_gdsc_sgrna_logfc.csv'
CRISPR_GENE_FC = 'data/crispr_gdsc_logfc.csv'
CRISPR_GENE_CORRECTED_FC = 'data/crispr_gdsc_logfc_corrected.csv'

# WGS ASCAT PATHS
WGS_ASCAT_BED = 'data/gdsc/wgs/ascat_bed/'

# WGS BRASS PATHS
WGS_BRASS_BEDPE = 'data/gdsc/wgs/brass_bedpe/'

# WGS + CRISPR BRCA samples
BRCA_SAMPLES = ['HCC1954', 'HCC1143', 'HCC38', 'HCC1187', 'HCC1937', 'HCC1395']

# SNP6 PICNIC PATHS
SNP_PICNIC_BED = 'data/gdsc/copynumber/snp6_bed/'

# CN PATHS
CN_GENE = 'data/crispy_copy_number_gene_{}.csv'
CN_GENE_RATIO = 'data/crispy_copy_number_gene_ratio_{}.csv'
CN_CHR = 'data/crispy_copy_number_chr_{}.csv'
CN_PLOIDY = 'data/crispy_copy_number_ploidy_{}.csv'

# CRISPY
CRISPY_OUTDIR = 'data/crispy/gdsc/'
CRISPY_WGS_OUTDIR = 'data/crispy/gdsc_brass/'

# NON-EXPRESSED GENES: RNA-SEQ
NON_EXP_PICKLE = 'data/gdsc/nexp_pickle.pickle'

# GENE-SETS
CANCER_GENES = 'data/gene_sets/Census_allThu Dec 21 15_43_09 2017.tsv'
HART_ESSENTIAL = 'data/gene_sets/curated_BAGEL_essential.csv'
HART_NON_ESSENTIAL = 'data/gene_sets/curated_BAGEL_nonEssential.csv'

# GTex
GTEX_TPM = 'data/GTEx_Analysis_2016-01-15_v7_RNASeQCv1.1.8_gene_median_tpm.gct'
