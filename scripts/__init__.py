#!/usr/bin/env python
# Copyright (C) 2018 Emanuel Goncalves

import os
import numpy as np
import pandas as pd
from crispy.utils import bin_bkdist
from crispy.ratio import BRASS_HEADERS
from crispy import get_essential_genes, get_non_essential_genes, get_crispr_lib


# META DATA PATHS
MANIFEST = 'data/gdsc/crispr/manifest.csv'
SAMPLESHEET = 'data/gdsc/samplesheet.csv'

# CRISPR LIBRARY
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
NON_EXP = 'data/non_expressed_genes.csv'
BIOMART_HUMAN_ID_TABLE = 'data/resources/biomart/biomart_human_id_table.csv'

GDSC_RNASEQ_SAMPLESHEET = 'data/gdsc/gene_expression/merged_sample_annotation.csv'
GDSC_RNASEQ_RPKM = 'data/gdsc/gene_expression/merged_rpkm.csv'

# SHRNA
SHRNA = 'data/ccle/DRIVE_RSA_data.txt'

# SAMPLES OVERLAP FILE
SAMPLES_OVERLAP_FILE = 'data/samples_overlap.csv'


# - GETTERS AND SETTERS
def get_sample_list():
    return list(pd.read_csv(SAMPLES_OVERLAP_FILE, header=None)[0])


def get_shrna():
    ss = get_samplesheet(index_col=None).dropna(subset=['CCLE ID']).set_index('CCLE ID')

    shrna = pd.read_csv(SHRNA, sep='\t', index_col=0)
    shrna.columns = [c.upper() for c in shrna]

    shrna = shrna.rename(columns=ss['Cell Line Name'].to_dict())

    return shrna


def get_rnaseq_rpkm():
    rpkm = pd.read_csv(GDSC_RNASEQ_RPKM, index_col=0)
    return rpkm


def get_crispr(dfile, scale=True):
    crispr = pd.read_csv(dfile, index_col=0).dropna()

    if scale:
        crispr = scale_crispr(crispr)

    return crispr


def get_crispr_sgrna(scale=True, metric=np.median, dropna=True):
    lib = get_crispr_lib()

    df = pd.read_csv(CRISPR_SGRNA_FC, index_col=0)

    if dropna:
        df = df.dropna()

    if scale:
        ess = get_essential_genes()
        ess_metric = metric(df.reindex(lib[lib['GENES'].isin(ess)].index).dropna(), axis=0)

        ness = get_non_essential_genes()
        ness_metric = metric(df.reindex(lib[lib['GENES'].isin(ness)].index).dropna(), axis=0)

        df = df.subtract(ness_metric).divide(ness_metric - ess_metric)

    return df


def get_non_exp():
    if not os.path.exists(NON_EXP):
        write_non_expressed_matrix()

    return pd.read_csv(NON_EXP, index_col=0)


def get_copynumber(dtype='snp'):
    return pd.read_csv(CN_GENE.format(dtype), index_col=0)


def get_copynumber_ratios(dtype='snp'):
    return pd.read_csv(CN_GENE_RATIO.format(dtype), index_col=0)


def get_ploidy(dtype='snp'):
    return pd.read_csv(CN_PLOIDY.format(dtype), index_col=0, names=['sample', 'ploidy'])['ploidy']


def get_chr_copies(dtype='snp'):
    return pd.read_csv(CN_CHR.format(dtype), index_col=0)


def get_samplesheet(index_col=0):
    return pd.read_csv(SAMPLESHEET, index_col=index_col)


def get_rnaseq_samplesheet():
    ss = pd.read_csv(GDSC_RNASEQ_SAMPLESHEET)
    return ss


def get_ensembl_gene_id_table():
    ensg = pd.read_csv(BIOMART_HUMAN_ID_TABLE)
    ensg = ensg.groupby('Gene name')['Gene stable ID'].agg(lambda x: set(x)).to_dict()
    return ensg


# -- METHODS
def scale_crispr(df, ess=None, ness=None, metric=np.median):
    """
    Min/Max scaling of CRISPR-Cas9 log-FC by median (default) Essential and Non-Essential.

    :param df: Float pandas.DataFrame
    :param ess: set(String)
    :param ness: set(String)
    :param metric: np.Median (default)
    :return: Float pandas.DataFrame
    """

    if ess is None:
        ess = get_essential_genes()

    if ness is None:
        ness = get_non_essential_genes()

    assert len(ess.intersection(df.index)) != 0, 'DataFrame has no index overlapping with essential list'
    assert len(ness.intersection(df.index)) != 0, 'DataFrame has no index overlapping with non-essential list'

    ess_metric = metric(df.reindex(ess).dropna(), axis=0)
    ness_metric = metric(df.reindex(ness).dropna(), axis=0)

    df = df.subtract(ness_metric).divide(ness_metric - ess_metric)

    return df


def write_non_expressed_matrix(rpkm_thres=1):
    # Gene map
    ensg = get_ensembl_gene_id_table()

    # Samplesheet
    ss = get_rnaseq_samplesheet()
    ss = pd.Series({k: v.split('.')[0] for k, v in ss.groupby('expression_matrix_name')['sample_name'].first().to_dict().items()})

    # RNA-seq RPKMs
    rpkm = get_rnaseq_rpkm()

    # - Low-expressed genes: take the highest abundance across the different transcripts
    nexp = pd.DataFrame(
        {g: (rpkm.reindex(ensg[g]) < rpkm_thres).astype(int).min() for g in ensg if len(ensg[g].intersection(rpkm.index)) > 0}
    ).T

    # Merged cell lines replicates: take the highest abundance across the different replicates
    nexp = nexp.groupby(ss.loc[nexp.columns], axis=1).min()

    # Export
    nexp.to_csv(NON_EXP)


def import_brass_bedpe(bedpe_file, bkdist=None, splitreads=False, convert_to_bed=False):
    # Import BRASS bedpe
    bedpe_df = pd.read_csv(bedpe_file, sep='\t', names=BRASS_HEADERS, comment='#')

    # Correct sample name
    bedpe_df['sample'] = bedpe_df['sample'].apply(lambda v: v.split(',')[0])

    # SV larger than threshold
    if bkdist is not None:
        bedpe_df = bedpe_df[bedpe_df['bkdist'] >= bkdist]

    # BRASS2 annotated SV
    if splitreads:
        bedpe_df = bedpe_df.query("assembly_score != '_'")

    # Parse chromosome name
    bedpe_df = bedpe_df.assign(chr1=bedpe_df['chr1'].apply(lambda x: 'chr{}'.format(x)).values)
    bedpe_df = bedpe_df.assign(chr2=bedpe_df['chr2'].apply(lambda x: 'chr{}'.format(x)).values)

    # Conver SVs to bed
    if convert_to_bed:
        bedpe_df = brass_bedpe_to_bed(bedpe_df)
        bedpe_df = bedpe_df[bedpe_df['start'] < bedpe_df['end']]

    return bedpe_df


def brass_bedpe_to_bed(bedpe_df):
    # Unfold translocations
    bed_translocations = []
    for idx in [1, 2]:
        df = bedpe_df.query("svclass == 'translocation'")

        bed_translocations.append(pd.concat([
            df['chr{}'.format(idx)].rename('#chr'),
            df['start{}'.format(idx)].astype(int).rename('start'),
            df['end{}'.format(idx)].astype(int).rename('end'),
            df['svclass'].rename('svclass')
        ], axis=1))

    bed_translocations = pd.concat(bed_translocations)

    # Merge other SVs
    df = bedpe_df.query("svclass != 'translocation'")

    bed_others = pd.concat([
        df['chr1'].rename('#chr'),
        df[['start1', 'end1']].mean(1).astype(int).rename('start'),
        df[['start2', 'end2']].mean(1).astype(int).rename('end'),
        df['svclass'].rename('svclass'),
        df['bkdist'].apply(bin_bkdist).rename('bkdist')
    ], axis=1)

    bed_df = bed_translocations.append(bed_others).replace({'bkdist': {np.nan: '_'}})

    return bed_df[['#chr', 'start', 'end', 'svclass', 'bkdist']]


def samples_overlap(export=True):
    # Import CRISPR-Cas9 sgRNAs
    crispr = get_crispr_sgrna()

    # Non-expressed genes
    nexp = get_non_exp()

    # Copy-number
    cnv = get_copynumber()

    # Overlap samples
    samples = list(set(cnv).intersection(crispr).intersection(nexp))
    print('[INFO] Samples overlap: {}'.format(len(samples)))

    # Export
    if export:
        samples = pd.Series(samples).rename('overlap')
        samples.to_csv(SAMPLES_OVERLAP_FILE, index=False)

    return samples
