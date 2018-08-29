#!/usr/bin/env python
# Copyright (C) 2018 Emanuel Goncalves

import os
import crispy as cy
import pandas as pd


DIR = 'data/gdsc/'

SAMPLE_INFO = 'sample_info.csv'
SAMPLESHEET = 'samplesheet.csv'
RAW_COUNTS = 'raw_read_counts.csv'
COPYNUMBER = 'copynumber_PICNIC_segtab.csv'

QC_FAILED_SGRNAS = [
    'DHRSX_CCDS35195.1_ex1_X:2161152-2161175:+_3-1',
    'DHRSX_CCDS35195.1_ex6_Y:2368915-2368938:+_3-3',
    'DHRSX_CCDS35195.1_ex4_X:2326777-2326800:+_3-2',
    'sgPOLR2K_1'
]

# WGS samples
BRCA_SAMPLES = ['HCC1954', 'HCC1143', 'HCC38', 'HCC1187', 'HCC1937', 'HCC1395']

# NON-EXPRESSED GENES: RNA-SEQ
NON_EXP = 'non_expressed_genes.csv'
BIOMART_HUMAN_ID_TABLE = 'rnaseq/biomart_human_id_table.csv'

GDSC_RNASEQ_SAMPLESHEET = 'rnaseq/merged_sample_annotation.csv'
GDSC_RNASEQ_RPKM = 'rnaseq/merged_rpkm.csv'


def samplesheet_suppmaterial():
    beds = import_crispy_beds()
    ploidy = pd.Series({s: beds[s]['ploidy'].mean() for s in beds})

    sinfo = pd.read_csv(f'{DIR}/{SAMPLE_INFO}', index_col=0)
    samples = pd.read_csv(f'{DIR}/{SAMPLESHEET}')

    cols = ['Master ID', 'COSMIC ID', 'Tissue', 'Cancer Type', 'CCLE ID']

    ss = sinfo[sinfo.index.isin(samples['name'])]
    ss = pd.concat([ss[cols], ploidy.rename('ploidy')], axis=1, sort=False)
    ss['COSMIC ID'] = ss['COSMIC ID'].astype(int)

    return ss


def get_gene_ratios():
    beds = import_crispy_beds()

    ratios = pd.DataFrame({s: beds[s].groupby('gene')['ratio'].median() for s in beds})

    return ratios


def import_crispy_beds(wgs=False):
    beds = {
        f.split('.')[0]:
            pd.read_csv(f'{DIR}/bed/{f}', sep='\t') for f in os.listdir(f'{DIR}/bed/') if (f.endswith('crispy.wgs.bed') if wgs else f.endswith('crispy.bed'))
    }
    return beds


def import_ccleanr():
    ccleanr = pd.read_csv(f'{DIR}/CRISPRcleaned_logFCs.tsv', sep='\t', index_col=0)
    return ccleanr


def import_brass_bedpes():
    return {s: cy.Utils.import_brass_bedpe(f'{DIR}/wgs/{s}.brass.annot.bedpe') for s in BRCA_SAMPLES}


def get_samplesheet():
    return pd.read_csv(f'{DIR}/{SAMPLESHEET}', index_col=0)


def get_raw_counts():
    return pd.read_csv(f'{DIR}/{RAW_COUNTS}', index_col=0)


def get_copy_number_segments():
    return pd.read_csv(f'{DIR}/{COPYNUMBER}')


def get_copy_number_segments_wgs(as_dict=False):
    if as_dict:
        wgs_copynumber = {
            s:
                pd.read_csv(f'{DIR}/wgs/{s}.wgs.ascat.bed', sep='\t', names=cy.Utils.ASCAT_HEADERS, comment='#')
            for s in BRCA_SAMPLES
        }

    else:
        wgs_copynumber = pd.concat([
            pd.read_csv(f'{DIR}/wgs/{s}.wgs.ascat.bed', sep='\t', names=cy.Utils.ASCAT_HEADERS, comment='#').assign(sample=s)
            for s in BRCA_SAMPLES
        ])

    return wgs_copynumber


def get_ensembl_gene_id_table():
    ensg = pd.read_csv(f'{DIR}/{BIOMART_HUMAN_ID_TABLE}')
    ensg = ensg.groupby('Gene name')['Gene stable ID'].agg(lambda x: set(x)).to_dict()
    return ensg


def get_rnaseq_samplesheet():
    return pd.read_csv(f'{DIR}/{GDSC_RNASEQ_SAMPLESHEET}')


def get_rnaseq_rpkm():
    return pd.read_csv(f'{DIR}/{GDSC_RNASEQ_RPKM}', index_col=0)


def get_non_exp():
    if not os.path.exists(f'{DIR}/{NON_EXP}'):
        write_non_expressed_matrix()

    return pd.read_csv(f'{DIR}/{NON_EXP}', index_col=0)


def get_sample_info(index_col=0):
    return pd.read_csv(f'{DIR}/{SAMPLE_INFO}', index_col=index_col)


def write_non_expressed_matrix(rpkm_thres=1):
    # Gene map
    ensg = get_ensembl_gene_id_table()

    # Samplesheet
    ss = get_rnaseq_samplesheet()
    ss = pd.Series({k: v.split('.')[0] for k, v in ss.groupby('expression_matrix_name')['sample_name'].first().to_dict().items()})

    # RNA-seq RPKMs
    rpkm = get_rnaseq_rpkm()

    # Low-expressed genes: take the highest abundance across the different transcripts
    nexp = pd.DataFrame(
        {g: (rpkm.reindex(ensg[g]) < rpkm_thres).astype(int).min() for g in ensg if len(ensg[g].intersection(rpkm.index)) > 0}
    ).T

    # Merged cell lines replicates: take the highest abundance across the different replicates
    nexp = nexp.groupby(ss.loc[nexp.columns], axis=1).min()

    # Remove genes defined as essential in the CRISPR screen
    nexp = nexp[~nexp.index.isin(cy.Utils.get_adam_core_essential())]

    # Export
    nexp.to_csv(f'{DIR}/{NON_EXP}')
