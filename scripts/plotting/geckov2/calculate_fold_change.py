#!/usr/bin/env python
# Copyright (C) 2018 Emanuel Goncalves

import numpy as np
import pandas as pd
import crispy as cy
import scripts as mp
import seaborn as sns
from pybedtools import BedTool
import matplotlib.pyplot as plt
from crispy.data_matrix import DataMatrix


GECKOV2_PLASMID = 'pDNA_pXPR003_120K_20140624'

GECKOV2_RAW_COUNTS = 'data/geckov2/Achilles_v3.3.8_rawreadcounts.gct'
GECKOV2_SAMPLESHEET = 'data/geckov2/Achilles_v3.3.8.samplesheet.txt'
GECKOV2_SGRNA_MAP = 'data/geckov2/Achilles_v3.3.8_sgRNA_mappings.txt'
GECKOV2_CN = 'data/geckov2/Achilles_v3.3.8_ABSOLUTE_CN_segtab.txt'

BED_NAMES = ['#chr', 'start', 'end', 'cn', 'chr', 'sgrna_start', 'sgrna_end', 'fc', 'sgrna_id']
BED_NAMES_ORDER = ['chr', 'start', 'end', 'sgrna_id', 'sgrna_start', 'sgrna_end', 'cn', 'fc', 'fc_scaled', 'chrm', 'ploidy', 'ratio', 'len']


def replicates_corr(ss):
    fc_corr = fc.corr()

    fc_corr = fc_corr.where(np.triu(np.ones(fc_corr.shape), 1).astype(np.bool))

    fc_corr = fc_corr.unstack().dropna().reset_index()

    fc_corr = fc_corr.rename(columns={'level_0': 'sample_1', 'level_1': 'sample_2', 0: 'corr'})

    fc_corr['name_1'] = [ss.loc[i.split(' Rep')[0], 'name'] for i in fc_corr['sample_1']]
    fc_corr['name_2'] = [ss.loc[i.split(' Rep')[0], 'name'] for i in fc_corr['sample_2']]

    fc_corr = fc_corr.assign(replicate=(fc_corr['name_1'] == fc_corr['name_2']).astype(int))

    return fc_corr


def import_lib():
    lib = pd.read_csv(GECKOV2_SGRNA_MAP, sep='\t').dropna(subset=['Chromosome', 'Pos', 'Strand'])
    lib['sgrna'] = lib['Guide'].apply(lambda v: v.split('_')[0])

    lib = lib.rename(columns={'Chromosome': '#chr', 'Pos': 'start'})

    lib['start'] = lib['start'].astype(int)
    lib['end'] = lib['start'] + lib['sgrna'].apply(len) + lib['PAM'].apply(len)

    lib = lib.groupby(['#chr', 'start', 'end', 'sgrna'])['PAM'].count().rename('ngenes').reset_index()

    return lib


def export_cn_bed_per_sample(samples):
    cnv = pd.read_csv(GECKOV2_CN, sep='\t')
    cnv = cnv.rename(columns=dict(Chromosome='#chr', Start='start', End='end', CN='total_cn'))
    cnv['#chr'] = cnv['#chr'].apply(lambda v: 'chr{}'.format(v))

    for s in samples:
        cnv.query('Sample == @s')[['#chr', 'start', 'end', 'total_cn']]\
            .to_csv(f'data/geckov2/crispr_bed/{s}.snp6.ABSOLUTE.bed', sep='\t', index=False)


def calculate_ploidy(snp_bed):
    df = snp_bed.to_dataframe().rename(columns={'name': 'total_cn', 'chrom': 'chr'})
    df = df.assign(length=df['end'] - df['start'])
    df = df.assign(cn_by_length=df['length'] * (df['total_cn'] + 1))

    chrm = df.groupby('chr')['cn_by_length'].sum().divide(df.groupby('chr')['length'].sum()) - 1

    ploidy = (df['cn_by_length'].sum() / df['length'].sum()) - 1

    return chrm, ploidy


def intersect_bed(sample, fc):
    # Import CRISPR lib
    lib = import_lib()

    # Input/Output files
    file_snp_bed = f'data/geckov2/crispr_bed/{sample}.snp6.ABSOLUTE.bed'
    file_crispr_bed = f'data/geckov2/crispr_bed/{sample}.crispr.bed'
    file_bed = f'data/geckov2/crispr_bed/{sample}.crispr.snp.bed'

    # Build CRISPR BED
    crispr_bed = lib.assign(fc=fc.loc[lib['sgrna'], sample].values)
    crispr_bed = crispr_bed[['#chr', 'start', 'end', 'fc', 'sgrna']]
    crispr_bed.to_csv(file_crispr_bed, index=False, sep='\t')

    # Import BED files
    snp_bed = BedTool(file_snp_bed).sort()
    crispr_bed = BedTool(file_crispr_bed).sort()

    # Intersect SNP6 copy-number segments with CRISPR-Cas9 sgRNAs
    bed = snp_bed.intersect(crispr_bed, wa=True, wb=True).to_dataframe(names=BED_NAMES)

    # Calculate chromosome copies and cell ploidy
    chrm, ploidy = calculate_ploidy(snp_bed)

    bed = bed.assign(chrm=chrm[bed['chr']].values)
    bed = bed.assign(ploidy=ploidy)

    # Calculate copy-number ratio
    bed = bed.assign(ratio=bed.eval('cn / chrm'))

    # Calculate segment length
    bed = bed.assign(len=bed.eval('end - start'))

    # Round
    round_cols = ['fc', 'chrm', 'ploidy', 'ratio']
    bed[round_cols] = bed[round_cols].round(5)

    # Scale fold-changes
    bed = scale_geckov2(bed)

    # Export
    bed[BED_NAMES_ORDER].to_csv(file_bed, index=False, sep='\t')

    return bed


def scale_geckov2(bed):
    lib = pd.read_csv(GECKOV2_SGRNA_MAP, sep='\t').dropna(subset=['Chromosome', 'Pos', 'Strand'])
    lib['sgrna'] = lib['Guide'].apply(lambda v: v.split('_')[0])
    lib['gene'] = lib['Guide'].apply(lambda v: v.split('_')[1])
    lib['cut'] = lib['Chromosome'] + '_' + lib['Pos'].astype(str)

    sgrna_offtarget = lib.groupby('sgrna')['cut'].agg(lambda x: len(set(x)))
    lib = lib[lib['sgrna'].isin(sgrna_offtarget[sgrna_offtarget == 1].index)]

    ess = cy.get_essential_genes()
    ness = cy.get_non_essential_genes()

    ess_metric = np.median(bed[bed['sgrna_id'].isin(lib[lib['gene'].isin(ess)]['sgrna'])]['fc'], axis=0)
    ness_metric = np.median(bed[bed['sgrna_id'].isin(lib[lib['gene'].isin(ness)]['sgrna'])]['fc'], axis=0)

    bed = bed.assign(fc_scaled=bed['fc'].subtract(ness_metric).divide(ness_metric - ess_metric))

    return bed


if __name__ == '__main__':
    # - Imports
    # Samplesheet
    ss = pd.read_csv(GECKOV2_SAMPLESHEET, sep='\t', index_col=1)

    # Raw Counts
    counts = DataMatrix(pd.read_csv(GECKOV2_RAW_COUNTS, index_col=0, skiprows=2, sep='\t'))
    counts = counts.drop('Description', axis=1)

    # - Processing
    # Add constant
    counts = counts + 1

    # Normalise by library size
    size_factors = counts.estimate_size_factors()
    counts = counts.divide(size_factors)

    # Remove plasmid low count sgRNAs
    counts.loc[counts[GECKOV2_PLASMID] < mp.LOW_COUNT_THRES, GECKOV2_PLASMID] = np.nan

    # Calculate log2 fold-changes: log2(sample / plasmid)
    fc = pd.DataFrame({c: np.log2(counts[c] / counts[GECKOV2_PLASMID]) for c in counts if c != GECKOV2_PLASMID})

    # Replicates correlation
    fc_corr = replicates_corr(ss)

    qc_samples = fc_corr[fc_corr['corr'] > fc_corr.query('replicate != 1')['corr'].mean()].query('replicate == 1')
    qc_samples = {s for s1, s2 in qc_samples[['sample_1', 'sample_2']].values for s in [s1, s2]}

    fc = fc[list(qc_samples)]

    # Average replicates
    fc = fc.groupby(ss.loc[[i.split(' Rep')[0] for i in fc.columns], 'name'].values, axis=1).mean()

    # -
    export_cn_bed_per_sample(list(fc))

    # - Intersect SNP6 segments with CRISPR-Cas9
    for s in fc:
        print(f'[INFO] BedTools: {s}')
        intersect_bed(s, fc)
