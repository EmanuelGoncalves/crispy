#!/usr/bin/env python
# Copyright (C) 2018 Emanuel Goncalves

import numpy as np
import pandas as pd
import scipy.stats as st
from pybedtools import BedTool


PSEUDO_COUNT = 1
LOW_COUNT_THRES = 30

COPY_NUMBER_COLUMNS = ['chr', 'start', 'end', 'copy_number']

CRISPR_LIB_COLUMNS = ['chr', 'start', 'end', 'sgrna']

BED_COLUMNS = ['chr', 'start', 'end', 'copy_number', 'sgrna_chr', 'sgrna_start', 'sgrna_end', 'sgrna', 'fold_change']


class Crispy(object):

    def __init__(
            self, raw_counts, copy_number, library, plasmid='control', qc_replicates=True, qc_replicates_thres=.7
    ):
        f"""
        Initialise a Crispy processing pipeline object

        :param raw_counts: pandas.DataFrame
            Unprocessed/Raw counts

        :param copy_number: pandas.DataFrame
            Copy-number segments, must have these columns {COPY_NUMBER_COLUMNS}

        :param library: pandas.DataFrame
            CRISPR library, must have these columns {CRISPR_LIB_COLUMNS}

        :param plasmid: str, optinal
            Column in raw_counts to compare to other samples

        :param qc_replicates: bool, optional
            Perform QC replicates analysis

        :param qc_replicates_thres: float, optional
            QC replicates analysis Pearson correlation coeficient threshold

        """

        self.raw_counts = raw_counts

        self.copy_number = copy_number

        self.library = library

        self.plasmid = plasmid

        self.qc_replicates = qc_replicates
        self.qc_replicates_thres = qc_replicates_thres

    def correct(self):
        fc = self.fold_changes()

        bed_df = self.intersect_sgrna_copynumber(fc)

        return bed_df

    def get_bed_copy_number(self):
        """
        Create BedTool object from copy-number segments data-frame

        :return: pybedtools.BedTool
        """
        cn_str = self.copy_number[COPY_NUMBER_COLUMNS].to_string(index=False, header=False)
        return BedTool(cn_str, from_string=True).sort()

    def get_bed_library(self, fold_change=None):
        """
        Create BedTool object for sgRNAs in CRISPR-Cas9 library

        :param fold_change: pandas.Series, optional
            Estimated fold-changes

        :return: pybedtools.BedTool
        """
        lib = self.library[CRISPR_LIB_COLUMNS]

        if fold_change is not None:
            lib['fold_change'] = fold_change[lib['sgrna'].values].values

        lib_str = lib.to_string(index=False, header=False)

        return BedTool(lib_str, from_string=True).sort()

    @staticmethod
    def calculate_ploidy(copy_number_bed):
        """
        Estimate ploidy and chromosomes number of copies from copy-number segments. Mean copy-number is
        weigthed by the length of the segment.

        :param copy_number_bed: pybedtools.BedTool

        :return: (pandas.Series, float)
            Chromosome copies, ploidy
        """
        df = copy_number_bed.to_dataframe().rename(columns={'name': 'copy_number', 'chrom': 'chr'})
        df = df.assign(length=df['end'] - df['start'])
        df = df.assign(cn_by_length=df['length'] * (df['copy_number'] + 1))

        chrm = df.groupby('chr')['cn_by_length'].sum().divide(df.groupby('chr')['length'].sum()) - 1

        ploidy = (df['cn_by_length'].sum() / df['length'].sum()) - 1

        return chrm, ploidy

    def intersect_sgrna_copynumber(self, fold_change):
        """
        Intersect sgRNAs fold-changes with copy-number segments

        :param fold_change: pandas.Series
            sgRNAs fold-changes

        :return: pandas.DataFrame
            DataFrame containing the intersection of the copy-number segments with the sgRNAs of the CRISPR library

        """
        # Build copy-number and sgRNAs bed files
        copy_number_bed = self.get_bed_copy_number()
        sgrnas_bed = self.get_bed_library(fold_change)

        # Intersect copy-number segments with sgRNAs
        bed_df = copy_number_bed.intersect(sgrnas_bed, wa=True, wb=True)\
            .to_dataframe(names=BED_COLUMNS)\
            .drop(['sgrna_chr'], axis=1)

        # Calculate chromosome copies and cell ploidy
        chrm, ploidy = self.calculate_ploidy(copy_number_bed)

        bed_df = bed_df.assign(chrm_copy=chrm[bed_df['chr']].values)
        bed_df = bed_df.assign(ploidy=ploidy)

        # Calculate copy-number ratio
        bed_df = bed_df.assign(ratio=bed_df.eval('copy_number / chrm_copy'))

        # Calculate segment length
        bed_df = bed_df.assign(len=bed_df.eval('end - start'))

        return bed_df

    def estimate_size_factors(self):
        """
        Normalisation coefficients are estimated similarly to DESeq2 (DOI: 10.1186/s13059-014-0550-8).

        :returns pandas.Series: Normalisation factors

        """
        geom_mean = st.gmean(self.raw_counts, axis=1)

        factors = self.raw_counts.divide(geom_mean, axis=0)\
            .median()\
            .rename('size_factors')

        return factors

    def replicates_correlation(self):
        """
        Raw counts pearson correlations

        :returns pandas.Series: Pearson correlation coefficients

        """
        corr = self.raw_counts.drop(self.plasmid, axis=1).corr()

        corr = corr.where(np.triu(np.ones(corr.shape), 1).astype(np.bool))
        corr = corr.unstack().dropna().reset_index()
        corr = corr.set_axis(['sample_1', 'sample_2', 'corr'], axis=1, inplace=False)

        return corr

    # TODO: Add print output to replicates excluded by QC
    def fold_changes(self, round_dec=5):
        """
        Fold-changes estimation

        :returns pandas.Series: fold-changes compared to plasmid

        """
        # Add pseudocount
        df = self.raw_counts + PSEUDO_COUNT

        # Calculate normalisation factors
        factors = self.estimate_size_factors()
        df = df.divide(factors)

        # Remove plasmid low count sgRNAs
        df = df.loc[self.raw_counts[self.plasmid] > LOW_COUNT_THRES]

        # Calculate log fold-changes
        df = df.divide(df[self.plasmid], axis=0)\
            .drop(self.plasmid, axis=1) \
            .apply(np.log2)

        # Remove replicates with low correlation
        if self.qc_replicates:
            corr = self.replicates_correlation()

            corr = corr.query(f'corr >= {self.qc_replicates_thres}')
            assert corr.shape[0] > 0, f'All replicates failed QC correlation threshold {self.qc_replicates_thres}'

            df = df[pd.unique(corr[['sample_1', 'sample_2']].unstack())]

        # Average replicates
        df = df.mean(1)

        # Round
        df = df.round(round_dec)

        return df

