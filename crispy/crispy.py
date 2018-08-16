#!/usr/bin/env python
# Copyright (C) 2018 Emanuel Goncalves

import warnings
import numpy as np
import pandas as pd
import crispy as cy
import scipy.stats as st
from pybedtools import BedTool
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import WhiteKernel, ConstantKernel, RBF


PSEUDO_COUNT = .5

LOW_COUNT_THRES = 30

COPY_NUMBER_COLUMNS = ['chr', 'start', 'end', 'copy_number']

CRISPR_LIB_COLUMNS = ['chr', 'start', 'end', 'sgrna']

BED_COLUMNS = ['chr', 'start', 'end', 'copy_number', 'sgrna_chr', 'sgrna_start', 'sgrna_end', 'sgrna', 'fold_change']


# TODO: add check for minimum number of reads 15M
class Crispy(object):

    def __init__(
            self, raw_counts, library, copy_number=None, plasmid='Plasmid_v1.1', kernel=None
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


        """
        self.raw_counts = raw_counts

        self.copy_number = copy_number

        self.library = cy.get_crispr_lib() if library is None else library

        self.plasmid = plasmid

        self.kernel = self.get_default_kernel() if kernel is None else kernel

    def correct(self, x_features=None, qc_replicates_thres=.7, round_dec=5):
        """
        Main pipeline function to process data from raw counts to corrected fold-changes.

        :param x_features: list
            Columns to be used as features to predict segments fold-changes

        :param qc_replicates_thres: float
            Replicate fold-change Pearson correlation threshold

        :param round_dec: int
            Number of decimal places for floating numbers. If equals to None no rounding is performed

        :return: pandas.DataFrame
        """
        assert self.copy_number is not None, 'Copy-number information not provided'

        x_features = ['copy_number', 'chr_copy', 'ratio', 'len_log2'] if x_features is None else x_features

        # - Estimate fold-changes from raw counts
        fc = self.fold_changes(qc_replicates_thres=qc_replicates_thres)

        if fc is None:
            return fc

        # - Intersect sgRNAs genomic localisation with copy-number segments
        bed_df = self.intersect_sgrna_copynumber(fc)

        # - Fit Gaussian Process on segment fold-changes
        seg_cols = ['chr', 'start', 'end']

        agg_fun = dict(
            fold_change=np.mean, copy_number=np.mean, chr_copy=np.mean, ploidy=np.mean, ratio=np.mean, len_log2=np.mean
        )

        # Aggregate features by segment
        bed_df_seg = bed_df.groupby(seg_cols).agg(agg_fun)

        # Fit averaged fold-changes across segments
        gp_mean = self.gp_fit(bed_df_seg[x_features], bed_df_seg['fold_change'])

        # Append estimated averages back into the bed file
        bed_df['gp_mean'] = gp_mean[bed_df.set_index(seg_cols).index].values

        # Correct fold-change by subtracting the estimated mean
        bed_df['corrected'] = bed_df.eval('fold_change - gp_mean')

        # Round floating
        if round_dec is not None:
            round_cols = ['fold_change', 'chr_copy', 'ploidy', 'ratio', 'len_log2', 'gp_mean', 'corrected']
            bed_df[round_cols] = bed_df[round_cols].round(round_dec)

        return bed_df

    @staticmethod
    def get_default_kernel():
        """
        Deafult Gaussian Process kernel

        :return: sklearn.gaussian_process.kernels.Sum
        """
        return ConstantKernel() * RBF(length_scale_bounds=(1e-5, 10)) + WhiteKernel()

    def gp_fit(self, x, y, alpha=1e-10, n_restarts_optimizer=3):
        """
        Fit Gaussian Process Regressor.

        :param x: pandas.DataFrame
            Features, e.g. copy-number, segment lenght, copy-number ratio

        :param y: pandas.Series
            Dependent variable, e.g. copy-number segments average fold-changes

        :param alpha:

        :param n_restarts_optimizer:

        :return: pandas.Series
            Estimated fold-changes
        """
        gpr = GaussianProcessRegressor(
            self.kernel, alpha=alpha, n_restarts_optimizer=n_restarts_optimizer
        )

        gpr = gpr.fit(x, y)

        gp_mean = pd.Series(gpr.predict(x), index=x.index).rename('gp_mean')

        return gp_mean

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

        bed_df = bed_df.assign(chr_copy=chrm[bed_df['chr']].values)
        bed_df = bed_df.assign(ploidy=ploidy)

        # Calculate copy-number ratio
        bed_df = bed_df.assign(ratio=bed_df.eval('copy_number / chr_copy'))

        # Calculate segment length
        bed_df = bed_df.assign(len=bed_df.eval('end - start'))
        bed_df = bed_df.assign(len_log2=bed_df['len'].apply(np.log2))

        return bed_df

    @staticmethod
    def estimate_size_factors(raw_counts):
        """
        Normalisation coefficients are estimated similarly to DESeq2 (DOI: 10.1186/s13059-014-0550-8).

        :returns pandas.Series: Normalisation factors

        """
        geom_mean = st.gmean(raw_counts, axis=1)

        factors = raw_counts.divide(geom_mean, axis=0)\
            .median()\
            .rename('size_factors')

        return factors

    @staticmethod
    def library_size_factor(raw_counts):
        return raw_counts.sum().rename('size_factors')

    @staticmethod
    def replicates_correlation(fold_changes):
        """
        Raw counts pearson correlations

        :returns pandas.Series: Pearson correlation coefficients

        """
        corr = fold_changes.corr()

        corr = corr.where(np.triu(np.ones(corr.shape), 1).astype(np.bool))
        corr = corr.unstack().dropna().reset_index()
        corr = corr.set_axis(['sample_1', 'sample_2', 'corr'], axis=1, inplace=False)

        return corr

    def scale_raw_counts(self, counts, scale=1e7):
        factors = self.library_size_factor(counts)

        return counts.divide(factors) * scale

    def replicates_foldchanges(self):
        """
        Estimate replicates fold-changes
        :return: pandas.DataFrame
        """
        # Remove plasmid low count sgRNAs
        df = self.raw_counts.loc[self.raw_counts[self.plasmid] >= LOW_COUNT_THRES]

        # Calculate normalisation factors
        df = self.scale_raw_counts(df)

        # Calculate log fold-changes
        df = df.add(PSEUDO_COUNT) \
            .divide(df[self.plasmid], axis=0)\
            .drop(self.plasmid, axis=1) \
            .apply(np.log2)

        return df

    def fold_changes(self, qc_replicates_thres=.7):
        """
        Fold-changes estimation

        :param qc_replicates_thres: float, optional default = 0.7
            Replicate fold-change Pearson correlation threshold

        :returns pandas.Series: fold-changes compared to plasmid

        """
        fold_changes = self.replicates_foldchanges()

        # Remove replicates with low correlation
        if qc_replicates_thres is not None:
            corr = self.replicates_correlation(fold_changes)

            samples = corr.query(f'corr > {qc_replicates_thres}')
            samples = pd.unique(samples[['sample_1', 'sample_2']].unstack())

            discarded_samples = [s for s in fold_changes if s not in samples]

            if len(discarded_samples) > 0:
                warnings.warn(f'QC correlation, replicates discarded: {discarded_samples}')

            if len(samples) == 0:
                warnings.warn(f'All replicates failed QC correlation threshold {qc_replicates_thres:.2f}')
                return None

            fold_changes = fold_changes[samples]

        # Average replicates
        fold_changes = fold_changes.mean(1)

        return fold_changes

    def gene_fold_changes(self, qc_replicates_thres=.7):
        """
        Gene level fold-changes

        :param qc_replicates_thres: float, optional default = 0.7
            Replicate fold-change Pearson correlation threshold

        :return: pandas.Series
        """
        fc = self.fold_changes(qc_replicates_thres=qc_replicates_thres)

        genes = self.library.dropna(subset=['gene']).set_index('sgrna')['gene']

        fc = fc.groupby(genes).mean()

        return fc
