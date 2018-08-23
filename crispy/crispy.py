#!/usr/bin/env python
# Copyright (C) 2018 Emanuel Goncalves

import warnings
import numpy as np
import pandas as pd
import crispy as cy
import scipy.stats as st
import matplotlib.pyplot as plt
from pybedtools import BedTool
from sklearn.model_selection import ShuffleSplit
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
            self, raw_counts, library, copy_number=None, plasmid='Plasmid_v1.1'
    ):
        f"""
        Initialise a Crispy processing pipeline object

        :param raw_counts: pandas.DataFrame
            Unprocessed/Raw counts

        :param copy_number: pandas.DataFrame
            Copy-number segments, must have these columns {COPY_NUMBER_COLUMNS}

        :param library: pandas.DataFrame
            CRISPR library, must have these columns {CRISPR_LIB_COLUMNS}

        :param plasmid: str or list, optinal
            Column(s) in raw_counts to compare to other samples


        """
        self.raw_counts = raw_counts

        self.copy_number = copy_number

        self.library = cy.Utils.get_crispr_lib() if library is None else library

        self.plasmid = [plasmid] if type(plasmid) == str else plasmid

    def correct(self, x_features=None, qc_replicates_thres=.7, n_sgrna=30, round_dec=5):
        """
        Main pipeline function to process data from raw counts to corrected fold-changes.

        :param x_features: list
            Columns to be used as features to predict segments fold-changes

        :param qc_replicates_thres: float
            Replicate fold-change Pearson correlation threshold

        :param n_sgrna: int
            Minimum number of guides per segment

        :param round_dec: int
            Number of decimal places for floating numbers. If equals to None no rounding is performed

        :return: pandas.DataFrame
        """
        assert self.copy_number is not None, 'Copy-number information not provided'

        x_features = ['ratio'] if x_features is None else x_features

        # - Estimate fold-changes from raw counts
        fc = self.fold_changes(qc_replicates_thres=qc_replicates_thres)

        if fc is None:
            return fc

        # - Intersect sgRNAs genomic localisation with copy-number segments
        bed_df = self.intersect_sgrna_copynumber(fc)

        # - Fit Gaussian Process on segment fold-changes
        gpr = CrispyGaussian(bed_df, n_sgrna=n_sgrna)\
            .fit(x=x_features, y='fold_change')\

        gp_mean = gpr.predict(x=x_features)

        bed_df['gp_mean'] = gp_mean[bed_df.set_index(gpr.SEGMENT_COLUMNS).index].values

        # - Correct fold-change by subtracting the estimated mean
        bed_df['corrected'] = bed_df.eval('fold_change - gp_mean')

        # - Add gene
        bed_df['gene'] = self.library.set_index('sgrna').reindex(bed_df['sgrna'])['gene'].values

        # - Round floating
        if round_dec is not None:
            round_cols = ['fold_change', 'chr_copy', 'ploidy', 'ratio', 'len_log2', 'gp_mean', 'corrected']
            bed_df[round_cols] = bed_df[round_cols].round(round_dec)

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
        pd.set_option('display.max_colwidth', 10000)

        lib = self.library[CRISPR_LIB_COLUMNS]

        if fold_change is not None:
            lib = lib[lib['sgrna'].isin(fold_change.index)]
            lib = lib.assign(fold_change=fold_change[lib['sgrna']].values)

        lib = lib.assign(start=lib['start'].astype(int).values)
        lib = lib.assign(end=lib['end'].astype(int).values)

        lib_str = lib.to_string(index=False, header=False)

        return BedTool(lib_str, from_string=True).sort()

    @staticmethod
    def calculate_ploidy(df):
        """
        Estimate ploidy and chromosomes number of copies from copy-number segments. Mean copy-number is
        weigthed by the length of the segment.

        :param df: pandas.DataFrame or pybedtools.BedTool

        :return: (pandas.Series, float)
            Chromosome copies, ploidy
        """
        if type(df) == BedTool:
            df = df.to_dataframe().rename(columns={'name': 'copy_number', 'chrom': 'chr'})

        df = df[~df['chr'].isin(['chrX', 'chrY'])]

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
        df = self.raw_counts.loc[self.raw_counts[self.plasmid].mean(1) >= LOW_COUNT_THRES]

        # Calculate normalisation factors
        df = self.scale_raw_counts(df)

        # Calculate log fold-changes
        df = df.add(PSEUDO_COUNT) \
            .divide(df[self.plasmid].mean(1), axis=0)\
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

    # TODO: Note to potential duplciate sgrnas in library
    def gene_fold_changes(self, bed_df=None, qc_replicates_thres=.7):
        """
        Gene level fold-changes

        :param bed_df:

        :param qc_replicates_thres: float, optional default = 0.7
            Replicate fold-change Pearson correlation threshold

        :return: pandas.Series
        """

        if bed_df is None:
            fc = self.fold_changes(qc_replicates_thres=qc_replicates_thres)
        else:
            fc = bed_df.set_index('sgrna')['fold_change']

        genes = self.library.dropna(subset=['gene']).set_index('sgrna')['gene']

        return fc.groupby(genes).mean()

    def gene_corrected_fold_changes(self, bed_df):
        genes = self.library.dropna(subset=['gene']).set_index('sgrna')['gene']

        fc = bed_df.set_index('sgrna')['corrected']

        return fc.groupby(genes).mean()

    @staticmethod
    def scale_crispr(df, ess=None, ness=None, metric=np.median):
        """
        Min/Max scaling of CRISPR-Cas9 log-FC by median (default) Essential and Non-Essential.

        :param df: pandas.DataFrame
            Gene fold-changes
        :param ess: set(String)
        :param ness: set(String)
        :param metric: np.Median (default)

        :return: Float pandas.DataFrame

        """

        if ess is None:
            ess = cy.Utils.get_essential_genes(return_series=False)

        if ness is None:
            ness = cy.Utils.get_non_essential_genes(return_series=False)

        assert len(ess.intersection(df.index)) != 0, 'DataFrame has no index overlapping with essential list'
        assert len(ness.intersection(df.index)) != 0, 'DataFrame has no index overlapping with non-essential list'

        ess_metric = metric(df.reindex(ess).dropna(), axis=0)
        ness_metric = metric(df.reindex(ness).dropna(), axis=0)

        df = df.subtract(ness_metric).divide(ness_metric - ess_metric)

        return df


class CrispyGaussian(GaussianProcessRegressor):
    SEGMENT_COLUMNS = ['chr', 'start', 'end']

    SEGMENT_AGG_FUN = dict(
        fold_change=np.mean, copy_number=np.mean, ratio=np.mean, chr_copy=np.mean, ploidy=np.mean,
        len=np.mean, len_log2=np.mean, sgrna='count'
    )

    def __init__(
            self, bed_df, kernel=None, n_sgrna=30, alpha=1e-10, n_restarts_optimizer=3, optimizer='fmin_l_bfgs_b',
            normalize_y=True, copy_x_train=False, random_state=None
    ):

        self.bed_seg = bed_df.groupby(self.SEGMENT_COLUMNS).agg(self.SEGMENT_AGG_FUN).query(f'sgrna >= {n_sgrna}')
        self.n_sgrna = n_sgrna

        super().__init__(
            alpha=alpha,
            kernel=self.get_default_kernel() if kernel is None else kernel,
            optimizer=optimizer,
            normalize_y=normalize_y,
            random_state=random_state,
            copy_X_train=copy_x_train,
            n_restarts_optimizer=n_restarts_optimizer,
        )

    @property
    def _constructor(self):
        return CrispyGaussian

    @staticmethod
    def get_default_kernel():
        """
        Deafult Gaussian Process kernel

        :return: sklearn.gaussian_process.kernels.Sum
        """
        return ConstantKernel() * RBF() + WhiteKernel()

    def fit(self, x, y, train_idx=None):
        if train_idx is not None:
            return super().fit(self.bed_seg.iloc[train_idx][x], self.bed_seg.iloc[train_idx][y])
        else:
            return super().fit(self.bed_seg[x], self.bed_seg[y])

    def predict(self, x=None, return_std=False, return_cov=False):
        if x is None:
            x = self.bed_seg[['ratio']]
        elif type(x) == list:
            x = self.bed_seg[x]

        if return_std or return_cov:
            gp_mean, gp_std_cor = super().predict(x, return_std=return_std, return_cov=return_cov)
            gp_mean = pd.Series(gp_mean, index=x.index).rename('gp_mean')
            return gp_mean, gp_std_cor

        else:
            gp_mean = super().predict(x)
            gp_mean = pd.Series(gp_mean, index=x.index).rename('gp_mean')
            return gp_mean

    def score(self, x=None, y=None, sample_weight=None):
        if x is None:
            x = self.bed_seg[['ratio']]
        elif type(x) == list:
            x = self.bed_seg[x]

        if y is None:
            y = self.bed_seg['fold_change']

        return super().score(x, y, sample_weight=sample_weight)

    def plot(self, x='ratio', y='fold_change', ax=None):
        """
        Plot original data and GPR

        :param x: str, x axis variable
        :param y: str, y axis variable
        :param ax: matplotlib.plot
        :return: matplotlib.plot
        """

        if ax is None:
            ax = plt.gca()

        # Plot data
        ax.scatter(self.bed_seg[x], self.bed_seg[y], marker='o', s=5, alpha=.4, color=cy.QCplot.PAL_DBGD[0], lw=0)

        # Plot GP fit
        x_ = np.arange(0, self.bed_seg[x].max(), .1)
        y_mean, y_std = self.predict(x_.reshape(-1, 1), return_std=True)

        ax.plot(x_, y_mean, color=cy.QCplot.PAL_DBGD[1], lw=1, label='GPR', zorder=3)
        ax.fill_between(x_, y_mean - y_std, y_mean + y_std, color=cy.QCplot.PAL_DBGD[1], alpha=0.2, zorder=3, lw=0)

        # Misc
        ax.axhline(0, lw=.3, color=cy.QCplot.PAL_DBGD[2], ls=':', zorder=0)

        ax.legend(frameon=False, prop={'size': 4}, loc=3)

        ax.set_xlabel('Copy-number')
        ax.set_ylabel('Fold-change')

        return ax
