#!/usr/bin/env python
# Copyright (C) 2018 Emanuel Goncalves

import warnings
import numpy as np
import pandas as pd
import crispy as cy
import scipy.stats as st
import matplotlib.pyplot as plt
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
            self, raw_counts, library=None, copy_number=None, plasmid=None
    ):
        f"""
        Initialise a Crispy processing pipeline object

        :param raw_counts: pandas.DataFrame
            Unprocessed/Raw counts

        :param copy_number: pandas.DataFrame
            Copy-number segments, must have these columns {COPY_NUMBER_COLUMNS}

        :param library: pandas.DataFrame
            CRISPR library, must have these columns {CRISPR_LIB_COLUMNS}

        :param plasmid: str or list, optional
            Column(s) in raw_counts to compare to other samples. If None last column is taken as control.


        """
        self.raw_counts = raw_counts

        self.copy_number = copy_number

        self.library = cy.Utils.get_crispr_lib() if library is None else library

        if plasmid is None:
            self.plasmid = [self.raw_counts.columns[-1]]

        elif type(plasmid) == str:
            self.plasmid = [plasmid]

        else:
            self.plasmid = plasmid

        self.gpr = None

    def correct(self, x_features=None, y_feature='fold_change', qc_replicates_thres=None, n_sgrna=10, round_dec=5):
        """
        Main pipeline function to process data from raw counts to corrected fold-changes.

        :param x_features: list
            Columns to be used as feature(s)

        :param y_feature: str
            Dependent variable

        :param qc_replicates_thres: float
            Replicate fold-change Pearson correlation threshold

        :param n_sgrna: int
            Minimum number of guides per segment

        :param round_dec: int
            Number of decimal places for floating numbers. If equals to None no rounding is performed

        :return: pandas.DataFrame
        """
        assert self.copy_number is not None, 'Copy-number information not provided'

        if x_features is None:
            x_features = ['ratio']

        elif type(x_features) == str:
            x_features = [x_features]

        else:
            x_features = x_features

        # - Estimate fold-changes from raw counts
        fc = self.fold_changes(qc_replicates_thres=qc_replicates_thres)

        if fc is None:
            return fc

        # - Intersect sgRNAs genomic localisation with copy-number segments
        bed_df = self.intersect_sgrna_copynumber(fc)

        # - Fit Gaussian Process on segment fold-changes
        self.gpr = CrispyGaussian(bed_df, n_sgrna=n_sgrna).fit(x=x_features, y=y_feature)

        bed_df['gp_mean'] = self.gpr.predict(bed_df[x_features])

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

    def fold_changes(self, qc_replicates_thres=.7, average_replicates=True):
        """
        Fold-changes estimation

        :param qc_replicates_thres: float, optional default = 0.7
            Replicate fold-change Pearson correlation threshold

        :param average_replicates: bool

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
        if average_replicates:
            fold_changes = fold_changes.mean(1)

        return fold_changes

    # TODO: Note to potential duplciate sgrnas in library
    def gene_fold_changes(self, bed_df=None, qc_replicates_thres=.7, average_replicates=True):
        """
        Gene level fold-changes

        :param bed_df:

        :param qc_replicates_thres: float, optional default = 0.7
            Replicate fold-change Pearson correlation threshold

        :param average_replicates:

        :return: pandas.Series
        """

        if bed_df is None:
            fc = self.fold_changes(qc_replicates_thres=qc_replicates_thres, average_replicates=average_replicates)
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
            self, bed_df, kernel=None, n_sgrna=10, alpha=1e-10, n_restarts_optimizer=3, optimizer='fmin_l_bfgs_b',
            normalize_y=False, copy_x_train=False, random_state=None
    ):

        self.bed_seg = bed_df.groupby(self.SEGMENT_COLUMNS).agg(self.SEGMENT_AGG_FUN)
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
            x = self.bed_seg.iloc[train_idx].query(f'sgrna >= {self.n_sgrna}')[x]
            y = self.bed_seg.iloc[train_idx].query(f'sgrna >= {self.n_sgrna}')[y]

        else:
            x = self.bed_seg.query(f'sgrna >= {self.n_sgrna}')[x]
            y = self.bed_seg.query(f'sgrna >= {self.n_sgrna}')[y]

        return super().fit(x, y)

    def predict(self, x=None, return_std=False, return_cov=False):
        if x is None:
            x = self.bed_seg[['ratio']]

        return super().predict(x, return_std=return_std, return_cov=return_cov)

    def score(self, x=None, y=None, sample_weight=None):
        if x is None:
            x = self.bed_seg[['ratio']]
        elif type(x) == list:
            x = self.bed_seg[x]

        if y is None:
            y = self.bed_seg['fold_change']

        return super().score(x, y, sample_weight=sample_weight)

    def plot(self, x_feature='ratio', y_feature='fold_change', ax=None):
        """
        Plot original data and GPR

        :param x_feature: str, x axis variable
        :param y_feature: str, y axis variable
        :param ax: matplotlib.plot
        :return: matplotlib.plot
        """

        if ax is None:
            ax = plt.gca()

        # - Data
        x, y = self.bed_seg.query(f'sgrna >= {self.n_sgrna}')[x_feature], self.bed_seg.query(f'sgrna >= {self.n_sgrna}')[y_feature]
        x_, y_ = self.bed_seg.query(f'sgrna < {self.n_sgrna}')[x_feature], self.bed_seg.query(f'sgrna < {self.n_sgrna}')[y_feature]

        x_pred = np.arange(0, x.max(), .1)
        y_pred, y_pred_std = self.predict(x_pred.reshape(-1, 1), return_std=True)

        # - Plot
        # Segments used for fitting
        ax.scatter(
            x, y, c=cy.QCplot.PAL_DBGD[0], alpha=.7, edgecolors='white', lw=.3, label=f'#(sgRNA) >= {self.n_sgrna}'
        )

        # Segments not used for fitting
        plt.scatter(
            x_, y_, c=cy.QCplot.PAL_DBGD[0], marker='X', alpha=.3, edgecolors='white', lw=.3, label=f'#(sgRNA) < {self.n_sgrna}'
        )

        # Plot GP fit
        # GP fit
        plt.plot(x_pred, y_pred, ls='-', lw=1., c=cy.QCplot.PAL_DBGD[1], label='GPR mean')
        plt.fill_between(x_pred, y_pred - y_pred_std, y_pred + y_pred_std, alpha=0.2, color=cy.QCplot.PAL_DBGD[1], lw=0)

        # Misc
        plt.axhline(0, ls=':', color=cy.QCplot.PAL_DBGD[2], lw=.3, zorder=0)

        plt.xlabel(f'Segment\n{x_feature}')
        plt.ylabel(f'Segment\nmean {y_feature}')

        plt.title(f'{self.kernel_}', fontsize=6)

        plt.legend(frameon=False)

        return ax
