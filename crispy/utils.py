#!/usr/bin/env python
# Copyright (C) 2018 Emanuel Goncalves

import random
import qc_plot
import operator
import numpy as np
import pandas as pd
import pkg_resources
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.stats import gaussian_kde, rankdata, norm


class Utils(object):
    DPATH = pkg_resources.resource_filename('crispy', 'data/')

    CHR_SIZES_HG19 = {
        'chr1': 249250621, 'chr2': 243199373, 'chr3': 198022430, 'chr4': 191154276, 'chr5': 180915260, 'chr6': 171115067, 'chr7': 159138663, 'chr8': 146364022,
        'chr9': 141213431, 'chr10': 135534747, 'chr11': 135006516, 'chr12': 133851895, 'chr13': 115169878, 'chr14': 107349540, 'chr15': 102531392, 'chr16': 90354753,
        'chr17': 81195210, 'chr18': 78077248, 'chr19': 59128983, 'chr20': 63025520, 'chr21': 48129895, 'chr22': 51304566, 'chrX': 155270560, 'chrY': 59373566
    }

    BRASS_HEADERS = [
        'chr1', 'start1', 'end1', 'chr2', 'start2', 'end2', 'id/name', 'brass_score', 'strand1', 'strand2',
        'sample', 'svclass', 'bkdist', 'assembly_score', 'readpair names', 'readpair count', 'bal_trans', 'inv',
        'occL', 'occH', 'copynumber_flag', 'range_blat', 'Brass Notation', 'non-template', 'micro-homology',
        'assembled readnames', 'assembled read count', 'gene1', 'gene_id1', 'transcript_id1', 'strand1', 'end_phase1',
        'region1', 'region_number1', 'total_region_count1', 'first/last1', 'gene2', 'gene_id2', 'transcript_id2',
        'strand2', 'phase2', 'region2', 'region_number2', 'total_region_count2', 'first/last2', 'fusion_flag'
    ]

    ASCAT_HEADERS = ['chr', 'start', 'end', 'copy_number']

    @staticmethod
    def bin_bkdist(distance):
        """
        Discretise genomic distances

        :param distance: float

        :return:

        """
        if distance == -1:
            bin_distance = 'diff. chr.'

        elif distance == 0:
            bin_distance = '0'

        elif 1 < distance < 10e3:
            bin_distance = '1-10 kb'

        elif 10e3 < distance < 100e3:
            bin_distance = '10-100 kb'

        elif 100e3 < distance < 1e6:
            bin_distance = '0.1-1 Mb'

        elif 1e6 < distance < 10e6:
            bin_distance = '1-10 Mb'

        else:
            bin_distance = '>10 Mb'

        return bin_distance

    @staticmethod
    def svtype(strand1, strand2, svclass, unfold_inversions):
        if svclass == 'translocation':
            svtype = 'translocation'

        elif strand1 == '+' and strand2 == '+':
            svtype = 'deletion'

        elif strand1 == '-' and strand2 == '-':
            svtype = 'tandem-duplication'

        elif strand1 == '+' and strand2 == '-':
            svtype = 'inversion_h_h' if unfold_inversions else 'inversion'

        elif strand1 == '-' and strand2 == '+':
            svtype = 'inversion_t_t' if unfold_inversions else 'inversion'

        else:
            assert False, 'SV class not recognised: strand1 == {}; strand2 == {}; svclass == {}'.format(strand1, strand2, svclass)

        return svtype

    @staticmethod
    def bin_cnv(value, thresold):
        if not np.isfinite(value):
            return np.nan

        value = int(round(value, 0))
        value = f'{value}' if value < thresold else f'{thresold}+'

        return value

    @staticmethod
    def qnorm(x):
        y = rankdata(x)
        y = -norm.isf(y / (len(x) + 1))
        return y

    @classmethod
    def get_example_data(cls):
        raw_counts = pd.read_csv('{}/{}'.format(cls.DPATH, 'example_rawcounts.csv'), index_col=0)
        copynumber = pd.read_csv('{}/{}'.format(cls.DPATH, 'example_copynumber.csv'))
        return raw_counts, copynumber

    @classmethod
    def get_essential_genes(cls, dfile='gene_sets/curated_BAGEL_essential.csv', return_series=True):
        geneset = set(pd.read_csv('{}/{}'.format(cls.DPATH, dfile), sep='\t')['gene'])

        if return_series:
            geneset = pd.Series(list(geneset)).rename('essential')

        return geneset

    @classmethod
    def get_non_essential_genes(cls, dfile='gene_sets/curated_BAGEL_nonEssential.csv', return_series=True):
        geneset = set(pd.read_csv('{}/{}'.format(cls.DPATH, dfile), sep='\t')['gene'])

        if return_series:
            geneset = pd.Series(list(geneset)).rename('non-essential')

        return geneset

    @classmethod
    def get_crispr_lib(cls, dfile='crispr_libs/KY_Library_v1.1_annotated.csv'):
        r_cols = dict(
            index='sgrna', CHRM='chr', STARTpos='start', ENDpos='end', GENES='gene', EXONE='exon', CODE='code', STRAND='strand'
        )

        lib = pd.read_csv('{}/{}'.format(cls.DPATH, dfile), index_col=0).reset_index()

        lib = lib.rename(columns=r_cols)

        lib['chr'] = lib['chr'].apply(lambda v: f'chr{v}' if str(v) != 'nan' else np.nan)

        return lib

    @classmethod
    def get_adam_core_essential(cls, dfile='gene_sets/pancan_core.csv'):
        return set(pd.read_csv('{}/{}'.format(cls.DPATH, dfile))['ADAM PanCancer Core-Fitness genes'].rename('adam_essential'))

    @classmethod
    def get_cytobands(cls, dfile='cytoBand.txt', chrm=None):
        cytobands = pd.read_csv('{}/{}'.format(cls.DPATH, dfile), sep='\t')

        if chrm is not None:
            cytobands = cytobands[cytobands['chr'] == chrm]

        assert cytobands.shape[0] > 0, '{} not found in cytobands file'

        return cytobands

    @classmethod
    def import_brass_bedpe(cls, bedpe_file, bkdist=None, splitreads=True):
        # Import BRASS bedpe
        bedpe_df = pd.read_csv(bedpe_file, sep='\t', names=cls.BRASS_HEADERS, comment='#')

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

        return bedpe_df

    @staticmethod
    def gkn(values, bound=1e7):
        kernel = gaussian_kde(values)
        kernel = pd.Series({
            k: np.log(kernel.integrate_box_1d(-bound, v) / kernel.integrate_box_1d(v, bound)) for k, v in values.to_dict().items()
        })
        return kernel


class DotDict(dict):
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


class SSGSEA(object):

    @classmethod
    def gsea(cls, dataset, signature, permutations=0):
        # Sort data-set by values
        _dataset = list(zip(*sorted(dataset.items(), key=operator.itemgetter(1), reverse=True)))
        genes, expression = _dataset[0], _dataset[1]

        # Signature overlapping the data-set
        _signature = set(signature).intersection(genes)

        # Check signature overlap
        e_score, p_value = np.NaN, np.NaN
        hits, running_hit = [], []
        if len(_signature) != 0:

            # ---- Calculate signature enrichment score
            n, sig_size = len(genes), len(_signature)
            nh = n - sig_size
            nr = sum([abs(dataset[g]) for g in _signature])

            e_score = cls.__es(genes, expression, _signature, nr, nh, n, hits, running_hit)

            # ---- Calculate statistical enrichment
            # Generate random signatures sampled from the data-set genes
            if permutations > 0:
                count = 0

                for i in range(permutations):
                    r_signature = random.sample(genes, sig_size)

                    r_nr = sum([abs(dataset[g]) for g in r_signature])

                    r_es = cls.__es(genes, expression, r_signature, r_nr, nh, n)

                    if (r_es >= e_score >= 0) or (r_es <= e_score < 0):
                        count += 1

                # If no permutation was above the Enrichment score the p-value is lower than 1 divided by the number of permutations
                p_value = 1 / permutations if count == 0 else count / permutations

            else:
                p_value = np.nan

        return e_score, p_value, hits, running_hit

    @staticmethod
    def __es(genes, expression, signature, nr, nh, n, hits=None, running_hit=None):
        hit, miss, es, r = 0, 0, 0, 0
        for i in range(n):
            if genes[i] in signature:
                hit += abs(expression[i]) / nr

                if hits is not None:
                    hits.append(1)

            else:
                miss += 1 / nh

                if hits is not None:
                    hits.append(0)

            r = hit - miss

            if running_hit is not None:
                running_hit.append(r)

            if abs(r) > abs(es):
                es = r

        return es

    @classmethod
    def plot_gsea(cls, hits, running_hit, dataset=None, vertical_lines=False, shade=False):
        x, y = np.array(range(len(hits))), np.array(running_hit)

        if dataset is not None:
            gs = GridSpec(2, 1, height_ratios=[3, 2], hspace=0.1)
            axs = [plt.subplot(gs[0]), plt.subplot(gs[1])]

        else:
            axs = [plt.gca()]

        # GSEA running hit
        axs[0].plot(x, y, '-', c=qc_plot.QCplot.PAL_DBGD[0])

        if shade:
            axs[0].fill_between(x, 0, y, alpha=.5, color=qc_plot.QCplot.PAL_DBGD[2])

        if vertical_lines:
            for i in x[np.array(hits, dtype='bool')]:
                axs[0].axvline(i, c=qc_plot.QCplot.PAL_DBGD[0], lw=.3, alpha=.2, zorder=0)

        axs[0].axhline(0, c=qc_plot.QCplot.PAL_DBGD[0], lw=.1, ls='-')
        axs[0].set_ylabel('Enrichment score')
        axs[0].get_xaxis().set_visible(False)
        axs[0].set_xlim([0, len(x)])

        if dataset is not None:
            dataset = list(zip(*sorted(dataset.items(), key=operator.itemgetter(1), reverse=False)))

            # Data
            axs[1].scatter(x, dataset[1], c=qc_plot.QCplot.PAL_DBGD[0], linewidths=0, s=2)

            if shade:
                axs[1].fill_between(x, 0, dataset[1], alpha=.5, color=qc_plot.QCplot.PAL_DBGD[2])

            if vertical_lines:
                for i in x[np.array(hits, dtype='bool')]:
                    axs[1].axvline(i, c=qc_plot.QCplot.PAL_DBGD[0], lw=.3, alpha=.2, zorder=0)

            axs[1].axhline(0, c='black', lw=.3, ls='-')
            axs[1].set_ylabel('Data value')
            axs[1].get_xaxis().set_visible(False)
            axs[1].set_xlim([0, len(x)])

        return axs[0] if dataset is None else axs
