#!/usr/bin/env python
# Copyright (C) 2018 Emanuel Goncalves

import pandas as pd
import scripts as mp
import seaborn as sns
import matplotlib.pyplot as plt
from crispy.benchmark_plot import plot_chromosome, import_cytobands


if __name__ == '__main__':
    # CRISPR lib
    lib = pd.read_csv(mp.LIBRARY, index_col=0)
    lib = lib.assign(pos=lib[['start', 'end']].mean(1).values)

    # - Chromosome plot
    for sample, chrm in [('HCC1954', 'chr8'), ('NCI-H2087', 'chr8')]:
        plot_df = pd.read_csv('data/crispy/gdsc/crispy_crispr_{}.csv'.format(sample), index_col=0).query("fit_by == '{}'".format(chrm))
        plot_df = plot_df.assign(pos=lib.groupby('gene')['pos'].mean().loc[plot_df.index])

        cytobands = import_cytobands(chrm=chrm)

        seg = pd.read_csv('data/gdsc/copynumber/snp6_bed/{}.snp6.picnic.bed'.format(sample), sep='\t')
        # seg = pd.read_csv('data/gdsc/wgs/ascat_bed/{}.wgs.ascat.bed.ratio.bed'.format(sample), sep='\t')

        ax = plot_chromosome(
            plot_df['pos'], plot_df['fc'].rename('CRISPR-Cas9'), plot_df['k_mean'].rename('Fitted mean'), seg=seg[seg['#chr'] == chrm],
            highlight=['MYC'], cytobands=cytobands, legend=True
        )

        ax.set_xlabel('Position on chromosome {} (Mb)'.format(chrm))
        ax.set_ylabel('Fold-change')
        ax.set_title('{}'.format(sample))

        plt.xlim(100, 148)

        plt.gcf().set_size_inches(2.5, 2.5)
        plt.savefig('reports/crispy/chromosomeplot_{}_{}.png'.format(sample, chrm), bbox_inches='tight', dpi=600)
        plt.close('all')