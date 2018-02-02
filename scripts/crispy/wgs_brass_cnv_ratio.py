#!/usr/bin/env python
# Copyright (C) 2018 Emanuel Goncalves

import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from crispy import bipal_dbgd
from pybedtools import BedTool
from matplotlib.colors import rgb2hex
from sklearn.metrics import roc_curve, auc
from crispy.ratio import read_gff_dataframe


def import_brass_bedpe(bedpe_file, svs=None, bedpe_headers=None, sv_count_thres=1, bkdist=None, split_read=True, copynumber_flag=False):
    svs = ['deletion', 'tandem-duplication', 'inversion', 'translocation'] if svs is None else svs

    bedpe_headers = [
        'chr1', 'start1', 'end1', 'chr2', 'start2', 'end2', 'id/name', 'brass_score', 'strand1', 'strand2', 'sample', 'svclass',
        'bkdist', 'assembly_score', 'readpair names', 'readpair count', 'bal_trans', 'inv', 'occL', 'occH', 'copynumber_flag',
        'range_blat', 'Brass Notation', 'non-template', 'micro-homology', 'assembled readnames', 'assembled read count', 'gene1',
        'gene_id1', 'transcript_id1', 'strand1', 'end_phase1', 'region1', 'region_number1', 'total_region_count1', 'first/last1',
        'gene2', 'gene_id2', 'transcript_id2', 'strand2', 'phase2', 'region2', 'region_number2', 'total_region_count2',
        'first/last2', 'fusion_flag'
    ] if bedpe_headers is None else bedpe_headers

    bed = BedTool(bedpe_file).sort()
    bed_df = bed.to_dataframe(names=bedpe_headers)

    # Consider regions with only X SV
    bed_df = bed_df[bed.coverage(bed, counts=True).to_dataframe()[46] <= sv_count_thres]

    # Correct sample name
    bed_df['sample'] = bed_df['sample'].apply(lambda v: v.split(',')[0])

    # Filter SV annotations
    bed_df = bed_df[bed_df['svclass'].isin(svs)]

    # Discard entries with no gene mapped
    bed_df = bed_df[bed_df['gene_id1'] != '_']
    # bed_df = bed_df[bed_df['gene_id1'] == bed_df['gene_id2']]

    # Discard SV without split read
    if split_read:
        bed_df = bed_df.query("assembly_score != '_'")

    # Minimum bkdist
    if bkdist is not None:
        bed_df = bed_df.query('bkdist > {}'.format(bkdist))

    # Copy-number flag
    if copynumber_flag:
        bed_df = bed_df.query('copynumber_flag == 1')

    return bed_df


# - Import
# Gene details
ginfo = read_gff_dataframe()

# Ploidy
ploidy = pd.read_csv('data/crispy_copy_number_ploidy_wgs.csv', index_col=0, names=['sample', 'ploidy'])['ploidy']

# Gene copy-number ratio
cnv_ratios = pd.read_csv('data/crispy_copy_number_gene_ratio_wgs.csv', index_col=0)

# Copy-number
cnv = pd.read_csv('data/crispy_copy_number_gene_wgs.csv', index_col=0)

# BRASS results
bedpe_dir = 'data/gdsc/wgs/brass_bedpe/'
brass = pd.concat([import_brass_bedpe(bedpe_dir+f) for f in os.listdir(bedpe_dir)]).reset_index(drop=True)


# -
samples = list(set(cnv_ratios).intersection(brass['sample']))
print('Samples: {}'.format(len(samples)))


# -
# Filter
svcount = brass.groupby(['sample', 'gene_id1'])['svclass'].count().sort_values()
svcount = {(s, g) for s, g in svcount[svcount != 1].index}
brass_df = brass[[(s, g) not in svcount for s, g in brass[['sample', 'gene_id1']].values]]

brass_df = brass_df.groupby(['sample', 'gene_id1'])['svclass'].agg(lambda x: set(x)).to_dict()
brass_df = {k: list(brass_df[k])[0] for k in brass_df if len(brass_df[k]) == 1}

#
sv_ratio = pd.concat([
    cnv_ratios[samples].unstack().rename('ratio'),
    cnv[samples].unstack().rename('cnv'),
], axis=1)
sv_ratio = sv_ratio.reset_index().rename(columns={'level_0': 'sample', 'level_1': 'gene'})

sv_ratio = sv_ratio.assign(chr=ginfo.loc[sv_ratio['gene'], 'chr'].values)
sv_ratio = sv_ratio.assign(start=ginfo.loc[sv_ratio['gene'], 'start'].values)
sv_ratio = sv_ratio.assign(end=ginfo.loc[sv_ratio['gene'], 'end'].values)

sv_ratio['svclass'] = [brass_df[(s, g)] if (s, g) in brass_df else '-' for s, g in sv_ratio[['sample', 'gene']].values]
print(sv_ratio.query("svclass != '-'").sort_values('ratio'))


# -
order = ['deletion', 'translocation', 'tandem-duplication', 'inversion']

plot_df = sv_ratio[sv_ratio['ratio'].apply(np.isfinite)]

ax = plt.gca()
for t, c in zip(order, map(rgb2hex, sns.light_palette(bipal_dbgd[0], len(order) + 1)[1:])):
    fpr, tpr, _ = roc_curve((plot_df['svclass'] == t).astype(int), plot_df['ratio'])
    ax.plot(fpr, tpr, label='%s=%.2f (AUC)' % (t, auc(fpr, tpr)), lw=1., c=c)

ax.plot((0, 1), (0, 1), 'k--', lw=.3, alpha=.5)
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.set_xlabel('False positive rate')
ax.set_ylabel('True positive rate')
ax.set_title('Structural rearrangements relation with\ncopy-number ratio')

legend = ax.legend(loc=4, prop={'size': 6})

plt.gcf().set_size_inches(3, 3)
plt.savefig('reports/crispy/brass_sv_ratio_cumdist.png', bbox_inches='tight', dpi=600)
plt.close('all')

#
sns.boxplot('ratio', 'svclass', data=sv_ratio, orient='h', order=order, color=bipal_dbgd[0], notch=True, linewidth=.5, fliersize=1)
plt.title('Structural rearrangements relation with\ncopy-number ratio')
plt.xlabel('Copy-number ratio (rank)')
plt.ylabel('')
plt.gcf().set_size_inches(3, 1)
plt.savefig('reports/crispy/brass_sv_ratio_boxplot.png', bbox_inches='tight', dpi=600)
plt.close('all')
