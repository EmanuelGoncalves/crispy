#!/usr/bin/env python
# Copyright (C) 2018 Emanuel Goncalves

import os
import argparse
import numpy as np
import pandas as pd
import scripts as mp
from bsub import bsub
from datetime import datetime as dt
from crispy.biases_correction import CRISPRCorrection
from scripts.plotting.annotate_bedpe import annotate_brass_bedpe

# BSUB options
BSUB_CORES = 16
BSUB_MEMORY = 32000
BSUB_QUEQUE = 'normal'

# WGS
SVTYPES = ['inversion', 'tandem-duplication', 'deletion']

FEATURES = {
    'all': ['cnv', 'tandemduplication', 'deletion', 'inversion'],
    'svs': ['tandemduplication', 'deletion', 'inversion'],
    'cnv': ['cnv']
}

BKDIST = 0
SPLITREADS = True


def count_svs(svs):
    counts = dict(zip(*(SVTYPES, np.repeat(0, len(SVTYPES)))))
    counts.update(dict(zip(*(np.unique(svs.split(','), return_counts=True)))))
    return counts


def present_svs(svs):
    svs_set = set(svs.split(','))
    return {sv.replace('-', ''): int(sv in svs_set) for sv in SVTYPES}


def run_correction(args):
    # - Imports
    # CRISPR lib
    sgrna_lib = pd.read_csv(args.crispr_lib_file, index_col=0)

    # Annotate BRASS bedpes
    bed_dfs = annotate_brass_bedpe(args.wgs_dir, bkdist=BKDIST, splitreads=SPLITREADS, samples=[args.sample])

    # CRISPR
    c_gdsc_fc = pd.read_csv(args.wgs_cnv_file, index_col=0).dropna()[args.sample]

    # Copy-number
    cnv = pd.read_csv(args.wgs_cnv_file, index_col=0)[args.sample]

    # - Overlap
    genes = set(c_gdsc_fc.index).intersection(cnv.index)

    # - Assemble matrix
    svmatrix = pd.DataFrame(bed_dfs.set_index('feature')['collapse'].apply(present_svs).to_dict()).T
    svmatrix = svmatrix.reindex(genes).replace(np.nan, 0).astype(int)

    svmatrix = svmatrix.assign(chr=sgrna_lib.groupby('gene')['chr'].first().reindex(svmatrix.index).values)
    svmatrix = svmatrix.assign(crispr=c_gdsc_fc.reindex(svmatrix.index).values)
    svmatrix = svmatrix.assign(cnv=cnv.reindex(svmatrix.index).values)

    svmatrix = svmatrix.dropna()

    # Correction
    df = CRISPRCorrection().rename(args.sample).fit_by(by=svmatrix['chr'], X=svmatrix[FEATURES[args.feature]], y=svmatrix['crispr'])

    df = pd.concat([v.to_dataframe() for k, v in df.items()])

    df.to_csv('{}/{}.{}.csv'.format(args.output_dir, args.sample, args.feature))


def run_correction_bsub(args):
    # Set job name
    jname = 'crispy_{}'.format(args.sample)

    # Define command
    j_cmd = \
        "python3.6.1 scripts/processing/correct_svs_bias.py " \
        "-crispr_file '{}' -crispr_lib_file '{}' -wgs_dir '{}' -wgs_cnv_file '{}' -feature '{}' -sample '{}' -output_folder '{}'" \
            .format(args.crispr_file, args.crispr_lib_file, args.wgs_dir, args.wgs_cnv_file, args.feature, args.sample, args.output_dir)

    # Create bsub
    j = bsub(
        jname, R='select[mem>{}] rusage[mem={}] span[hosts=1]'.format(BSUB_MEMORY, BSUB_MEMORY), M=BSUB_MEMORY,
        verbose=True, q=BSUB_QUEQUE, J=jname, o=jname, e=jname, n=BSUB_CORES,
    )

    # Submit
    j(j_cmd)
    print(jname, j_cmd)


if __name__ == '__main__':
    # Command line args parser
    parser = argparse.ArgumentParser(description='Run Crispy correction with WGS')

    parser.add_argument('-crispr_file', nargs='?', default=mp.CRISPR_GENE_FC)
    parser.add_argument('-crispr_lib_file', nargs='?', default=mp.LIBRARY)

    parser.add_argument('-wgs_dir', nargs='?', default=mp.WGS_BRASS_BEDPE)
    parser.add_argument('-wgs_cnv_file', nargs='?', default=mp.CN_GENE.format('wgs'))

    parser.add_argument('-feature', nargs='?')
    parser.add_argument('-sample', nargs='?')

    parser.add_argument('-output_dir', nargs='?', default=mp.CRISPY_WGS_OUTDIR)

    parser.add_argument('-bsub', action='store_true', default=False)

    parser.add_argument('-all', action='store_true', default=False)

    pargs = parser.parse_args()
    print(pargs)

    # - Correction
    # s, f = 'HCC1143', 'all'
    if pargs.all:
        print('[{}] Crispy: all'.format(dt.now().strftime('%Y-%m-%d %H:%M:%S')))

        crispr_samples = set(pd.read_csv(pargs.crispr_file, index_col=0).dropna())
        wgs_samples = set(annotate_brass_bedpe(pargs.wgs_dir, bkdist=BKDIST, splitreads=SPLITREADS)['sample'])

        samples = wgs_samples.intersection(crispr_samples)
        print('[INFO] Samples: {} CRISPR / {} WGS / {} Overlap'.format(len(crispr_samples), len(wgs_samples), len(samples)))

        for s in samples:
            for t in FEATURES:
                if pargs.bsub:
                    run_correction_bsub(pargs)
                else:
                    run_correction(pargs)

    elif pargs.bsub:
        print('[{}] Crispy: bsub {}'.format(dt.now().strftime('%Y-%m-%d %H:%M:%S'), pargs.sample))
        run_correction_bsub(pargs)

    else:
        print('[{}] {}: {}'.format(dt.now().strftime('%Y-%m-%d %H:%M:%S'), pargs.sample, pargs.output_dir))

        run_correction(pargs)
