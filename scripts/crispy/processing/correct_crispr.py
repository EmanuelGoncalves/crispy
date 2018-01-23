#!/usr/bin/env python
# Copyright (C) 2018 Emanuel Goncalves

import sys
import pandas as pd
from bsub import bsub
from datetime import datetime as dt
from crispy.biases_correction import CRISPRCorrection


BSUB_CORES = 16
BSUB_MEMORY = 32000
BSUB_QUEQUE = 'normal'


def correct_cnv(sample, crispr, cnv, lib):
    # Build data-frame
    df = pd.concat([crispr[sample].rename('fc'), cnv[sample].rename('cnv'), lib.rename('chr')], axis=1).dropna()

    # Correction
    crispy = CRISPRCorrection().rename(sample).fit_by(by=df['chr'], X=df[['cnv']], y=df['fc'])
    crispy = pd.concat([v.to_dataframe() for k, v in crispy.items()])

    # Export
    crispy.to_csv('data/crispy/crispr_gdsc_crispy_%s.csv' % sample)

    return crispy


def correct_cnv(crispr_file='data/crispr_gdsc_logfc.csv', cnv_file='data/crispy_gene_copy_number_snp.csv', lib_file='data/crispr_libs/KY_Library_v1.1_updated.csv'):
    # Import sgRNA library
    lib = pd.read_csv(lib_file, index_col=0).groupby('gene')['chr']

    # CRISPR fold-changes
    crispr = pd.read_csv(crispr_file, index_col=0)

    # Copy-number
    cnv = pd.read_csv(cnv_file, index_col=0)

    # Crispy
    for sample in set(crispr).intersection(cnv):
        print('[INFO] bsub option detected')
        print('[INFO] Crispy: {}'.format(sample))

        if len(sys.argv) > 1 and sys.argv[1] == 'bsub':
            jname = 'crispy_GDSC_{}'.format(sample)

            # Define command
            j_cmd = 'python3.6.1 scripts/crispy/processing/correct_crispr.py bsub {}'.format(sample)

            # Create bsub
            j = bsub(
                jname, R='select[mem>{}] rusage[mem={}] span[hosts=1]'.format(BSUB_MEMORY, BSUB_MEMORY), M=BSUB_MEMORY,
                verbose=True, q=BSUB_QUEQUE, J=jname, o=jname, e=jname, n=BSUB_CORES,
            )

            # Submit
            j(j_cmd)
            print('[INFO {}] bsub GDSC {}'.format(dt.now().strftime('%Y-%m-%d %H:%M:%S'), sample))

        else:
            # Bias correction
            crispy = correct_cnv(sample, crispr, cnv, lib)


def main():
    if len(sys.argv) > 2 and sys.argv[1] == 'bsub':
        print('[INFO] bsub option detected')


    else:
        correct_cnv()


if __name__ == '__main__':
    main()
