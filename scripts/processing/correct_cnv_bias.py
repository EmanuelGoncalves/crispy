#!/usr/bin/env python
# Copyright (C) 2018 Emanuel Goncalves

import os
import argparse
import pandas as pd
import scripts as mp
from bsub import bsub
from datetime import datetime as dt
from crispy.association import CRISPRCorrection


# BSUB options
BSUB_CORES = 16
BSUB_MEMORY = 32000
BSUB_QUEQUE = 'normal'


def run_correction(sample_name, crispr_file, crispr_lib_file, cnv_file, output_folder):
    # Import sgRNA library
    lib = pd.read_csv(crispr_lib_file, index_col=0).groupby('gene')['chr'].first()

    # CRISPR fold-changes
    crispr = pd.read_csv(crispr_file, index_col=0)

    # Copy-number
    cnv = pd.read_csv(cnv_file, index_col=0)

    assert sample_name in crispr.columns and sample_name in cnv.columns, 'Sample not in CRISPR and/or CNV matrices'

    # Build data-frame
    df = pd.concat([crispr[sample_name].rename('fc'), cnv[sample_name].rename('cnv'), lib.rename('chr')], axis=1).dropna()

    # Correction
    crispy = CRISPRCorrection().rename(sample_name).fit_by(by=df['chr'], X=df[['cnv']], y=df['fc'])
    crispy = pd.concat([v.to_dataframe() for k, v in crispy.items()])

    # Export
    crispy.to_csv('{}/crispy_crispr_{}.csv'.format(output_folder, sample_name))


def iterate_correction(crispr_file, crispr_lib_file, cnv_file, output_folder, bsub_flag):
    # CRISPR and Copy-number samples
    crispr, cnv = pd.read_csv(crispr_file, index_col=0), pd.read_csv(cnv_file, index_col=0)

    # Overlap
    overlap_samples = set(crispr).intersection(cnv)
    overlap_genes = set(crispr.index).intersection(cnv.index)

    assert len(overlap_samples) > 0, 'No samples (columns) overlap between CRISPR and Copy-number matrices'
    assert len(overlap_genes) > 0, 'No genes (rows) overlap between CRISPR and Copy-number matrices'

    # run correction for each cell line
    for sample in overlap_samples:
        if bsub_flag:
            print('[{}] Crispy: bsub {}'.format(dt.now().strftime('%Y-%m-%d %H:%M:%S'), sample))

            # Set job name
            jname = 'crispy_{}'.format(sample)

            # Define command
            j_cmd = "python3.6.1 scripts/processing/correct_cnv_bias.py -crispr_file '{}' -crispr_lib_file '{}' -cnv_file '{}' -output_folder '{}' -sample '{}'"\
                .format(crispr_file, crispr_lib_file, cnv_file, output_folder, sample)

            # Create bsub
            j = bsub(
                jname, R='select[mem>{}] rusage[mem={}] span[hosts=1]'.format(BSUB_MEMORY, BSUB_MEMORY), M=BSUB_MEMORY,
                verbose=True, q=BSUB_QUEQUE, J=jname, o=jname, e=jname, n=BSUB_CORES,
            )

            # Submit
            j(j_cmd)

        else:
            print('[{}] {}: {}'.format(dt.now().strftime('%Y-%m-%d %H:%M:%S'), sample, output_folder))
            run_correction(sample, crispr_file, crispr_lib_file, cnv_file, output_folder)


def assemble_matrix(output_folder, field='regressed_out'):
    crispy_fc = pd.DataFrame({
        os.path.splitext(i)[0].split('_')[-1]: pd.read_csv(output_folder + i, index_col=0)[field] for i in os.listdir(output_folder) if i.startswith('crispy_crispr_')
    })
    return crispy_fc


def main():
    # Command line args parser
    parser = argparse.ArgumentParser(description='Run Crispy correction')

    parser.add_argument('-crispr_file', nargs='?', default=mp.CRISPR_GENE_FC)
    parser.add_argument('-crispr_lib_file', nargs='?', default=mp.LIBRARY)
    parser.add_argument('-cnv_file', nargs='?', default=mp.CN_GENE.format('snp'))
    parser.add_argument('-output_folder', nargs='?', default=mp.CRISPY_OUTDIR)
    parser.add_argument('-sample', nargs='?')
    parser.add_argument('-bsub', action='store_true', default=False)

    args = parser.parse_args()

    print(args)

    if args.sample is not None:
        for sample in map(str.strip, args.sample.split(',')):
            run_correction(sample, args.crispr_file, args.crispr_lib_file, args.cnv_file, args.output_folder)
            print('[{}] Crispy {}'.format(dt.now().strftime('%Y-%m-%d %H:%M:%S'), sample))

    else:
        iterate_correction(args.crispr_file, args.crispr_lib_file, args.cnv_file, args.output_folder, args.bsub)
        print('[{}] Crispy'.format(dt.now().strftime('%Y-%m-%d %H:%M:%S')))


if __name__ == '__main__':
    main()
