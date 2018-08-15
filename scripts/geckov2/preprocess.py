#!/usr/bin/env python
# Copyright (C) 2018 Emanuel Goncalves

import numpy as np
import pandas as pd
import geckov2 as gecko
from crispy import Crispy


def replicates_correlation():
    rep_fold_changes = pd.concat([
        Crispy(
            raw_counts=raw_counts[manifest[sample] + [gecko.PLASMID]],
            copy_number=copy_number.query(f"Sample == '{sample}'"),
            library=lib_guides,
            plasmid=gecko.PLASMID
        ).replicates_foldchanges() for sample in manifest
    ], axis=1)

    corr = rep_fold_changes.corr()

    corr = corr.where(np.triu(np.ones(corr.shape), 1).astype(np.bool))
    corr = corr.unstack().dropna().reset_index()
    corr = corr.set_axis(['sample_1', 'sample_2', 'corr'], axis=1, inplace=False)
    corr['replicate'] = [int(s1.split(' ')[0] == s2.split(' ')[0]) for s1, s2 in corr[['sample_1', 'sample_2']].values]

    corr.sort_values('corr').to_csv(f'{gecko.DIR}/replicates_foldchange_correlation.csv', index=False)

    return corr.query('replicate == 0')['corr'].mean()


if __name__ == '__main__':
    # - Imports
    # Samplesheet
    samplesheet = gecko.get_samplesheet()

    # Data-sets
    raw_counts = gecko.get_raw_counts()
    copy_number = gecko.get_copy_number_segments()

    # GeCKOv2 library
    lib = gecko.get_crispr_library()
    lib_guides = lib.groupby(['chr', 'start', 'end', 'sgrna'])['gene'].agg(lambda v: ';'.join(set(v))).rename('gene').reset_index()

    # - Sample manifest
    manifest = {
        v: [c for c in raw_counts if c.split(' ')[0] == i] for i, v in samplesheet['name'].iteritems()
    }

    # - Correction
    corr_threshold = replicates_correlation()
    print(f'[INFO] Correlation fold-change thredhold = {corr_threshold:.2f}')

    for s in manifest:
        print(f'[INFO] Sample: {s}')

        s_crispy = Crispy(
            raw_counts=raw_counts[manifest[s] + [gecko.PLASMID]],
            copy_number=copy_number.query(f"Sample == '{s}'"),
            library=lib_guides,
            plasmid=gecko.PLASMID,
            qc_replicates_thres=corr_threshold
        )

        bed_df = s_crispy.correct()

        if bed_df is not None:
            bed_df.to_csv(f'{gecko.DIR}/bed/{s}.crispy.bed', index=False, sep='\t')
