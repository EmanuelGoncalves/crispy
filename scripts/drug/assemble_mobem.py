#!/usr/bin/env python
# Copyright (C) 2018 Emanuel Goncalves

import pandas as pd
import scripts.drug as dc

MOBEM_PANCAN = './data/gdsc/mobems/PANCAN_simple_MOBEM.rdata.tsv'


def assemble_mobem(samplesheet, file=None, filter_hyper=True):
    file = MOBEM_PANCAN if file is None else file

    mobem = pd.read_csv(file, sep='\t', index_col=0)

    # Map COSMIC id to sample name
    mobem.columns = list(samplesheet.loc[list(map(int, mobem)), 'Cell Line Name'])

    # Discard hyper methylation features
    if filter_hyper:
        mobem = mobem.drop([i for i in mobem.index if "_HypMET" in i])

    return mobem


if __name__ == '__main__':
    # Samplesheet
    ss = pd.read_csv(dc.SAMPLESHEET_FILE, index_col=2).dropna(subset=['Cancer Type'])

    mobem = assemble_mobem(ss, MOBEM_PANCAN)
    mobem.to_csv(dc.MOBEM_FILE)
