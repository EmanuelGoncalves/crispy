#!/usr/bin/env python
# Copyright (C) 2018 Emanuel Goncalves

from pandas import Series, read_csv

# Samplesheet
samplesheet = read_csv('./data/gdsc/samplesheet.csv', index_col='COSMIC ID').dropna(subset=['Cancer Type'])

# - Import pancan MOBEM
mobem = read_csv('./data/gdsc/mobems/PANCAN_simple_MOBEM.rdata.tsv', sep='\t', index_col=0)

# - Map COSMIC id to sample name
mobem.columns = list(samplesheet.loc[Series(list(mobem)).astype(int), 'Cell Line Name'])

# # - Discard hyper methylation features
# mobem = mobem.drop([i for i in mobem.index if "_HypMET" in i])

# - Export
mobem.to_csv('./data/PANCAN_simple_MOBEM.rdata.annotated.all.csv')
