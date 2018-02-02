#!/usr/bin/env python
# Copyright (C) 2018 Emanuel Goncalves

import pandas as pd


OMNIPATH_DB = 'data/resources/omnipath/omnipath_pathways_pathways.tab'


def ppi_omnipath(db_file=None):
    db_file = OMNIPATH_DB if db_file is None else db_file

    # Import omnipath
    ppi = pd.read_table(db_file, sep='\t')

    # Consider non conflicting signed interactions
    ppi = ppi[ppi[['Stimulatory_A-B', 'Inhibitory_A-B', 'Stimulatory_B-A', 'Inhibitory_B-A']].count(1) == 1]

    # Convert data-frame to signed directed interactions
    ppi_a = pd.DataFrame([{
        'source': pa, 'target': pb, 'interaction': i
    } for i, c in [(1, 'Stimulatory_A-B'), (-1, 'Inhibitory_A-B')] for pa, pb, dbs in ppi[['GeneSymbol_A', 'GeneSymbol_B', c]].dropna().values])

    ppi_b = pd.DataFrame([{
        'source': pb, 'target': pa, 'interaction': i
    } for i, c in [(1, 'Stimulatory_B-A'), (-1, 'Inhibitory_B-A')] for pa, pb, dbs in ppi[['GeneSymbol_A', 'GeneSymbol_B', c]].dropna().values])

    ppi = pd.concat([ppi_a, ppi_b]).drop_duplicates()

    return ppi
