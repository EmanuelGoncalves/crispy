#!/usr/bin/env python
# Copyright (C) 2018 Emanuel Goncalves

import numpy as np
import pandas as pd
import seaborn as sns
import scripts.drug as dc
import matplotlib.pyplot as plt

if __name__ == '__main__':
    # CRISPR gene-level corrected fold-changes
    crispr = pd.read_csv(dc.CRISPR_GENE_FC_CORRECTED, index_col=0, sep='\t').dropna()
    crispr_scaled = dc.scale_crispr(crispr)
