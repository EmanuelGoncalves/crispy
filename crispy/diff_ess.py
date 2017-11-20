#!/usr/bin/env python
# Copyright (C) 2017 Emanuel Goncalves

import patsy
import numpy as np
import pymc3 as pm
import pandas as pd
import scipy.stats as st
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Import metadata
colData = pd.read_csv('/Users/eg14/Downloads/GSE37704_metadata.csv', index_col=0)

# Import countdata
countData = pd.read_csv('/Users/eg14/Downloads/GSE37704_featurecounts.csv', index_col=0, sep=',').drop('length', axis=1)

# Filter low counts
countData = countData.loc[countData.sum(1) > 1]

# Select
countData = countData.loc[[
    'ENSG00000117519', 'ENSG00000183508', 'ENSG00000159176', 'ENSG00000150938', 'ENSG00000116016',
    'ENSG00000166507', 'ENSG00000175279', 'ENSG00000133997', 'ENSG00000140600', 'ENSG00000160948'
]]

# Design
design = patsy.dmatrix('~condition', colData, return_type='dataframe')
# design = sm.add_constant(pd.get_dummies(colData))

# -
g = 'ENSG00000117519'

# Statsmodels
glm = smf.GLM(countData.loc[g], design, family=sm.families.NegativeBinomial()).fit()
print(glm.summary())

# PyMC3
with pm.Model() as model:
    pm.glm.GLM(y=countData.loc[g], x=design, family=pm.glm.families.NegativeBinomial(), intercept=False)

    trace = pm.sample(2000, njobs=2)

    print(pm.df_summary(trace[1000:]))
    print(countData.loc[g].mean())
