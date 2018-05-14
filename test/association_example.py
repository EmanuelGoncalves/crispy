#!/usr/bin/env python
# Copyright (C) 2018 Emanuel Goncalves

import pandas as pd
from crispy.association import CRISPRCorrection


# Import data
data = pd.read_csv('extdata/association_example_data.csv', index_col=0)

# Association analysis
crispy = CRISPRCorrection()\
    .fit_by(by=data['chr'], X=data[['cnv']], y=data['fc'])

# Export
crispy = pd.concat([v.to_dataframe() for k, v in crispy.items()])\
    .sort_values(['cnv', 'k_mean'], ascending=[False, True])

print(crispy)
