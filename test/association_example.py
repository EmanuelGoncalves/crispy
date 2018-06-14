#!/usr/bin/env python
# Copyright (C) 2018 Emanuel Goncalves

import pandas as pd
import crispy as cy

# Import data
data = cy.get_example_data()

# Association analysis
crispy = cy.CRISPRCorrection()\
    .fit_by(by=data['chr'], X=data[['cnv']], y=data['fc'])

# Export
crispy = pd.concat([v.to_dataframe() for k, v in crispy.items()])\
    .sort_values(['cnv', 'k_mean'], ascending=[False, True])

print(crispy)
