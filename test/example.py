#!/usr/bin/env python
# Copyright (C) 2018 Emanuel Goncalves

import crispy as cy
import matplotlib.pyplot as plt

# Import data
rawcounts, copynumber = cy.Utils.get_example_data()

# Import CRISPR-Cas9 library
lib = cy.Utils.get_crispr_lib()

# Instantiate Crispy
crispy = cy.Crispy(
    raw_counts=rawcounts, copy_number=copynumber, library=lib
)

# Fold-changes and correction integrated funciton.
# Output is a modified/expanded BED output with sgRNA and segments information
bed_df = crispy.correct(x_features='ratio', y_feature='fold_change')
print(bed_df.head())

# Gaussian Process Regression is stored
crispy.gpr.plot(x_feature='ratio', y_feature='fold_change')
plt.show()
