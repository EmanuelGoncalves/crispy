![Crispy logo](crispy/data/images/logo.png)

[![License](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause) [![PyPI version](https://badge.fury.io/py/cy.svg)](https://badge.fury.io/py/cy) [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.2530755.svg)](https://doi.org/10.5281/zenodo.2530755)


Method to correct gene independent copy-number effects on CRISPR-Cas9 screens.


Description
--
Crispy uses [Sklearn](http://scikit-learn.org/stable/index.html) implementation of [Gaussian Process Regression](http://scikit-learn.org/stable/modules/generated/sklearn.gaussian_process.GaussianProcessRegressor.html#sklearn.gaussian_process.GaussianProcessRegressor), fitting each sample independently.

Install
--

Install [`pybedtools`](https://daler.github.io/pybedtools/main.html#quick-install-via-conda) and then install `Crispy`

```
conda install -c bioconda pybedtools

pip install cy
```

Example
--
```python
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
# Output is a modified/expanded BED formated data-frame with sgRNA and segments information
bed_df = crispy.correct(x_features='ratio', y_feature='fold_change')
print(bed_df.head())

# Gaussian Process Regression is stored
crispy.gpr.plot(x_feature='ratio', y_feature='fold_change')
plt.show()

```
![GPR](crispy/data/images/example_gp_fit.png)


Credits and License
--
Developed at the [Wellcome Sanger Institue](https://www.sanger.ac.uk/) (2017-2019).

For citation please refer to:

[Gonçalves E, Behan FM, Louzada S, Arnol D, Stronach EA, Yang F, Yusa K, Stegle O, Iorio F, Garnett MJ (2019) Structural 
rearrangements generate cell-specific, gene-independent CRISPR-Cas9 loss of fitness effects. Genome Biol 20: 27](https://doi.org/10.1186/s13059-019-1637-z)
