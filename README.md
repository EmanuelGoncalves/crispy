![Crispy logo](crispy/data/images/logo.png)

[![License](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause) [![PyPI version](https://badge.fury.io/py/cy.svg)](https://badge.fury.io/py/cy) [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.2530755.svg)](https://doi.org/10.5281/zenodo.2530755)


Module with utility functions to process CRISPR-based screens and method to correct gene independent copy-number effects.


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

Examples
--
Support to library imports:
```python
from crispy.CRISPRData import Library

# Master Library, standardised assembly of KosukeYusa V1.1, Avana, Brunello and TKOv3 
# CRISPR-Cas9 libraries.
master_lib = Library.load_library("MasterLib_v1.csv.gz")


# Genome-wide minimal CRISPR-Cas9 library. 
minimal_lib = Library.load_library("MinLibCas9.csv.gz")

# Some of the most broadly adopted CRISPR-Cas9 libraries:
# 'Avana_v1.csv.gz', 'Brunello_v1.csv.gz', 'GeCKO_v2.csv.gz', 'Manjunath_Wu_v1.csv.gz', 
# 'TKOv3.csv.gz', 'Yusa_v1.1.csv.gz'
brunello_lib = Library.load_library("Brunello_v1.csv.gz")
```

Select sgRNAs (across multiple CRISPR-Cas9 libraries) for a given gene:
```python
from crispy.GuideSelection import GuideSelection

# sgRNA selection class
gselection = GuideSelection()

# Select 5 optimal sgRNAs for MCL1 across multiple libraries 
gene_guides = gselection.select_sgrnas(
    "MCL1", n_guides=5, offtarget=[1, 0], jacks_thres=1, ruleset2_thres=.4
)

# Perform different rounds of sgRNA selection with increasingly relaxed efficiency thresholds 
gene_guides = gselection.selection_rounds("TRIM49", n_guides=5, do_amber_round=True, do_red_round=True)
```

Copy-number correction:
```python
import crispy as cy
import matplotlib.pyplot as plt
from CRISPRData import ReadCounts, Library

"""
Import sample data
"""
rawcounts, copynumber = cy.Utils.get_example_data()

"""
Import CRISPR-Cas9 library

Important:
      Library has to have the following columns: "Chr", "Start", "End", "Approved_Symbol"
      Library and segments have to have consistent "Chr" formating: "Chr1" or "chr1" or "1"
      Gurantee that "Start" and "End" columns are int
"""
lib = Library.load_library("Yusa_v1.1.csv.gz")

lib = lib.rename(
    columns=dict(start="Start", end="End", chr="Chr", Gene="Approved_Symbol")
).dropna(subset=["Chr", "Start", "End"])

lib["Chr"] = "chr" + lib["Chr"]

lib["Start"] = lib["Start"].astype(int)
lib["End"] = lib["End"].astype(int)

"""
Calculate fold-change
"""
plasmids = ["ERS717283"]
rawcounts = ReadCounts(rawcounts).remove_low_counts(plasmids)
sgrna_fc = rawcounts.norm_rpm().foldchange(plasmids)

"""
Correct CRISPR-Cas9 sgRNA fold changes
"""
crispy = cy.Crispy(
    sgrna_fc=sgrna_fc.mean(1), copy_number=copynumber, library=lib.loc[sgrna_fc.index]
)

# Fold-changes and correction integrated funciton.
# Output is a modified/expanded BED formated data-frame with sgRNA and segments information
#   n_sgrna: represents the minimum number of sgRNAs required per segment to consider in the fit.
#            Recomended default values range between 4-10.
bed_df = crispy.correct(n_sgrna=10)
print(bed_df.head())

# Gaussian Process Regression is stored
crispy.gpr.plot(x_feature="ratio", y_feature="fold_change")
plt.show()
```
![GPR](crispy/data/images/example_gp_fit.png)


Credits and License
--
Developed at the [Wellcome Sanger Institue](https://www.sanger.ac.uk/) (2017-2020).

For citation please refer to:

[Gonçalves E, Behan FM, Louzada S, Arnol D, Stronach EA, Yang F, Yusa K, Stegle O, Iorio F, Garnett MJ (2019) Structural 
rearrangements generate cell-specific, gene-independent CRISPR-Cas9 loss of fitness effects. Genome Biol 20: 27](https://doi.org/10.1186/s13059-019-1637-z)

[Gonçalves E, Thomas M, Behan FM, Picco G, Pacini C, Allen F, Parry-Smith D, Iorio F, Parts L, Yusa K, Garnett MJ (2019) 
Minimal genome-wide human CRISPR-Cas9 library. bioRxiv](https://www.biorxiv.org/content/10.1101/848895v1)
