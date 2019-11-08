Minimal human genome-wide CRISPR-Cas9 library
=
CRISPR-Cas9 genome-wide minimal library for human cells. 

Install
--

```
pip install cy
```

Examples
--
```python
from crispy.CRISPRData import Library

# Master Library, standardised assembly of KosukeYusa V1.1, Avana, Brunello and TKOv3 CRISPR-Cas9 libraries.
#
master_lib = Library.load_library("MasterLib_v1.csv.gz")


# Genome-wide minimal CRISPR-Cas9 library. 
#
minimal_lib = Library.load_library("MinLibCas9.csv.gz")

# List all supported libraries. 
#
import pkg_resources, os
minimal_lib = Library.load_library("MinLibCas9.csv.gz")
print(os.listdir(pkg_resources.resource_filename("crispy", "data/crispr_libs/")))
```


### Credits and License
[![License](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)

Developed at the [Wellcome Sanger Institue](https://www.sanger.ac.uk/) (2017-2019).
