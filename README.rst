.. -*- mode: rst -*-


Crispy
============

Use Crispy to identify associations between genomic alterations (e.g. structural variation, copy-number alterations) and CRISPR-Cas9 knockout response.

```python
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

```

To install run:

```
python setup.py install
```

Enrichment module has Cython files, to compile run:

```
python gsea_setup.py build_ext --inplace
```

Regression module has Cython files, to compile run:

```
python linear_setup.py build_ext --inplace
```
