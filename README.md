Crispy
============

Identify associations between genomic alterations (e.g. structural variation, copy-number variation) and CRISPR-Cas9 knockout response.

Description
--
Crispy uses [Sklearn](http://scikit-learn.org/stable/index.html) implementation of [Gaussian Process Regression](http://scikit-learn.org/stable/modules/generated/sklearn.gaussian_process.GaussianProcessRegressor.html#sklearn.gaussian_process.GaussianProcessRegressor), fitting by default each chromosome of each sample independently.


[Tandem duplications lead to loss of fitness effects in CRISPR-Cas9 data](https://www.biorxiv.org/content/early/2018/05/25/325076)


Example
--
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
```


Install
--

```
python setup.py install
```

Enrichment module has Cython files, to compile run:

```
python crispy/enrichment/gsea_setup.py build_ext --inplace
```

Regression module has Cython files, to compile run:

```
python crispy/regression/linear_setup.py build_ext --inplace
```
