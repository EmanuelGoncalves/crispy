# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.3'
#       jupytext_version: 1.0.0
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %matplotlib inline
# %autosave 0
# %load_ext autoreload
# %autoreload 2

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from crispy.CRISPRData import Library

#

lib = Library()

sgrnas = lib.load_library_sgrnas(verbose=1)


#

sgrnas["lib"].value_counts()


#

sgrnas.groupby(["lib", "len"])["sgRNA"].count().sort_values(ascending=False)


#

sgrnas_counts = sgrnas.groupby("sgRNA")["lib"].agg(set)
sgrnas_counts = sgrnas_counts.to_frame().assign(count=sgrnas_counts.apply(len))
sgrnas_counts["lib"] = [";".join(i) for i in sgrnas_counts["lib"]]
sgrnas_counts.to_csv(f"{lib.ddir}/Master_sgRNA_db.csv.gz", compression="gzip")


# Copyright (C) 2019 Emanuel Goncalves
