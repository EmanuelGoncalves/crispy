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


import logging
import pkg_resources
from natsort import natsorted
from swath_proteomics.LMModels import LMModels
from crispy.DataImporter import Proteomics, GeneExpression, CRISPR


LOG = logging.getLogger("Crispy")
DPATH = pkg_resources.resource_filename("crispy", "data")
RPATH = pkg_resources.resource_filename("notebooks", "swath_proteomics/reports/")


# SWATH-Proteomics
#

prot = Proteomics().filter(perc_measures=0.75).drop(columns=["HEK", "h002"])
LOG.info(f"Proteomics: {prot.shape}")


# Gene-expression
#

gexp = GeneExpression().filter()
LOG.info(f"Transcriptomics: {gexp.shape}")


# CRISPR
#

crispr = CRISPR().filter(abs_thres=0.5, min_events=3)
LOG.info(f"CRISPR: {crispr.shape}")


# Overlaps
#

samples = list(set.intersection(set(prot), set(gexp), set(crispr)))
genes = natsorted(list(set.intersection(set(prot.index), set(gexp.index))))
LOG.info(f"Genes: {len(genes)}; Samples: {len(samples)}")


# Protein ~ CRISPR LMMs
#

res_prot = LMModels.write_limix_lmm(
    Y=prot.T[genes[:15]],
    X=crispr.T,
    M=LMModels.define_covariates(add_institute=True),
    K=LMModels.kinship(crispr.T),
    output_folder=f"{RPATH}/lmm_protein_crispr/",
)
# res_prot.to_csv(f"{RPATH}/lmm_protein_crispr.csv", index=False, compression="gzip")

# # Associations Transcript
# res_gexp = lmm.matrix_limix_lmm(Y=y2[genes], X=x, M=m, K=k, x_feature_type="all")
# res_gexp.to_csv(f"{RPATH}/lmm_gexp_crispr.csv", index=False, compression="gzip")
#
