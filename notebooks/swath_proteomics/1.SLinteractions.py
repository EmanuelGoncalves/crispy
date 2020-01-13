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
from swath_proteomics.LMModels import LMModels
from crispy.DataImporter import Proteomics, GeneExpression, CRISPR


LOG = logging.getLogger("Crispy")
DPATH = pkg_resources.resource_filename("crispy", "data")
RPATH = pkg_resources.resource_filename("notebooks", "swath_proteomics/reports/")


# SWATH-Proteomics
#

prot = Proteomics().filter(perc_measures=0.75).drop(columns=["HEK", "h002"])
LOG.info(f"Proteomics: {prot.shape}")


# # Gene-expression
# #
#
# gexp = GeneExpression().filter()
# LOG.info(f"Transcriptomics: {gexp.shape}")


# CRISPR
#

crispr = CRISPR().filter(abs_thres=0.5, min_events=3)
LOG.info(f"CRISPR: {crispr.shape}")


# Protein ~ CRISPR LMMs
#

prot_lmm = LMModels(y=prot.T.iloc[:, :15], x=crispr.T)
prot_lmm = prot_lmm.matrix_lmm()
prot_lmm.to_csv(f"{RPATH}/lmm_protein_crispr.csv.gz", index=False, compression="gzip")
# prot_lmm.write_lmm(output_folder=f"{RPATH}/lmm_protein_crispr/")

# # Associations Transcript
# res_gexp = lmm.matrix_limix_lmm(Y=y2[genes], X=x, M=m, K=k, x_feature_type="all")
# res_gexp.to_csv(f"{RPATH}/lmm_gexp_crispr.csv", index=False, compression="gzip")
#
