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

import os
import sys
import logging
import argparse
import pandas as pd
import pkg_resources
from natsort import natsorted
from itertools import zip_longest
from swath_proteomics.LMModels import LMModels
from crispy.DataImporter import Proteomics, GeneExpression, CRISPR

sys.path.extend(
    [
        "/Users/eg14/Projects/crispy",
        "/Users/eg14/Projects/crispy/crispy",
        "/Users/eg14/Projects/crispy/notebooks",
    ]
)

LOG = logging.getLogger("Crispy")
DPATH = pkg_resources.resource_filename("crispy", "data")
RPATH = pkg_resources.resource_filename("notebooks", "swath_proteomics/reports/")


if __name__ == "__main__":
    # Args
    #
    parser = argparse.ArgumentParser(description="Run LMM Protein/GExp ~ CRISPR")
    parser.add_argument("--genes", nargs="+", default="None")
    parser.add_argument("--mode", nargs="?", default="None")
    parser.add_argument("--port", nargs="?", default="None")
    args = parser.parse_args()

    # Data-sets
    #
    prot, gexp, crispr = Proteomics(), GeneExpression(), CRISPR()

    # Samples
    #
    samples = set.intersection(
        set(prot.get_data()), set(gexp.get_data()), set(crispr.get_data())
    )
    LOG.info(f"Samples: {len(samples)}")

    # Filter data-sets
    #
    prot = prot.filter(subset=samples, perc_measures=0.75)
    LOG.info(f"Proteomics: {prot.shape}")

    gexp = gexp.filter(subset=samples)
    LOG.info(f"Transcriptomics: {gexp.shape}")

    crispr = crispr.filter(subset=samples, abs_thres=0.5, min_events=5)
    LOG.info(f"CRISPR: {crispr.shape}")

    # Genes
    #
    genes = natsorted(list(set.intersection(set(prot.index), set(gexp.index))))
    LOG.info(f"Genes: {len(genes)}")

    if args.genes != "None":
        LOG.info(f"Args genes: {len(args.genes)}")

        # Protein ~ CRISPR LMMs
        #
        prot_lmm = LMModels(y=prot.T[args.genes], x=crispr.T).matrix_lmm()
        prot_lmm.to_csv(
            f"{RPATH}/lmm_protein_crispr/{'_'.join(args.genes)}.csv.gz",
            index=False,
            compression="gzip",
        )

        # Gene-expression ~ CRISPR LMMs
        #
        gexp_lmm = LMModels(y=gexp.T[args.genes], x=crispr.T).matrix_lmm()
        gexp_lmm.to_csv(
            f"{RPATH}/lmm_gexp_crispr/{'_'.join(args.genes)}.csv.gz",
            index=False,
            compression="gzip",
        )

    elif args.gene == "Assemble":
        LOG.info(f"Assembling files")

        # Assemble protein
        #
        ddir = f"{RPATH}/lmm_protein_crispr/"
        prot_table = pd.concat(
            [pd.read_csv(f"{ddir}/{f}") for f in os.listdir(ddir)],
            ignore_index=True,
            sort=False,
        ).sort_values("fdr")[LMModels.RES_ORDER]
        prot_table.to_csv(
            f"{RPATH}/lmm_protein_crispr.csv.gz", index=False, compression="gzip"
        )

        # Assemble gexp
        #
        ddir = f"{RPATH}/lmm_gexp_crispr/"
        gexp_table = pd.concat(
            [pd.read_csv(f"{ddir}/{f}") for f in os.listdir(ddir)],
            ignore_index=True,
            sort=False,
        )
        gexp_table.to_csv(
            f"{RPATH}/lmm_gexp_crispr.csv.gz", index=False, compression="gzip"
        )

    else:
        LOG.info(f"Master mode called")

        def grouper(iterable, n, fillvalue=None):
            args = [iter(iterable)] * n
            return zip_longest(*args, fillvalue=fillvalue)

        for args_genes in grouper(genes, 30, None):
            args_genes = [g for g in args_genes if g is not None]
            os.system(
                f"python /Users/eg14/Projects/crispy/notebooks/swath_proteomics/1.SLinteractions.py --genes {' '.join(args_genes)}"
            )
