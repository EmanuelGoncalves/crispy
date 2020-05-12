#!/usr/bin/env python
# Copyright (C) 2019 Emanuel Goncalves

import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()


version = {}
with open("crispy/__init__.py") as f:
    exec(f.read(), version)


with open("requirements.txt") as f:
    requirements = f.read().split("\n")


included_files = {
    "crispy": [
        "data/cytoBand.txt",
        "data/example_copynumber.csv",
        "data/example_rawcounts.csv",

        "data/crispr_libs/Avana_v1.csv.gz",
        "data/crispr_libs/Brunello_v1.csv.gz",
        "data/crispr_libs/GeCKO_v2.csv.gz",
        "data/crispr_libs/Manjunath_Wu_v1.csv.gz",
        "data/crispr_libs/MasterLib_v1.csv.gz",
        "data/crispr_libs/MinLibCas9.csv.gz",
        "data/crispr_libs/Sabatini_Lander_v3.csv.gz",
        "data/crispr_libs/TKOv3.csv.gz",
        "data/crispr_libs/Yusa_v1.1.csv.gz",

        "data/crispr_manifests/project_score_exclude_samples.csv",
        "data/crispr_manifests/GeCKO2_Achilles_v3.3.8_dropped_guides.csv.gz",
        "data/crispr_manifests/Avana_DepMap19Q2_sample_map.csv.gz",
        "data/crispr_manifests/Avana_DepMap19Q2_dropped_guides.csv.gz",
        "data/crispr_manifests/Avana_DepMap19Q3_sample_map.csv.gz",
        "data/crispr_manifests/Avana_DepMap19Q3_dropped_guides.csv",
        "data/crispr_manifests/KM12_coverage_samplesheet.xlsx",

        "data/gene_sets/EssentialGenes.csv",
        "data/gene_sets/NonessentialGenes.csv",
        "data/gene_sets/pancan_core.csv",
        "data/gene_sets/curated_BAGEL_essential.csv",
        "data/gene_sets/curated_BAGEL_nonEssential.csv",
        "data/gene_sets/pan_core_essential_broad_depmap18Q3.txt",
        "data/gene_sets/pan_core_essential_broad_depmap18Q4.txt",
        "data/gene_sets/essential_sanger_depmap19.tsv",

        "data/images/example_gp_fit.png",
        "data/images/logo.png",
        "data/model_list_20200204.csv",
        "data/GrowthRates_v1.3.0_20190222.csv",
        "data/SIDMvsMedia.xlsx",
        "data/crispr/CRISPR_Institute_Origin_20191108.csv.gz",
    ]
}


setuptools.setup(
    name="cy",
    version=version["__version__"],
    author="Emanuel Goncalves",
    author_email="eg14@sanger.ac.uk",
    long_description=long_description,
    description="Modelling CRISPR dropout data",
    long_description_content_type="text/markdown",
    url="https://github.com/EmanuelGoncalves/crispy",
    packages=["crispy", "test"],
    include_package_data=True,
    package_data=included_files,
    install_requires=requirements,
    classifiers=(
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: BSD License",
        "Operating System :: OS Independent",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "Topic :: Scientific/Engineering :: Information Analysis",
    ),
)
