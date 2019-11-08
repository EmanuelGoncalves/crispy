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
        "data/crispr_libs/GeCKO_v1.csv.gz",
        "data/crispr_libs/GeCKO_v2.csv.gz",
        "data/crispr_libs/Manjunath_Wu_v1.csv.gz",
        "data/crispr_libs/MasterLib_v1.csv.gz",
        "data/crispr_libs/MinimalLib_top2_disconcordant.csv.gz",
        "data/crispr_libs/MinimalLib_top2.csv.gz",
        "data/crispr_libs/MinLibCas9.csv.gz",
        "data/crispr_libs/Sabatini_Lander_v1.csv.gz",
        "data/crispr_libs/Sabatini_Lander_v2.csv.gz",
        "data/crispr_libs/Sabatini_Lander_v3.csv.gz",
        "data/crispr_libs/TKOv1.csv.gz",
        "data/crispr_libs/TKOv3.csv.gz",
        "data/crispr_libs/Yusa_v1.1.csv.gz",
        "data/crispr_libs/Yusa_v1.csv.gz",

        "data/crispr_rawcounts/KM12_coverage.csv.gz",
        "data/crispr_rawcounts/Yusa_v1.1_organoids.csv.gz",
        "data/crispr_rawcounts/Yusa_v1.1_HT29_dabraf.csv.gz",

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
        "data/model_list_2018-09-28_1452.csv",
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
    packages=setuptools.find_packages(),
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
