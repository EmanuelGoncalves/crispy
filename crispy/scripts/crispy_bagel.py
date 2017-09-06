#!/usr/bin/env python
# Copyright (C) 2017 Emanuel Goncalves

import os
import numpy as np
import pandas as pd
import scipy.stats as st
from random import choices
from datetime import datetime as dt


def fold_changes(data_dir, files_list=None, control=None, min_reads=30, sep="\t", file_extension=".read_count.tsv"):
	# Initialise fold-change dictionary
	for f in (os.listdir(data_dir) if files_list is None else files_list):
		if f.endswith(file_extension):
			# Import
			raw_counts_f = pd.read_csv("%s/%s" % (data_dir, f), sep=sep)

			# If control is None last column is used as control
			ctrl = raw_counts_f.columns[-1:][0] if control is None else control
			print("[%s] %s control: %s" % (dt.now().strftime("%Y-%m-%d %H:%M:%S"), f, ctrl))

			# Fill missing gene annotation and read count
			raw_counts_f["gene"] = raw_counts_f["gene"].fillna("NO_GENE_NAME")
			raw_counts_f = raw_counts_f.fillna(0)
			raw_counts_f = raw_counts_f.set_index(["sgRNA", "gene"])

			# Normalise each sample to a fixed total read count and discard low counts
			sum_reads = raw_counts_f.select_dtypes(include=['integer']).sum()

			raw_counts_f = raw_counts_f[raw_counts_f[ctrl] >= min_reads]
			raw_counts_f[list(sum_reads.index)] = raw_counts_f[sum_reads.index].divide(sum_reads) * 10000000

			# Calculate fold-changes
			counts_columns = list(raw_counts_f.select_dtypes(include=['floating']))
			raw_counts_f[counts_columns] = np.log2((raw_counts_f[counts_columns] + 0.5).divide((raw_counts_f[ctrl] + 0.5), axis=0))

			# Drop control
			raw_counts_f = raw_counts_f.iloc[:, 0].reset_index()

			# Export
			raw_counts_f.to_csv("%s/%s.foldchange.tsv" % (data_dir, os.path.splitext(f)[0]), sep='\t', index=False)

			print("[%s] %s fold-changes: done" % (dt.now().strftime("%Y-%m-%d %H:%M:%S"), f))

	print("[%s] Fold-changes finished")

def bagel_scores(fcs_file, subset=None, num_bootstraps=1000, fc_thres=2**-7, range_min=-10, range_max=2, ess_ref="./resources/curated_BAGEL_essential.csv", ness_ref="./resources/curated_BAGEL_nonEssential.csv", sep="\t"):
	# Reference sets
	genes_ess, genes_non_ess = set(pd.read_csv(ess_ref)["gene"]), set(pd.read_csv(ness_ref)["gene"])

	# Import fold-changes
	raw_counts_f = pd.read_csv(fcs_file, sep=sep).set_index("gene").drop("sgRNA", axis=1)

	# Sub-set samples
	if subset is not None:
		raw_counts_f = raw_counts_f[subset]

	# List genes
	genes = set(raw_counts_f.index)
	print("[%s] Unique genes: %d" % (dt.now().strftime("%Y-%m-%d %H:%M:%S"), len(genes)))

	# Bagel factor estimation
	bg = pd.DataFrame()
	for iteration in range(num_bootstraps):
		print("[%s] Iteration: %d" % (dt.now().strftime("%Y-%m-%d %H:%M:%S"), iteration + 1))

		# Randomly define train and test gene sets
		genes_train = set(choices(list(genes), k=len(genes)))
		genes_test = genes.difference(genes_train)

		# Calculate empirical essential and non-essential fold-change distributions
		gk_ess = st.gaussian_kde(raw_counts_f.loc[genes_train.intersection(genes_ess)].values.ravel())
		gk_non_ess = st.gaussian_kde(raw_counts_f.loc[genes_train.intersection(genes_non_ess)].values.ravel())

		# Define empirical upper and lower bounds within which to calculate BF = f(fold change)
		x = np.arange(range_min, range_max, 0.01)

		# Define the lower bound empirical fold change threshold: mininum FC where gk_non_ess is above the threshold
		x_non_ess_fit = gk_non_ess.evaluate(x)
		x_min = round(min(x[x_non_ess_fit > fc_thres]), 2)

		# Define upper bound empirical fold change threshold:  minimum value of log2(ess/non)
		x = np.arange(x_min, max(x[x_non_ess_fit > fc_thres]), 0.01)
		logratio_sample = np.log2(gk_ess.evaluate(x) / gk_non_ess.evaluate(x))
		x_max = round(x[logratio_sample.argmin()], 2)
		print("[%s] X min = %.2f; X max = %.2f" % (dt.now().strftime("%Y-%m-%d %H:%M:%S"), x_min, x_max))

		# Precalculate logratios and build lookup table (for speed)
		x = np.arange(x_min, x_max + 0.01, 0.01)
		logratio_lookup = pd.Series(dict(zip(*(np.round(x, 2), np.log2(gk_ess.evaluate(x) / gk_non_ess.evaluate(x))))))

		# Calculate BFs from lookup table for withheld test set
		df = raw_counts_f.loc[genes_test].copy().round(2).clip(x_min, x_max).unstack().reset_index().rename(columns={"level_0": "sample", 0: "bg"})
		df["bg"] = logratio_lookup[df["bg"]].values

		bg = bg.append(df.groupby("gene")["bg"].sum().reset_index())

	# Build results data-frame
	res = bg.groupby("gene")["bg"].agg([np.mean, np.std])

	return res
