## HT-29 CRISPR/Cas9 screen followed with dabrafenib treatment.

CRISPR/Cas9 sgRNA raw counts are analysed with the following pipeline:

. QC failed sgRNAs were removed 
	DHRSX_CCDS35195.1_ex1_X:2161152-2161175:+_3-1,
	DHRSX_CCDS35195.1_ex6_Y:2368915-2368938:+_3-3,
	DHRSX_CCDS35195.1_ex4_X:2326777-2326800:+_3-2,
	sgPOLR2K_1
. sgRNAs with low counts (<30) in the Plasmid_V1.1 (control) coniditon were excluded
. Similar to DESeq2 median-of-ratios method was used to correct by library size
. Counts were log2 transformed
. Fold-changes were calculated for each condition comparing to Plasmid_V1.1
. Limma was used to estimate statistically significant gene essentiality across the different time-points


# Files

HT29_dabraf_foldchanges_sgrna: sgRNA level fold-changes calculated versus the control (Plasmid_V1.1).

HT29_dabraf_foldchanges_gene.csv: Gene level fold-changes calculated by averaging matching sgRNAs fold-changes.

HT29_dabraf_foldchanges_gene_limma.csv: Gene level statistical comparisons of day 10, 14, 18 and 21 versus day 8 (e.g. day18 log2 fold-changes - day8 log2 fold-changes) obtained with Limma.
