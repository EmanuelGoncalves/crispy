.. -*- mode: rst -*-


Crispy: analysis
============

# processing
calculate_fold_change.py: Generate gene-level fold-changes (averaging QC passed replicates).

calculate_cnv_ratio.py: Takes a copy-number segment file per sample and generates gene-level copy-number estimations, ratios, chromosome copies and sample ploidy.

correct_cnv_bias.py: Runs Crispy copy-number bias correction of CRISPR data (per sample/per chromossome).

# plotting
qc_bias_assessment.py: Capacity to recall a priori known essential genes, estimation of the copy-number bias and capacity to correct for those tested.