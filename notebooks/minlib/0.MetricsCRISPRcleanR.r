setwd("~/Projects/crispy/")

export_folder = "notebooks/C50K/reports/KosukeYusa_v1.1_crisprcleanr/"

# - Install DNAcopy package
if (!requireNamespace("BiocManager", quietly = TRUE))
  install.packages("BiocManager")
BiocManager::install("DNAcopy")
library(DNAcopy)

# - Install CRISPRcleanR
install.packages("devtools")
library(devtools)

install_github("francescojm/CRISPRcleanR")
library(CRISPRcleanR)

# - Import CRISPR library
KY_Library_v1.1 = read.csv("crispy/data/crispr_libs/Yusa_v1.1_CRISPRcleanR.csv", row.names = 1, stringsAsFactors = F, check.names = F)

# - Import Project Score V1.1 sample map
samplemap = read.csv(gzfile("crispy/data/crispr_manifests/project_score_sample_map_KosukeYusa_v1.1.csv.gz", "rt"), stringsAsFactors = F, check.names = F)

for (metric in c("ks_control_min", "all")) {
  message(c("Metric: ", metric))
  
  for (n_guides in c(2, 3)) {
    message(c("N guides: ", n_guides))
    
    # - Import CRISPR sgRNA fold-changes
    if (metric == "all") {
      sgrna_fc_file = "notebooks/C50K/reports/KosukeYusa_v1.1_sgrna_counts_all.csv.gz"
      
      if (n_guides != 2) next
      
    } else {
      sgrna_fc_file = paste("notebooks/C50K/reports/KosukeYusa_v1.1_sgrna_counts_", metric, "_top", n_guides, ".csv.gz", sep = "")
    
    }
    
    counts = read.csv(gzfile(sgrna_fc_file, "rt"), stringsAsFactors = F, row.names = 1, check.names = F)
    
    sgrnas = rownames(counts)
    
    # - Copy-number correction
    for (sample in unlist(unique(samplemap["model_id"]))) {
      message(c("Sample: ", sample))
      
      # Export files
      sample_segments_file = paste(export_folder, sample, "_", metric, "_", "top", n_guides, "_", "segments", ".csv", sep="")
      sample_fc_file = paste(export_folder, sample, "_", metric, "_", "top", n_guides, "_", "corrected_fc", ".csv", sep="")
      
      if (file.exists(sample_fc_file)) next
      
      # Build count data-frame
      sample_counts = data.frame(
        sgRNA=sgrnas,
        gene=KY_Library_v1.1[sgrnas, "GENES"],
        plasmid=counts["CRISPR_C6596666.sample"],
        counts[sgrnas, samplemap[samplemap["model_id"] == sample, "s_lib"]],
        stringsAsFactors = FALSE,
        row.names = sgrnas
      )
      
      # Calculate log fold-changes
      sample_logFCs = ccr.NormfoldChanges(
        Dframe=sample_counts, min_reads = 30, EXPname=sample, libraryAnnotation=KY_Library_v1.1, outdir="None"
      )
      
      # Sorted fold-changes
      sample_sortedFCs = ccr.logFCs2chromPos(sample_logFCs$logFCs[sgrnas,], KY_Library_v1.1[sgrnas,])
      
      # Corrected fold-changes
      sample_correctedFCs = ccr.GWclean(sample_sortedFCs, display = T, label = sample)
      
      # Export segment file
      write.table(sample_correctedFCs$segments, sample_segments_file, quote = F, sep = ",")
      
      # Export corrected fold-changes
      write.table(sample_correctedFCs$corrected_logFCs, sample_fc_file, quote = F, sep = ",")
    }
  }
}