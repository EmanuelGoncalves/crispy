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

for (metric in c("ks", "all")) {
  message(c("Metric: ", metric))
  
  # - Import CRISPR sgRNA fold-changes
  sgrna_fc_file = paste("notebooks/C50K/reports/KosukeYusa_v1.1_sgrna_fc_", metric, ".csv.gz", sep = "")
  sgrna_fc = read.csv(gzfile(sgrna_fc_file, "rt"), stringsAsFactors = F, row.names = 1, check.names = F)
  
  order = rownames(sgrna_fc)
  
  # - Copy-number correction
  for (sample in unlist(unique(samplemap["model_id"]))) {
    message(c("Sample: ", sample))
    
    # Export files
    sample_segments_file = paste(export_folder, sample, "_", metric, "_", "segments", ".csv", sep="")
    sample_fc_file = paste(export_folder, sample, "_", metric, "_", "corrected_fc", ".csv", sep="")
    
    if (file.exists(sample_fc_file)) next
    
    # Build data-frame
    sample_logFCs = data.frame(
      sgRNA=order, 
      GENES=KY_Library_v1.1[order, "GENES"], 
      sgrna_fc[order, samplemap[samplemap["model_id"] == sample, "s_lib"]],
      stringsAsFactors = FALSE,
      row.names = order
    )
    
    sample_sortedFCs = ccr.logFCs2chromPos(sample_logFCs[order,], KY_Library_v1.1[order,])
    
    sample_correctedFCs = ccr.GWclean(sample_sortedFCs, display = T, label = sample)
    
    # Export segment file
    write.table(sample_correctedFCs$segments, sample_segments_file, quote = F, sep = ",")
  
    # Export corrected fold-changes
    write.table(sample_correctedFCs$corrected_logFCs, sample_fc_file, quote = F, sep = ",")
  }
}