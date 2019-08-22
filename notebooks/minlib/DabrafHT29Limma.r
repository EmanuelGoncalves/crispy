library(limma)

# Configurations
setwd("~/Projects/crispy/")

# Import samplesheet
samplesheet = read.csv(gzfile("crispy/data/crispr_manifests/HT29_Dabraf_samplesheet.csv.gz", "rt"), stringsAsFactors = F, check.names = F, row.names = 7)
samplesheet$id = paste(samplesheet$medium, samplesheet$time, sep="_")

# Import data
for (metric in c("All", "Minimal")) {
  message(c("Metric: ", metric))
  
  data <- read.csv(
    gzfile(
      paste(
        "notebooks/minlib/reports/HT29_Dabraf_gene_fc_", metric, ".csv.gz", sep=""), 
      "rt"
    ), 
    row.names = 1, 
    check.names = F, 
    stringsAsFactors = F
  )
  colnames(data) <- samplesheet[colnames(data), 'id']
  
  # Design matrix
  l <- c(
    "Initial_D8", 
    "Dabraf_D10", "Dabraf_D14", "Dabraf_D18", "Dabraf_D21", 
    "DMSO_D10", "DMSO_D14", "DMSO_D18", "DMSO_D21"
  )
  f <- factor(colnames(data), levels = l)
  design <- model.matrix(~0 + f)
  colnames(design) <- l
  
  # Linear fit: design matrix
  fit <- lmFit(data, design)
  
  # Contrast matrix
  contrasts.m <- makeContrasts(
    Dif10d=Dabraf_D10 - DMSO_D10,
    Dif14d=Dabraf_D14 - DMSO_D14,
    Dif18d=Dabraf_D18 - DMSO_D18,
    Dif21d=Dabraf_D21 - DMSO_D21,
    levels=design)
  
  # Linear fit: contrast matrix
  fit2 <- contrasts.fit(fit, contrasts.m)
  
  # Bayesian shrinkage
  fit2 <- eBayes(fit2)
  
  # Export results
  res <- topTableF(fit2, adjust="fdr", number=Inf)
  write.csv(res, paste("notebooks/minlib/reports/HT29_Dabraf_limma_", metric, ".csv", sep=""), quote = F)
}
