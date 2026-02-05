library(Matrix)
library(SingleCellExperiment)
library(tricycle)

# --- Load data ---
mat <- readMM(file.path(DATA_FOLDER, "counts.mtx"))
genes <- read.table(file.path(DATA_FOLDER, "genes_id.tsv"), stringsAsFactors = FALSE)[,1]
cells <- read.table(file.path(DATA_FOLDER, "cells.tsv"), stringsAsFactors = FALSE)[,1]

colnames(mat) <- genes
rownames(mat) <- cells

# Metadata containing Cell_ID and Celltype
meta <- read.csv(file.path(DATA_FOLDER, "metadata.csv"), stringsAsFactors = FALSE)

# Reorder metadata to match the matrix rows
meta <- meta[match(cells, meta$Cell_ID), ]

# Unique cell types
celltypes <- unique(meta$Celltype)

# Collect all results here
all_results <- list()

for (ct in celltypes) {
  message("Processing celltype: ", ct)
  
  # Select cells of this cell type
  idx <- which(meta$Celltype == ct)
  if (length(idx) == 0) next
  
  mat_sub <- mat[idx, ]
  
  # Create SCE object for this cell type
  sce <- SingleCellExperiment(assays = list(logcounts = t(mat_sub)))
  
  # Run tricycle
  sce <- project_cycle_space(sce, species = "mouse")
  sce <- estimate_cycle_position(sce, species = "mouse")
  
  # Extract results
  res <- as.data.frame(sce@int_colData$reducedDims)
  res$Cell_ID <- rownames(mat_sub)
  res$Celltype <- ct
  
  # Save in list
  all_results[[ct]] <- res
}

# --- Merge results into a single table ---
merged_results <- do.call(rbind, all_results)

# --- Reorder to match metadata.csv order (Cell_ID column) ---
merged_results <- merged_results[match(meta$Cell_ID, merged_results$Cell_ID), ]

# Save
write.csv(
  merged_results,
  file.path(DATA_FOLDER, "tricycle_results.csv"),
  row.names = FALSE
)
