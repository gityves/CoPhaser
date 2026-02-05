library(Matrix)
library(Seurat)
library(dplyr)
DATA_FOLDER <- "C:/Users/yvesp/Desktop/PhD/data/VASA/vasa_R"

# --- Load data ---
mat <- readMM(file.path(DATA_FOLDER, "counts.mtx"))
genes <- read.table(file.path(DATA_FOLDER, "genes_id.tsv"), stringsAsFactors = FALSE)[,1]
cells <- read.table(file.path(DATA_FOLDER, "cells.tsv"), stringsAsFactors = FALSE)[,1]

# Clean gene names
genes <- tolower(genes)

# Remove invalid gene names
valid <- !(is.na(genes) | genes == "" | genes == " " | startsWith(genes, "?"))
genes <- genes[valid]
mat <- mat[, valid]

# Ensure uniqueness
genes <- make.unique(genes)
colnames(mat) <- genes

rownames(mat) <- cells


# Seurat expects genes × cells
mat_seu <- t(mat)
colnames(mat_seu) <- meta$Cell_ID

# Create Seurat object
seu <- CreateSeuratObject(counts = mat_seu)

# Normalize + scale
seu <- NormalizeData(seu)
seu <- ScaleData(seu)

# Load Seurat cell-cycle genes (mouse)
s.genes  <- tolower(cc.genes.updated.2019$s.genes)
g2m.genes <- tolower(cc.genes.updated.2019$g2m.genes)

# Score all cells
seu <- CellCycleScoring(
  seu,
  s.features = s.genes,
  g2m.features = g2m.genes,
  set.ident = FALSE
)

# Extract results
results <- data.frame(
  Cell_ID   = colnames(seu),
  S_score   = seu$S.Score,
  G2M_score = seu$G2M.Score,
  Phase     = seu$Phase,
  Celltype  = meta$Celltype
)

# Reorder to match metadata order
results <- results[match(meta$Cell_ID, results$Cell_ID), ]

# Save
write.csv(
  results,
  file.path(DATA_FOLDER, "seurat_cellcycle_scores_all_cells.csv"),
  row.names = FALSE
)
