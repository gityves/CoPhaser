library(Matrix)
library(ccRemover)

############################
# Load counts
############################

DATA_FOLDER <- "data/vasa_mtx/"
counts <- readMM(paste0(DATA_FOLDER, "counts.mtx"))

genes <- readLines(paste0(DATA_FOLDER, "genes.txt"))
cells <- readLines(paste0(DATA_FOLDER, "cells.txt"))

rownames(counts) <- genes
colnames(counts) <- cells

############################
# Library-size normalization
############################

libsize <- Matrix::colSums(counts)

expr <- t(
  t(counts) / libsize
) * 10000

############################
# log(1+x)
############################

expr <- log1p(expr)

############################
# Center genes
# (official example)
############################

expr_centered <- t(
  scale(
    t(as.matrix(expr)),
    center = TRUE,
    scale = FALSE
  )
)

############################
# Find cell-cycle genes
############################

gene_names <- rownames(expr_centered)

cell_cycle_gene_indices <- gene_indexer(
  gene_names,
  species = "mouse",   # change to "human" if needed
  name_type = "symbol"
)

if_cc <- rep(FALSE, nrow(expr_centered))
if_cc[cell_cycle_gene_indices] <- TRUE

############################
# Create input object
############################

dat <- list(
  x = expr_centered,
  if_cc = if_cc
)

############################
# Run ccRemover
############################

xhat <- ccRemover(
  dat,
  cutoff = 3,
  max_it = 4,
  nboot = 200,
  ntop = 10
)

############################
# Save output
############################

write.csv(
  xhat,
  file = paste0(DATA_FOLDER, "ccremover_corrected.csv")
)