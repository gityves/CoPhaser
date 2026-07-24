import numpy as np
from CoPhaser import utils
from CoPhaser.trainer import Trainer
from CoPhaser.loss import Loss
from CoPhaser import plotting
from CoPhaser.model import CoPhaser
from CoPhaser import gene_sets

import matplotlib.pyplot as plt
import seaborn as sns

import anndata
import scanpy as sc

import pandas as pd
import tqdm

from itertools import product

# adata = anndata.read_h5ad("../data/cellcycle_maxine/RPE_37C_Rep1_full.h5ad")
DATA_FOLDER = "data/"
adata = anndata.read_h5ad(f"{DATA_FOLDER}VASA_gt_phase.h5ad")
adata
# context genes
g = utils.get_variable_genes(adata)
len(g)
SMALL_CYCLING_GENE_SET = gene_sets.SMALL_CELL_CYCLE_GENE_SET

results = {
    "log_transformed": [],
    "divide_by_library_size": [],
    "scale_by_1e4": [],
    "scale_rhythmic_genes": [],
    "jensen_shannon": [],
}
# for l_transform, divide_by_lib, scale_by_1e4, _ in tqdm.tqdm(
#     product([True, False], [True, False], [True, False], range(5)), total=45
# ):
#     model = CoPhaser(
#         SMALL_CYCLING_GENE_SET,
#         g,
#         apply_scale_rhythmic_genes=False,
#         apply_log_transform=l_transform,
#         divide_by_library_size=divide_by_lib,
#         scale_by_1e4=scale_by_1e4,
#     )
#     model.load_anndata(adata, layer_to_use="spliced")
#     trainer = Trainer(
#         model,
#         Loss.compute_loss,
#         entropy_weight_factor=50,
#         MI_weight=50,
#         closed_circle_weight=10,
#     )
#     trainer.train_model(batch_size=2048, silent=True)
#     model.to("cpu")
#     generative_outputs, space_outputs = model.get_outputs()
#     thetas = space_outputs["theta"].detach().numpy()
#     adata.obs["inferred_theta"] = thetas
#     js_distance = utils.get_jensenshannon(
#         adata=adata, pseudotime_column="inferred_theta", hue="S-phase"
#     )
#     results["log_transformed"].append(l_transform)
#     results["divide_by_library_size"].append(divide_by_lib)
#     results["scale_by_1e4"].append(scale_by_1e4)
#     results["jensen_shannon"].append(js_distance)
#     results["scale_rhythmic_genes"].append(False)
#     pd.DataFrame(results).to_csv("vasa_normalization_results.csv", index=False)

for _ in tqdm.tqdm(range(5), total=5):
    model = CoPhaser(
        SMALL_CYCLING_GENE_SET,
        g,
        apply_scale_rhythmic_genes=True,
        apply_log_transform=True,
        divide_by_library_size=True,
        scale_by_1e4=True,
        apply_L2_norm_rhythmic_genes=True,
    )
    model.load_anndata(adata, layer_to_use="spliced")
    trainer = Trainer(
        model,
        Loss.compute_loss,
        entropy_weight_factor=50,
        MI_weight=50,
        closed_circle_weight=10,
    )
    trainer.train_model(batch_size=2048, silent=True)
    model.to("cpu")
    generative_outputs, space_outputs = model.get_outputs()
    thetas = space_outputs["theta"].detach().numpy()
    adata.obs["inferred_theta"] = thetas
    js_distance = utils.get_jensenshannon(
        adata=adata, pseudotime_column="inferred_theta", hue="S-phase"
    )
    results["log_transformed"].append(True)
    results["divide_by_library_size"].append(True)
    results["scale_by_1e4"].append(True)
    results["jensen_shannon"].append(js_distance)
    results["scale_rhythmic_genes"].append(True)
    pd.DataFrame(results).to_csv("vasa_normalization_results.csv", index=False)
