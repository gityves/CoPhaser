import numpy as np
from CoPhaser import utils
from CoPhaser.trainer import Trainer
from CoPhaser.loss import Loss
from CoPhaser import plotting
from CoPhaser.model import CoPhaser
from CoPhaser import gene_sets
import pickle


import matplotlib.pyplot as plt
import seaborn as sns

import anndata
import scanpy as sc

import pandas as pd
from CoPhaser import utils

import tqdm
from itertools import product

SMALL_CYCLING_GENE_SET = gene_sets.SMALL_CELL_CYCLE_GENE_SET
SMALL_CYCLING_GENE_SET = [g.upper() for g in SMALL_CYCLING_GENE_SET]

# load snr fitted on RPE data
res = pd.read_csv("data/RPE_cyclic_snr.csv")
res["gene"] = res["gene"]  # .str.capitalize()
res_cycling_g2m = res[res.gene.isin(SMALL_CYCLING_GENE_SET) & res.G2M]
res_cycling_non_g2m = res[res.gene.isin(SMALL_CYCLING_GENE_SET) & ~res.G2M]

adata = anndata.read_h5ad("data/RPE_with_gt_phase.h5ad")
g = utils.get_variable_genes(adata)

results = {
    "circcorelation": [],
    "i_min": [],
    "n_genes": [],
    "snr_g2m": [],
    "snr_non_g2m": [],
}

for i_min, size, _ in tqdm.tqdm(
    product([0, 20, -1], [1, 3, 10, 20, 30, 40], range(5)), total=3 * 6 * 5
):
    if i_min == -1:
        i_min = -size
        i_max = None
    else:
        i_max = i_min + size
    cycling_g2m = res_cycling_g2m.sort_values("snr", ascending=False).gene.tolist()[
        i_min:i_max
    ]
    cycling_non_g2m = res_cycling_non_g2m.sort_values(
        "snr", ascending=False
    ).gene.tolist()[i_min:i_max]
    subset_cycling_genes = cycling_g2m + cycling_non_g2m

    # Train model
    model = CoPhaser(
        subset_cycling_genes,
        g,
    )
    model.load_anndata(adata, layer_to_use="spliced")
    trainer = Trainer(
        model,
        Loss.compute_loss,
        noise_model="NB",
        non_rhythmic_likelihood_weight=1,
        rhythmic_likelihood_weight=1,
        L2_Z_decoder_loss_weight=0,
        closed_circle_weight=10,
        cycling_status_prior=1,
        MI_weight=150,
        entropy_weight_factor=200,
        MI_detach="f",
    )
    trainer.train_model(
        n_epochs=200, lr=1e-2, device="cuda", batch_size=2048, silent=True
    )

    # get results
    model.to("cpu")
    generative_outputs, space_outputs = model.get_outputs()
    thetas = space_outputs["theta"]
    circcorr = np.abs(
        utils.circular_correlation(thetas, adata.obs["ground_truth_phase"])
    )

    # Save results
    results["circcorelation"].append(circcorr)
    results["i_min"].append(i_min)
    results["n_genes"].append(size)
    results["snr_g2m"].append(
        res_cycling_g2m[res_cycling_g2m.gene.isin(subset_cycling_genes)].snr.sum()
    )
    results["snr_non_g2m"].append(
        res_cycling_non_g2m[
            res_cycling_non_g2m.gene.isin(subset_cycling_genes)
        ].snr.sum()
    )

results_df = pd.DataFrame(results)
results_df.to_csv("data/RPE_results.csv", index=False)
