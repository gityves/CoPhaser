import numpy as np
from CoPhaser import utils
from CoPhaser.trainer import Trainer
from CoPhaser.loss import Loss
from CoPhaser import plotting
from CoPhaser.model import CoPhaser
from CoPhaser.model import VAEModelLoader
from CoPhaser import gene_sets
from scanpy.pp import downsample_counts

import matplotlib.pyplot as plt
import seaborn as sns

import anndata
import scanpy as sc

import pandas as pd
import tqdm
from itertools import product

DATA_FOLDER = "data/"

adata = anndata.read_h5ad(f"{DATA_FOLDER}/aorta_gt.h5ad")
amp_phase_circadian = {
    "Bmal1": (1, 5.5),
    "Npas2": (0.5, 5.5),
    "Rorc": (0.5, 5.7),
    "Nr1d1": (1, 2.0),
    "Nr1d2": (0.75, 2.09),
    "Tef": (0.75, 2.35),
    "Ciart": (1, 2.35),
    "Dbp": (1, 2.09),
    "Per3": (1, 2.36),
    "Cry1": (0.5, 4.71),
    "Cry2": (0.25, 3),
    "Per2": (0.6, 3.14),
    "Per1": (0.5, 3.14),
    "Hlf": (0.5, 3.14),
}

g = utils.get_variable_genes(adata, 2000)
SMALL_CYCLING_GENE_SET = gene_sets.SMALL_CIRCADIAN_GENE_SET
genes_tempo = [
    "Arntl",
    "Arntl2",
    "Bhlhe41",
    "Ciart",
    "Clock",
    "Cry1",
    "Cry2",
    "Csnk1a1",
    "Csnk1d",
    "Csnk1e",
    "Dbp",
    "Dec1",
    "Dec2",
    "Fbxo21",
    "Fbxo3",
    "Gm129",
    "Hlf",
    "Nampt",
    "Nfil3",
    "Npas2",
    "Nr1d1",
    "Nr1d2",
    "Per1",
    "Per2",
    "Per3",
    "Rora",
    "Rorc",
    "Tef",
]

SMALL_CYCLING_GENE_SET = np.array(
    list(
        set(list(gene_sets.SMALL_CIRCADIAN_GENE_SET) + genes_tempo)
        & set(adata.var_names)
    )
)

results = {
    "circcorelation": [],
    "total_counts": [],
}
for total_counts, _ in tqdm.tqdm(
    product(range(1000, 14_000, 1000), range(5)), desc="Downsampling", total=65
):
    # downsample umi
    adata.X = adata.layers["total"].copy()
    downsample_counts(adata, counts_per_cell=int(total_counts))
    adata.layers["downsampled"] = adata.X

    model = CoPhaser(
        SMALL_CYCLING_GENE_SET,
        g,
        n_latent=10,
        n_harm=1,
        rhythmic_decoder_to_all_genes=True,
        use_mu_z_encoder=True,
        z_range=20,
    )
    model.load_anndata(adata, layer_to_use="downsampled")

    VAEModelLoader.define_decoder_prior(
        amp_phase_prior=amp_phase_circadian, model=model
    )
    trainer = Trainer(
        model,
        Loss.compute_loss,
        non_rhythmic_likelihood_weight=2,
        rhythmic_likelihood_weight=10,
        unfreeze_epoch_layer=[(10, "rhythmic_decoder")],
        L2_Z_decoder_loss_weight=0,
        closed_circle_weight=0,
        MI_weight=50,
        entropy_weight_factor=50,
        cycling_status_prior=1,
        MI_detach="f",
    )
    trainer.train_model(
        n_epochs=200,
        lr=1e-2,
        device="cuda",
        batch_size=2048,
        silent=True,
    )
    model.to("cpu")
    generative_outputs, space_outputs = model.get_outputs()
    thetas = space_outputs["theta"].cpu().detach().numpy()

    circcorr = np.abs(
        utils.circular_correlation(thetas, adata.obs["ground_truth_phase"])
    )
    results["circcorelation"].append(circcorr)
    results["total_counts"].append(total_counts)
    print(f"Downsampled to {total_counts} counts, circular correlation: {circcorr:.4f}")

results_df = pd.DataFrame(results)
results_df.to_csv(f"{DATA_FOLDER}/aorta_downsample_results.csv", index=False)
