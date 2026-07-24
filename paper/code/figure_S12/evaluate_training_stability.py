from importlib import resources

import numpy as np
from CoPhaser import utils
from CoPhaser.trainer import Trainer
from CoPhaser.loss import Loss
from CoPhaser.model import CoPhaser
from CoPhaser import gene_sets

import anndata

import pandas as pd
import torch
import tqdm

DATA_FOLDER = "data/"

adata = anndata.read_h5ad(f"{DATA_FOLDER}/breast_cancer_gt.h5ad")
g = utils.get_variable_genes(adata)

SMALL_CYCLING_GENE_SET = gene_sets.SMALL_CELL_CYCLE_GENE_SET
SMALL_CYCLING_GENE_SET = [g.upper() for g in SMALL_CYCLING_GENE_SET]
SMALL_CYCLING_GENE_SET = np.array(
    list(set(SMALL_CYCLING_GENE_SET) & set(adata.var_names))
)

results = {
    "circcorelation": [],
}
for _ in tqdm.tqdm(range(100)):
    try:
        model = CoPhaser(
            SMALL_CYCLING_GENE_SET,
            g,
        )
        model.load_anndata(adata, layer_to_use="total")
        f_coeffs_path = (
            resources.files("CoPhaser") / "resources" / "fourier_coefficients_RPE.csv"
        )
        f_coeffs = pd.read_csv(f_coeffs_path, index_col=0)
        f_coeffs.drop("A_0", axis=1, inplace=True)
        f_coeffs = f_coeffs.loc[model.rhythmic_gene_names].copy()
        old_weights = (
            model.rhythmic_decoder.fourier_coefficients.weight.detach().clone()
        )
        old_weights[model.rhythmic_gene_indices, :] = torch.tensor(
            f_coeffs.values
        ).float()
        old_weights = torch.nn.Parameter(old_weights)
        model.rhythmic_decoder.fourier_coefficients.weight = old_weights
        model.rhythmic_decoder.freeze_weights_genes(model.rhythmic_gene_indices)
        trainer = Trainer(
            model,
            Loss.compute_loss,
            calculate_entropy_per_batch=False,
            L2_Z_decoder_loss_weight=0,
            entropy_weight_factor=100,
            closed_circle_weight=10,
            MI_weight=100,
            cycling_status_prior=0.1,
            beta_kl_cycling_status=20,
            unfreeze_epoch_layer=[(20, "rhythmic_decoder")],
            rhythmic_likelihood_weight=20,
            non_rhythmic_likelihood_weight=5,
        )
        trainer.train_model(
            n_epochs=200,
            lr=1e-2,
            device="cuda",
            batch_size=4096,
            silent=True,
        )
        model.to("cpu")
        generative_outputs, space_outputs = model.get_outputs()
        thetas = space_outputs["theta"].cpu().detach().numpy()

        circcorr = np.abs(
            utils.circular_correlation(thetas, adata.obs["ground_truth_phase"])
        )
        results["circcorelation"].append(circcorr)
        print(f"Circular correlation: {circcorr:.4f}")
    except Exception as e:
        print(f"Error: {e}")
        results["circcorelation"].append(np.nan)
    results_df = pd.DataFrame(results)
    results_df.to_csv(f"{DATA_FOLDER}/breast_cancer_stability_results.csv", index=False)
