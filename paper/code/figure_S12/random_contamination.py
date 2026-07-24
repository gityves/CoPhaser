import numpy as np
from CoPhaser import utils
from CoPhaser.trainer import Trainer
from CoPhaser.loss import Loss
from CoPhaser import plotting
from CoPhaser.model import CoPhaser
from CoPhaser import gene_sets


import anndata

import pandas as pd
from CoPhaser import utils

import tqdm
from itertools import product

SMALL_CYCLING_GENE_SET = gene_sets.SMALL_CELL_CYCLE_GENE_SET

# load snr fitted on RPE data
res = pd.read_csv("data/RPE_cyclic_snr.csv")
res["gene"] = res["gene"].str.capitalize()
res_cycling_g2m = res[res.gene.isin(SMALL_CYCLING_GENE_SET) & res.G2M]
res_cycling_non_g2m = res[res.gene.isin(SMALL_CYCLING_GENE_SET) & ~res.G2M]

adata = anndata.read_h5ad("data/RPE_with_gt_phase.h5ad")
g = utils.get_variable_genes(adata)

# check capitalization of gene names using Top2a
to_upper = False
if "TOP2A" in adata.var_names:
    SMALL_CYCLING_GENE_SET = [gene.upper() for gene in SMALL_CYCLING_GENE_SET]
    res["gene"] = res["gene"].str.upper()
    to_upper = True
elif not "Top2a" in adata.var_names:
    raise ValueError(
        "Gene names in adata do not match expected capitalization. Please check gene names in adata."
    )


# compute mean normalized expression of the genes
norm_expr = np.log1p(
    adata[:, g].layers["spliced"].toarray()
    / adata[:, g].layers["spliced"].toarray().sum(axis=1, keepdims=True)
    * 1e4
).mean(axis=0)
df_norm_expr = pd.DataFrame({"gene": g, "norm_expr": norm_expr})
res_variable_genes = res[res["gene"].isin(g)]
res_variable_genes = res_variable_genes.merge(df_norm_expr, on="gene")
unrelated_genes = (
    res_variable_genes[res_variable_genes["norm_expr"] > 0.1]
    .sort_values("snr")
    .head(100)
    .gene.tolist()
)

results = {
    "circcorelation": [],
    "percent_random": [],
    "n_random_genes": [],
    "snr_g2m": [],
    "snr_non_g2m": [],
}

unrelated_genes = [gene for gene in unrelated_genes if gene in g]
print(f"Number of unrelated genes: {len(unrelated_genes)}")
res_cycling_g2m = res[res.gene.isin(SMALL_CYCLING_GENE_SET) & res.G2M]
res_cycling_non_g2m = res[res.gene.isin(SMALL_CYCLING_GENE_SET) & ~res.G2M]
# fix random seed for reproducibility
np.random.seed(0)
for percent_random, _ in tqdm.tqdm(
    product([1, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100], range(10)), total=120
):
    # compute number of genes to drop
    n_g2m_drop = np.round(len(res_cycling_g2m) * percent_random / 100)
    n_non_g2m_drop = np.round(len(res_cycling_non_g2m) * percent_random / 100)
    # drop genes with the lowest snr
    g2m_genes = (
        res_cycling_g2m.sort_values("snr")
        .head(len(res_cycling_g2m) - int(n_g2m_drop))
        .gene.tolist()
    )
    non_g2m_genes = (
        res_cycling_non_g2m.sort_values("snr")
        .head(len(res_cycling_non_g2m) - int(n_non_g2m_drop))
        .gene.tolist()
    )
    subset_cycling_genes = g2m_genes + non_g2m_genes
    n_random_genes = int(n_g2m_drop + n_non_g2m_drop)
    random_genes = np.random.choice(
        unrelated_genes, size=n_random_genes, replace=False
    ).tolist()

    # Train model
    model = CoPhaser(
        subset_cycling_genes + random_genes,
        g,
        n_latent=2,
    )
    model.load_anndata(adata, layer_to_use="spliced")
    try:
        trainer = Trainer(
            model,
            Loss.compute_loss,
            noise_model="NB",
            closed_circle_weight=10,
            MI_weight=150,
            entropy_weight_factor=200,
            MI_detach="f",
            rhythmic_likelihood_weight=1,
        )
        trainer.train_model(
            n_epochs=200,
            lr=1e-2,
            batch_size=1024,
            silent=True,
        )
    except Exception as e:
        print(
            f"Error occurred while training model for percent_random {percent_random}: {e}"
        )
        continue
    model.to("cpu")
    generative_outputs, space_outputs = model.get_outputs()
    thetas = space_outputs["theta"]
    circcorr = np.abs(
        utils.circular_correlation(thetas, adata.obs["ground_truth_phase"])
    )
    results["circcorelation"].append(circcorr)
    results["percent_random"].append(percent_random)
    results["n_random_genes"].append(n_random_genes)
    results["snr_g2m"].append(
        res_cycling_g2m[res_cycling_g2m.gene.isin(subset_cycling_genes)].snr.sum()
    )
    results["snr_non_g2m"].append(
        res_cycling_non_g2m[
            res_cycling_non_g2m.gene.isin(subset_cycling_genes)
        ].snr.sum()
    )
    print(
        f"Percent random: {percent_random}%, Circcorrelation: {circcorr:.4f}, SNR G2M: {results['snr_g2m'][-1]:.4f}, SNR non-G2M: {results['snr_non_g2m'][-1]:.4f}"
    )

results_df = pd.DataFrame(results)
results_df.to_csv("data/RPE_contamination_results.csv", index=False)
