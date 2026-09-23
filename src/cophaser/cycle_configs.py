"""Paper-default configs for non-cell-cycle cycles, transcribed from the figure notebooks."""

from __future__ import annotations

import pandas as pd

from cophaser import gene_sets
from cophaser._resources import resource_path

Species = str  # "mouse" | "human"


MENSTRUAL_MI_THRESHOLD = 0.1


def somite_amp_phase_prior(var_names=None) -> pd.DataFrame:
    """Somite decoder warm-start table (index: gene, columns: amp, phase), optionally
    restricted to `var_names`."""
    df = pd.read_csv(resource_path("somite_genes_amplitude_phase.csv"), index_col=0)
    if var_names is not None:
        df = df[df.index.isin(list(var_names))]
    return df


def rhythmic_genes_for(cycle: str, species: Species) -> list[str]:
    """Packaged rhythmic gene list for a (cycle, species) pair.

    The menstrual list comes from an MI scan on endometrial stromal/fibroblast cells, so it
    is specific to that cell type.
    """
    if species not in ("mouse", "human"):
        raise ValueError(f"species must be 'mouse' or 'human', got {species!r}.")
    if cycle == "cell_cycle":
        genes = (
            gene_sets.HUMAN_CELL_CYCLE_GENE_SET
            if species == "human"
            else gene_sets.SMALL_CELL_CYCLE_GENE_SET
        )
        return list(genes)
    if cycle == "circadian":
        genes = (
            gene_sets.HUMAN_CIRCADIAN_GENE_SET
            if species == "human"
            else gene_sets.SMALL_CIRCADIAN_GENE_SET
        )
        return list(genes)
    if cycle == "somite":
        if species != "mouse":
            raise ValueError("Somite clock is only supported for mouse.")
        return pd.read_csv(
            resource_path("somite_genes_amplitude_phase.csv")
        )["gene"].tolist()
    if cycle == "menstrual":
        if species != "human":
            raise ValueError("Menstrual cycle is only supported for human.")
        mi = pd.read_csv(resource_path("mi_fibroblast.csv"), index_col=0)
        return mi[
            mi["mutual_information"] > MENSTRUAL_MI_THRESHOLD
        ].index.tolist()
    raise ValueError(f"No packaged rhythmic gene list for cycle={cycle!r}.")


# Circadian warm-start (amplitude, phase in rad), from paper/code/figure_5/aorta.ipynb.
# Matched case-insensitively.
CIRCADIAN_AMP_PHASE_PRIOR: dict[str, tuple[float, float]] = {
    "Bmal1": (1, 5.5),
    "Arntl": (1, 5.5),  # older name of Bmal1
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


CYCLE_TRAINER_DEFAULTS: dict[str, dict] = {
    "circadian": dict(
        source="paper/code/figure_5/aorta.ipynb (cells 17, 19, 20)",
        model=dict(n_latent=10, n_harm=1),
        trainer=dict(
            non_rhythmic_likelihood_weight=2,
            rhythmic_likelihood_weight=10,
            # Releases the genes CIRCADIAN_AMP_PHASE_PRIOR froze; change both together.
            unfreeze_epoch_layer=[(10, "rhythmic_decoder")],
            L2_Z_decoder_loss_weight=0,
            closed_circle_weight=0,
            MI_weight=50,
            entropy_weight_factor=50,
            cycling_status_prior=1,
            MI_detach="f",
        ),
        train=dict(n_epochs=200, lr=1e-2),
    ),
    "somite": dict(
        source="paper/code/figure_7/somite_clock.ipynb (cells 6-9)",
        model=dict(
            n_latent=10,
            n_harm=3,
            use_mu_z_encoder=True,
            lambda_range=2,
            z_range=20,
        ),
        n_variable_genes=1000,
        decoder_prior_resource="somite_genes_amplitude_phase.csv",
        trainer=dict(
            noise_model="NB",
            L2_Z_decoder_loss_weight=0,
            entropy_weight_factor=150,
            beta_kl_f=0.1,
            MI_weight=100,
            closed_circle_weight=15,
            cycling_status_prior=1,
            rhythmic_likelihood_weight=10,
            non_rhythmic_likelihood_weight=4,
            unfreeze_epoch_layer=[(20, "rhythmic_decoder")],
            MI_detach="none",
        ),
        train=dict(n_epochs=200, lr=1e-2, batch_size=8192),
    ),
    "menstrual": dict(
        source="paper/code/figure_6/menstrual_cycle.ipynb (cells 16-20)",
        model=dict(n_latent=10, n_harm=3, use_mu_z_encoder=True),
        n_variable_genes=2000,
        trainer=dict(
            noise_model="NB",
            L2_Z_decoder_loss_weight=0,
            entropy_weight_factor=100,
            MI_weight=100,
            closed_circle_weight=30,
            cycling_status_prior=1,
        ),
        train=dict(n_epochs=200, lr=1e-2, batch_size=1024),
    ),
}
