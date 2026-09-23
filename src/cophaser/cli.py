"""Headless training entrypoint, used standalone and by the GUI.

    python -m cophaser.cli train --config run.json

The GUI inlines the body of ``train()`` into the script it runs (see gui/script_gen.py) and
parses its printed lines for progress.

Config fields
-------------
h5ad_path : str
layer : str or None
    Counts layer; None uses `.X` (only allowed when the file has no layers).
cycle : "cell_cycle" | "circadian" | "somite" | "menstrual"
species : "mouse" | "human"
gene_symbol_column : str, optional
    `.var` column to use as `.var_names` (e.g. when var_names are Ensembl IDs).
rhythmic_genes : list[str]
    Rhythmic gene names; those absent from the dataset are dropped with a warning.
n_variable_genes : int, optional (default 2000)
min_counts, max_counts : float, optional
    QC thresholds on total counts per cell in `layer`.
model_kwargs : dict, optional
    Passed to CoPhaser(...).
preload_fourier_prior : bool, optional
    Warm-start and freeze the rhythmic decoder from the RPE Fourier prior (cell_cycle only).
    Unset: on in autotune mode when cycling_status_prior < 1.
decoder_amp_phase_prior : dict, optional
    {gene: (amplitude, phase_in_rad)} to warm-start and freeze the rhythmic decoder; pair it
    with a trainer `unfreeze_epoch_layer` entry.
decoder_amp_phase_resource : str, optional
    Same, from a packaged CSV (index: gene, columns: amp, phase). Takes precedence.
hyperparams_mode : "autotune" | "fixed" | "manual"
    auto_hyperparameters / cycle_configs.CYCLE_TRAINER_DEFAULTS[cycle] / trainer_kwargs only.
    Default: autotune for cell_cycle, fixed otherwise.
trainer_kwargs : dict, optional
    Verbatim in "manual" mode, overrides otherwise.
train_kwargs : dict, optional
    Passed to Trainer.train_model(...).
seeds : list[int], optional (default [0])
    One fresh model per seed.
run_names : list[str], optional (default [f"seed_{s}" for s in seeds])
    Output subfolder per seed; unique, same length as `seeds`.
output_dir : str
    Outputs go to `output_dir/<run_name>/{model.pt, inferred_phases.csv}`.
"""

from __future__ import annotations

import argparse
import json
import os
import random

import numpy as np
import pandas as pd
import torch

import scanpy as sc

from cophaser import Loss, Trainer, CoPhaser, auto_hyperparameters, utils, cycle_configs


def _resolve_trainer_kwargs(config: dict, model, cycle: str, adata=None, layer=None) -> dict:
    mode = config.get("hyperparams_mode") or ("autotune" if cycle == "cell_cycle" else "fixed")
    overrides = config.get("trainer_kwargs", {}) or {}

    if mode == "manual":
        return dict(overrides)

    if mode == "autotune":
        if cycle not in cycle_configs.CYCLE_TRAINER_DEFAULTS and cycle != "cell_cycle":
            raise ValueError(f"Autotune is not available for cycle={cycle!r} yet.")
        # a user-set prior also changes the derived likelihood weights
        prior = overrides.get("cycling_status_prior") if cycle == "cell_cycle" else None
        result = auto_hyperparameters(
            model, cycle=cycle, adata=adata, layer=layer, cycling_status_prior=prior
        )
        kwargs = result["trainer"]
    elif mode == "fixed":
        if cycle not in cycle_configs.CYCLE_TRAINER_DEFAULTS:
            raise ValueError(f"No fixed default trainer config for cycle={cycle!r}.")
        kwargs = dict(cycle_configs.CYCLE_TRAINER_DEFAULTS[cycle]["trainer"])
    else:
        raise ValueError(f"Unknown hyperparams_mode={mode!r}.")

    kwargs.update(overrides)
    return kwargs


def train(config: dict) -> None:
    output_dir = config["output_dir"]
    os.makedirs(output_dir, exist_ok=True)

    cycle = config["cycle"]
    layer = config.get("layer")

    print(f"Loading {config['h5ad_path']} ...", flush=True)
    adata = sc.read_h5ad(config["h5ad_path"])
    if layer is None:
        if len(adata.layers):
            raise ValueError(
                f"layer=None reads counts from .X, but this file has layers "
                f"{list(adata.layers)}; set `layer` to the one holding raw counts."
            )
        adata.layers["total"] = adata.X
        layer = "total"

    symbol_column = config.get("gene_symbol_column")
    if symbol_column:
        adata.var_names = adata.var[symbol_column].astype(str).values
        adata.var_names_make_unique()
        print(f"Using .var['{symbol_column}'] as gene names", flush=True)

    min_counts = config.get("min_counts")
    max_counts = config.get("max_counts")
    if min_counts is not None or max_counts is not None:
        total_counts = adata.layers[layer].sum(axis=1)
        total_counts = getattr(total_counts, "A1", total_counts)
        keep = (total_counts >= min_counts if min_counts is not None else True) & (
            total_counts <= max_counts if max_counts is not None else True
        )
        n_before = adata.n_obs
        adata = adata[keep].copy()
        print(f"QC filter: kept {adata.n_obs}/{n_before} cells", flush=True)

    context_genes = utils.get_variable_genes(
        adata, n_variable_genes=config.get("n_variable_genes", 2000), layer=layer
    )

    seeds = list(config.get("seeds") or [config.get("seed", 0)])
    run_names = list(config.get("run_names") or [f"seed_{s}" for s in seeds])
    if len(run_names) != len(seeds) or len(set(run_names)) != len(run_names):
        raise ValueError(
            f"run_names must be unique and match seeds one-to-one; got {run_names} for seeds {seeds}."
        )
    mode = config.get("hyperparams_mode") or ("autotune" if cycle == "cell_cycle" else "fixed")
    model_kwargs = config.get("model_kwargs", {}) or {}
    train_kwargs = config.get("train_kwargs", {}) or {}
    trainer_cycle = cycle if cycle in ("cell_cycle", "circadian") else None
    rhythmic_genes = [g for g in config["rhythmic_genes"] if g in adata.var_names]
    missing = [g for g in config["rhythmic_genes"] if g not in adata.var_names]
    if missing:
        print(
            f"Warning: {len(missing)} rhythmic genes not in the dataset, dropped: {missing}",
            flush=True,
        )
    if not rhythmic_genes:
        raise ValueError("None of the rhythmic genes are in the dataset.")

    for run_idx, (seed, run_name) in enumerate(zip(seeds, run_names), start=1):
        print(
            f"=== RUN {run_idx}/{len(seeds)} ({run_name}, seed={seed}) ===", flush=True
        )
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        model = CoPhaser(rhythmic_genes, context_genes, **model_kwargs)
        model.load_anndata(adata, layer_to_use=layer)

        trainer_kwargs = _resolve_trainer_kwargs(config, model, cycle, adata=adata, layer=layer)

        preload_fourier = config.get("preload_fourier_prior")
        if preload_fourier is None:
            preload_fourier = mode == "autotune" and float(
                trainer_kwargs.get("cycling_status_prior", 1)
            ) < 1
        if preload_fourier and cycle == "cell_cycle":
            from cophaser._resources import resource_path
            from cophaser.model.decoder_prior import DecoderPrior

            DecoderPrior.load_fourier_coefficients_prior(
                resource_path("fourier_coefficients_RPE.csv"),
                model,
                freeze_defined_weights=True,
            )
            print(
                "Rhythmic decoder warm-started from the packaged RPE Fourier coefficients",
                flush=True,
            )

        amp_phase_prior = config.get("decoder_amp_phase_prior")
        prior_resource = config.get("decoder_amp_phase_resource")
        if prior_resource:
            from cophaser._resources import resource_path

            amp_phase_prior = pd.read_csv(
                resource_path(prior_resource),
                index_col=0,
            )
            amp_phase_prior = amp_phase_prior[
                amp_phase_prior.index.isin(adata.var_names)
            ]
        if amp_phase_prior is not None and len(amp_phase_prior):
            from cophaser.model.decoder_prior import DecoderPrior

            DecoderPrior.define_decoder_prior(
                amp_phase_prior=amp_phase_prior, model=model
            )

        warm_started = (preload_fourier and cycle == "cell_cycle") or (
            amp_phase_prior is not None and len(amp_phase_prior) > 0
        )
        if warm_started:
            # otherwise the warm-started decoder stays frozen for the whole run
            default_unfreeze = cycle_configs.CYCLE_TRAINER_DEFAULTS.get(cycle, {}).get(
                "trainer", {}
            ).get("unfreeze_epoch_layer", [(20, "rhythmic_decoder")])
            trainer_kwargs.setdefault("unfreeze_epoch_layer", default_unfreeze)

        trainer = Trainer(
            model, Loss.compute_loss, cycle=trainer_cycle, **trainer_kwargs
        )

        run_train_kwargs = dict(train_kwargs)
        if run_train_kwargs.get("subsample") is not None:
            # train_model's subsample uses its own RNG; tie it to the seed for reproducibility.
            run_train_kwargs.setdefault("subsample_seed", seed)

        trainer.train_model(**run_train_kwargs)

        run_output_dir = os.path.join(output_dir, run_name)
        os.makedirs(run_output_dir, exist_ok=True)
        model_path = os.path.join(run_output_dir, "model.pt")
        model.to("cpu")
        model.save(model_path)
        print(f"Saved model to {model_path}", flush=True)

        generative_outputs, space_outputs = model.get_outputs()
        thetas = space_outputs["theta"]
        if cycle == "cell_cycle":
            thetas = model.orient_align_pseudotimes(thetas, plot=False)

        z = space_outputs["z"].detach().numpy()
        df = pd.DataFrame(
            z, columns=[f"z{i}" for i in range(z.shape[1])], index=adata.obs_names
        )
        df.insert(0, "inferred_phase", thetas.detach().numpy())
        csv_path = os.path.join(run_output_dir, "inferred_phases.csv")
        df.to_csv(csv_path)
        print(f"Saved inferred phases and latent dims to {csv_path}", flush=True)

    print("DONE", flush=True)


def main():
    parser = argparse.ArgumentParser(prog="python -m cophaser.cli")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser(
        "train", help="Train a CoPhaser model from a config JSON."
    )
    train_parser.add_argument(
        "--config", required=True, help="Path to a run config JSON file."
    )

    args = parser.parse_args()
    if args.command == "train":
        with open(args.config) as f:
            config = json.load(f)
        train(config)


if __name__ == "__main__":
    main()
