import pathlib
import re
import torch
import torch.nn as nn
from typing import List, Dict, Any, Optional
from CoPhaser.model.CoPhaser import CoPhaser
import warnings
import numpy as np
import pandas as pd


class VAEModelLoader:
    """Handles loading of decoder priors."""

    @staticmethod
    def define_decoder_prior(
        amp_phase_prior: dict | pd.DataFrame,
        model: CoPhaser,
        freeze_defined_weights=True,
    ):
        """
        Set the fourier coefficients according to the prior in phase (in rad) and amplitude.

        Parameters:
        -----------
        amp_phase_prior: dict with as keys gene names, and values (amp,phase) or df with gene as index and amp, phase rows.
        model: model to be modified
        freeze_defined_weights: freeze the newly genes with newly set phase and amplitude

        """
        if isinstance(amp_phase_prior, pd.DataFrame):
            amp_phase_prior = {
                gene.upper(): (row.amp, row.phase)
                for gene, row in amp_phase_prior.iterrows()
            }
        else:
            amp_phase_prior = {k.upper(): v for k, v in amp_phase_prior.items()}

        new_state_dict = model.rhythmic_decoder.state_dict()
        new_weights = new_state_dict["fourier_coefficients.weight"]
        gene_names = (
            model.context_genes
            if model.rhythmic_decoder_to_all_genes
            else model.rhythmic_gene_names
        )
        gene_mapping = {gene.upper(): idx for idx, gene in enumerate(gene_names.copy())}
        genes_modified = torch.zeros(len(gene_names), dtype=bool)
        for gene in amp_phase_prior.keys():
            gene = gene.upper()
            if not gene in gene_mapping.keys():
                warnings.warn(
                    f"Gene {gene} was not found in the model's genes, prior not loaded"
                )
                continue
            amp, phase = amp_phase_prior[gene]
            a = amp * np.cos(phase)
            b = amp * np.sin(phase)
            with torch.no_grad():
                new_weights[gene_mapping[gene], 0] = torch.tensor(
                    a, device=model.rhythmic_decoder.fourier_coefficients.weight.device
                )
                new_weights[gene_mapping[gene], 1] = torch.tensor(
                    b, device=model.rhythmic_decoder.fourier_coefficients.weight.device
                )
                # set higher harmonics to 0
                new_weights[gene_mapping[gene], 2:] *= 0
            genes_modified[gene_mapping[gene]] = True
        new_state_dict["fourier_coefficients.weight"] = torch.nn.Parameter(new_weights)
        with torch.no_grad():
            model.rhythmic_decoder.load_state_dict(new_state_dict)
        if freeze_defined_weights:
            model.rhythmic_decoder.freeze_weights_genes(genes_modified)

    @staticmethod
    def load_fourier_coefficients_prior(
        fourier_coeffs: str | pathlib.PurePath | pd.DataFrame,
        model: CoPhaser,
        freeze_defined_weights: bool = True,
    ) -> torch.Tensor:
        """
        Load a table of Fourier coefficients (columns A_1, B_1, A_2, B_2, ...,)
        as a prior for the rhythmic decoder's fourier
        coefficient weights. Genes are matched by name (case-insensitive).

        Parameters
        ----------
        fourier_coeffs: path to a csv, or a DataFrame, indexed by gene name
            with columns A_1, B_1, A_2, B_2, ... (A_0 ignored).
        model: model to modify.
        freeze_defined_weights: freeze the weights of genes that were set.

        Returns
        -------
        genes_modified: bool tensor over the model's gene dimension indicating
            which genes had their weights set from the prior.
        """
        # --- load ---
        if isinstance(fourier_coeffs, (str, pathlib.PurePath)):
            f_coeffs = pd.read_csv(fourier_coeffs, index_col=0)
        elif isinstance(fourier_coeffs, pd.DataFrame):
            f_coeffs = fourier_coeffs.copy()
        else:
            raise TypeError(
                f"fourier_coeffs must be a path or DataFrame, got {type(fourier_coeffs)}"
            )

        f_coeffs.index = f_coeffs.index.astype(str).str.upper()
        if f_coeffs.index.duplicated().any():
            dupes = f_coeffs.index[f_coeffs.index.duplicated()].unique().tolist()
            warnings.warn(
                f"fourier_coeffs has duplicate gene entries, keeping the first "
                f"occurrence for: {dupes}"
            )
            f_coeffs = f_coeffs[~f_coeffs.index.duplicated(keep="first")]

        # --- identify harmonic columns present (A_n/B_n, n>=1); drop everything else (e.g. A_0) ---
        harmonic_pattern = re.compile(r"^[AB]_(\d+)$")
        harmonics_available = sorted(
            {
                int(m.group(1))
                for col in f_coeffs.columns
                if (m := harmonic_pattern.match(str(col))) and int(m.group(1)) > 0
            }
        )
        if not harmonics_available:
            raise ValueError(
                "No harmonic columns (A_1, B_1, A_2, B_2, ...) found in fourier_coeffs."
            )

        # --- how many harmonics does the model expect? ---
        fourier_weight = model.rhythmic_decoder.fourier_coefficients.weight
        n_features = fourier_weight.shape[1]
        if n_features % 2 != 0:
            raise ValueError(
                f"Expected an even number of fourier coefficient columns (A_n/B_n "
                f"pairs), but model has {n_features}."
            )
        n_harmonics_model = n_features // 2
        harmonics_model = list(range(1, n_harmonics_model + 1))

        # --- build ordered A_n,B_n columns matching the model, zero-padding as needed ---
        ordered_cols = []
        for h in harmonics_model:
            for coeff in ("A", "B"):
                col = f"{coeff}_{h}"
                if col not in f_coeffs.columns:
                    f_coeffs[col] = 0.0
                ordered_cols.append(col)

        dropped_harmonics = sorted(set(harmonics_available) - set(harmonics_model))
        if dropped_harmonics:
            warnings.warn(
                f"Model uses {n_harmonics_model} harmonics; dropping unused "
                f"harmonics from prior: {dropped_harmonics}"
            )
        missing_harmonics = sorted(set(harmonics_model) - set(harmonics_available))
        if missing_harmonics:
            warnings.warn(
                f"Model uses {n_harmonics_model} harmonics but prior only provides "
                f"harmonics {harmonics_available}; setting missing harmonics "
                f"{missing_harmonics} to 0."
            )

        f_coeffs = f_coeffs[ordered_cols]

        # --- match genes ---
        gene_names = (
            model.context_genes
            if model.rhythmic_decoder_to_all_genes
            else model.rhythmic_gene_names
        )
        gene_names_upper = [g.upper() for g in gene_names]
        gene_mapping = {gene: idx for idx, gene in enumerate(gene_names_upper)}

        genes_in_prior_not_model = sorted(
            set(f_coeffs.index) - set(gene_mapping.keys())
        )
        if genes_in_prior_not_model:
            preview = genes_in_prior_not_model[:10]
            suffix = "..." if len(genes_in_prior_not_model) > 10 else ""
            warnings.warn(
                f"{len(genes_in_prior_not_model)} genes in fourier_coeffs were not "
                f"found in the model and will be ignored: {preview}{suffix}"
            )

        matched_genes = [g for g in f_coeffs.index if g in gene_mapping]
        if not matched_genes:
            raise ValueError(
                "None of the genes in fourier_coeffs were found in the model."
            )
        matched_indices = [gene_mapping[g] for g in matched_genes]

        # --- write weights ---
        new_weights = fourier_weight.detach().clone()
        values = torch.tensor(
            f_coeffs.loc[matched_genes].values,
            dtype=new_weights.dtype,
            device=new_weights.device,
        )
        new_weights[matched_indices, :] = values
        model.rhythmic_decoder.fourier_coefficients.weight = torch.nn.Parameter(
            new_weights
        )

        genes_modified = torch.zeros(len(gene_names_upper), dtype=torch.bool)
        genes_modified[matched_indices] = True

        if freeze_defined_weights:
            model.rhythmic_decoder.freeze_weights_genes(genes_modified)

        return genes_modified
