import torch
import torch.nn as nn
from typing import List, Dict, Any, Optional
from CoPhaser.model.CoPhaser import CoPhaser
import warnings
import numpy as np
import pandas as pd


class VAEModelLoader:
    """Handles loading of saved models, adapting the VAE model to new gene sets."""

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
