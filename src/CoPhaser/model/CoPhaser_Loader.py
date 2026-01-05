import torch
import torch.nn as nn
from typing import List, Dict, Any, Optional
from CoPhaser.model.CoPhaser import CoPhaser
import warnings
import numpy as np
import pandas as pd


# TODO: modify this to keep only used functions
class VAEModelLoader:
    """Handles loading of saved models, adapting the VAE model to new gene sets."""

    @staticmethod
    def load(
        file_path: str,
        new_rhythmic_genes: List[str] = None,
        new_variable_genes: List[str] = None,
        new_n_harm: int = None,
        new_n_latent: int = None,
        rhythmic_decoder_to_all_genes: bool = None,
    ):
        """Loads model weights and adapts them for the new dataset."""

        # load old model parameters
        checkpoint = torch.load(file_path, map_location=torch.device("cpu"))
        old_rhythmic_genes = checkpoint["rhythmic_gene_names"]
        old_variable_genes = checkpoint["context_genes"]
        old_n_harm = checkpoint["n_harm"]
        old_n_latent = checkpoint["n_latent"]
        old_rhythmic_decoder_to_all_genes = checkpoint["rhythmic_decoder_to_all_genes"]

        # Create new model
        model_args = {}
        model_args["rhythmic_gene_names"] = (
            new_rhythmic_genes if new_rhythmic_genes is not None else old_rhythmic_genes
        )
        model_args["context_genes"] = (
            new_variable_genes if new_variable_genes is not None else old_variable_genes
        )
        model_args["n_latent"] = (
            new_n_latent if new_n_latent is not None else old_n_latent
        )
        model_args["n_harm"] = new_n_harm if new_n_harm is not None else old_n_harm
        model_args["rhythmic_decoder_to_all_genes"] = (
            rhythmic_decoder_to_all_genes
            if rhythmic_decoder_to_all_genes is not None
            else old_rhythmic_decoder_to_all_genes
        )
        model = CoPhaser(**model_args)
        new_variable_genes = model.context_genes
        new_rhythmic_genes = model.rhythmic_gene_names

        with torch.no_grad():
            # adapt dispersion parameter
            VAEModelLoader._adapt_dispersion_parameter(
                model,
                old_variable_genes,
                new_variable_genes,
                checkpoint["log_theta_dispersion"],
            )
            # Adapt rhythmic encoder
            VAEModelLoader._adapt_module_weights(
                model.rhythmic_encoder,
                checkpoint["rhythmic_encoder"],
                old_rhythmic_genes,
                new_rhythmic_genes,
                input_layer_names=["fc1.weight", "kappa_net.neural_net.0.weight"],
                bias_name="b",
            )

            # Adapt rhythmic decoder with special handling for n_harm
            old_decoder_genes = (
                old_variable_genes
                if old_rhythmic_decoder_to_all_genes
                else old_rhythmic_genes
            )
            new_decoder_genes = (
                model.context_genes
                if model.rhythmic_decoder_to_all_genes
                else model.rhythmic_gene_names
            )

            VAEModelLoader._adapt_rhythmic_decoder(
                model.rhythmic_decoder,
                checkpoint["rhythmic_decoder"],
                old_decoder_genes,
                new_decoder_genes,
                old_n_harm,
                model.n_harm,
            )

            # Only adapt latent-dependent parts if latent dimensions match
            if old_n_latent == model.n_latent:
                for encoder_name in ["mean_encoder", "var_encoder"]:
                    VAEModelLoader._adapt_module_weights(
                        getattr(model, encoder_name),
                        checkpoint[encoder_name],
                        old_variable_genes,
                        new_variable_genes,
                        input_layer_names=["neural_net.0.weight"],
                    )

                VAEModelLoader._adapt_module_weights(
                    model.decoder_non_rhythmic_contribution,
                    checkpoint["decoder_non_rhythmic_contribution"],
                    old_variable_genes,
                    new_variable_genes,
                    output_layer_names=["neural_net.2.weight"],
                    bias_name="neural_net.2.bias",
                )

                VAEModelLoader._adapt_module_weights(
                    model.mu_z_decoder,
                    checkpoint["mu_z_decoder"],
                    old_rhythmic_genes,
                    new_rhythmic_genes,
                    output_layer_names=["neural_net.2.weight"],
                    bias_name="neural_net.2.bias",
                )

        return model

    @staticmethod
    def _adapt_dispersion_parameter(
        model: CoPhaser,
        old_genes: List[str],
        new_genes: List[str],
        old_dispersion,
    ):
        gene_mapping = {gene: idx for idx, gene in enumerate(old_genes)}
        new_indices = [gene_mapping[gene] for gene in new_genes if gene in gene_mapping]
        for i, old_idx in enumerate(new_indices):
            model.log_theta_dispersion[i] = old_dispersion[old_idx]

    @staticmethod
    def _adapt_module_weights(
        module: nn.Module,
        old_state_dict: Dict[str, Any],
        old_genes: List[str],
        new_genes: List[str],
        input_layer_names: List[str] = [],
        output_layer_names: List[str] = [],
        bias_name: Optional[str] = None,
    ):
        """
        Generic function to adapt module weights when genes are added or removed.

        Parameters:
        -----------
        module: The module to modify
        old_state_dict: The state dict with old weights
        old_genes: List of gene names in the old model
        new_genes: List of gene names in the new model
        input_layer_name: Name of the input layer weight parameter (if adapting inputs)
        output_layer_name: Name of the output layer weight parameter (if adapting outputs)
        bias_name: Name of the input bias parameter
        """
        new_state_dict = module.state_dict()
        gene_mapping = {gene: idx for idx, gene in enumerate(old_genes)}
        new_indices = [gene_mapping[gene] for gene in new_genes if gene in gene_mapping]

        for name, param in old_state_dict.items():
            if name in new_state_dict:
                if name in input_layer_names:
                    # Input layer weight adjustment (columns correspond to input genes)
                    old_weights = param
                    new_weights = new_state_dict[name]

                    # Copy existing weights
                    for i, old_idx in enumerate(new_indices):
                        new_weights[:, i] = old_weights[:, old_idx]

                    new_state_dict[name] = torch.nn.Parameter(new_weights)

                elif name in output_layer_names:
                    # Output layer weight adjustment (rows correspond to output genes)
                    old_weights = param
                    new_weights = new_state_dict[name]

                    # Copy existing weights
                    for i, old_idx in enumerate(new_indices):
                        new_weights[i] = old_weights[old_idx]

                    new_state_dict[name] = torch.nn.Parameter(new_weights)

                elif bias_name and bias_name == name:
                    # Input bias adjustment
                    old_bias = param
                    new_bias = new_state_dict[name]

                    # Copy existing biases
                    for i, old_idx in enumerate(new_indices):
                        new_bias[i] = old_bias[old_idx]

                    new_state_dict[name] = torch.nn.Parameter(new_bias)

                else:
                    # Copy unchanged parameters
                    new_state_dict[name] = param

        module.load_state_dict(new_state_dict)

    @staticmethod
    def _adapt_rhythmic_decoder(
        decoder: nn.Module,
        old_state_dict: Dict[str, Any],
        old_genes: List[str],
        new_genes: List[str],
        old_n_harm: int,
        new_n_harm: int,
    ):
        """
        Special adapter for rhythmic decoder that handles both gene mapping and
        harmonic count changes.
        """
        new_state_dict = decoder.state_dict()
        gene_mapping = {gene: idx for idx, gene in enumerate(old_genes)}
        new_indices = [gene_mapping[gene] for gene in new_genes if gene in gene_mapping]

        param = old_state_dict["fourier_coefficients.weight"]
        old_weights = param  # Shape: (old_n_genes, old_n_harm*2 + 1)
        new_weights = new_state_dict["fourier_coefficients.weight"]

        # Copy old weights where possible, handling both gene and harmonic dimension changes
        for i, old_idx in enumerate(new_indices):
            n_harm_to_keep = min(new_n_harm * 2 + 1, old_weights.shape[1])
            new_weights[i, :n_harm_to_keep] = old_weights[old_idx, :n_harm_to_keep]

        new_state_dict["fourier_coefficients.weight"] = torch.nn.Parameter(new_weights)

        # Copy any other parameters
        for name, param in old_state_dict.items():
            if name in new_state_dict and name != "fourier_coefficients.weight":
                new_state_dict[name] = param

        decoder.load_state_dict(new_state_dict)

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
