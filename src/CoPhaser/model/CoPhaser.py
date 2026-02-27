import torch
import torch.nn as nn
from torch.distributions import Normal
from typing import List
from CoPhaser.model.neuralNet import NeuralNet
from CoPhaser.model.rhythmic_encoder_VAE import RhythmicEncoderVAE
from CoPhaser.model.rhythmic_decoder import RhythmicDecoder
from CoPhaser.model.bernoulli_encoder import RelaxedBernoulliEncoder
from CoPhaser.loss import Loss
from CoPhaser import plotting
import anndata
import pandas as pd
from scipy.sparse import csr_matrix
from CoPhaser import utils
import warnings
import pkg_resources
import numpy as np
import tqdm


class CoPhaser(nn.Module):
    """
    CoPhaser model combining a rhythmic space and a latent space for non-rhythmic variation, combined using
    a Fourier-based VAE rhythmic decoder, a context-dependant mean shift and amplitude scaling factor.
    """

    def __init__(
        self,
        rhythmic_gene_names: List[str],
        context_genes: List[str],
        n_latent: int = 10,
        n_harm: int = 3,
        rhythmic_decoder_to_all_genes: bool = True,
        use_mu_z_encoder: bool = True,
        use_lambda: bool = True,
        use_latent_z: bool = True,
        rhythmic_encoder_weights=[],
        z_range=20,
        lambda_range=2,
        rhythmic_z_scale=1,
        non_cycling_cells_angle=-1.5,
        force_context_genes_order=None,
    ):
        """
        CoPhaser model combining a rhythmic space and a latent space for non-rhythmic variation, combined using
        a Fourier-based VAE rhythmic decoder, a context-dependant mean shift and amplitude scaling factor.

        Parameters
        ----------
        rhythmic_gene_names: List[str]
            List of rhythmic genes used to learn the rhythmic space f.
        context_genes: List[str]
            List of context genes used to learn the context space z. Usually obtained using the 2000 most variable genes.
        n_latent: int
            Number of latent dimensions for the context space z. Default is 10.
        n_harm: int
            Number of harmonics used to decode the rhythmic contribution. Default is 3.
        rhythmic_decoder_to_all_genes: bool
            If True, the rhythmic decoder decodes to all context genes, otherwise only to rhythmic genes. Default is True.
        use_mu_z_encoder: bool
            If True, corrects the input of the rhythmic encoder using a mean shift dependent on the latent space z. Default is True.
        use_lambda: bool
            If True, use an amplitude scaling factor dependent on the latent space z. Default is True.
        use_latent_z: bool
            If True, use the latent space z in the generative model. Setting it to False transforms CoPhaser into a PCA-like model. Default is True.
        rhythmic_encoder_weights: List[np.array]
            Pretrained weights for the rhythmic encoder. Can be used to transfer learned weights that performs a 2d projection from rhythmic genes to a f space.
        z_range: float
            Range of the mean shifts derived from z used in the generative model. Default is 20.
        lambda_range: float
            Range of the amplitude scaling factor used in the generative model. Default is 2.
        rhythmic_z_scale: float
            Scaling factor applied to the mean shift derived from z for rhythmic genes. Default is 1.
        non_cycling_cells_angle: float
            Default angle assigned to non-cycling cells in the rhythmic encoder. Default is -1.5, useful when not all cells are cycling (cycling_status_prior of trainer < 1).
        """
        super().__init__()
        # store initialization variables
        self.rhythmic_gene_names = rhythmic_gene_names
        if len(self.rhythmic_gene_names) == 0:
            raise ValueError("At least one rhythmic gene must be provided.")
        self.adata_loaded = False
        self.batch_corrected = False
        self.rhythmic_decoder_is_pretrained = False
        self.n_harm = n_harm
        self.n_latent = n_latent
        self.use_mu_z_encoder = use_mu_z_encoder
        self.use_lambda = use_lambda
        self.rhythmic_decoder_to_all_genes = rhythmic_decoder_to_all_genes
        self.z_range = z_range
        self.lambda_range = lambda_range

        # modified by the trainer
        self.cycling_status_prior = False

        # set up variable and rhythmic genes
        if force_context_genes_order is not None:
            self.context_genes = force_context_genes_order
        else:
            self.context_genes = list(set(context_genes) | set(rhythmic_gene_names))
        n_variable_genes = len(self.context_genes)
        self.rhythmic_gene_indices = [
            self.context_genes.index(x) for x in self.rhythmic_gene_names
        ]
        self.non_rhythmic_gene_indices = [
            i for i in range(n_variable_genes) if i not in self.rhythmic_gene_indices
        ]

        self.rhythmic_z_scale = torch.ones(len(self.context_genes))
        self.rhythmic_z_scale[self.rhythmic_gene_indices] = rhythmic_z_scale

        # load rhythmic encoder weights if given
        if len(rhythmic_encoder_weights):
            rhythmic_encoder_weights = np.array(rhythmic_encoder_weights)
            assert rhythmic_encoder_weights.shape == (len(self.rhythmic_gene_names), 2)

        # rhythmic autoencoder
        self.rhythmic_encoder = RhythmicEncoderVAE(
            len(self.rhythmic_gene_names),
            self.n_harm,
            preload_weights=rhythmic_encoder_weights,
            default_angle=non_cycling_cells_angle,
        )
        if rhythmic_decoder_to_all_genes:
            n_outputs = len(self.context_genes)
        else:
            n_outputs = len(self.rhythmic_gene_names)

        self.rhythmic_decoder = RhythmicDecoder(n_outputs, self.n_harm)

        # variational autoencoder
        self.mean_encoder = NeuralNet(
            n_variable_genes,
            n_latent,
            "none",
        )
        self.var_encoder = NeuralNet(
            n_variable_genes,
            n_latent,
            "exp",
        )

        # bernoulli VAE
        self.cycling_status_encoder = RelaxedBernoulliEncoder(
            len(self.rhythmic_gene_names), n_latent
        )

        # NB/ZINB parameters
        self.log_theta_dispersion = torch.nn.Parameter(torch.randn(n_variable_genes))
        self.dropout_rate = torch.nn.Parameter(torch.tensor(-5.0))

        # delta Mean decoder
        self.decoder_non_rhythmic_contribution = NeuralNet(
            n_latent,
            n_variable_genes,
            "sigmoid",
        )
        self.decoder_raw_output = NeuralNet(n_latent, n_variable_genes, "none")
        self.mu_z_encoder = NeuralNet(
            n_latent,
            len(self.rhythmic_gene_names),
            "sigmoid",
        )

        self.lambda_decoder = NeuralNet(
            n_latent,
            1,
            "sigmoid",
        )

        self._use_latent_z = use_latent_z

    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        self.rhythmic_z_scale = self.rhythmic_z_scale.to(*args, **kwargs)

        if self.adata_loaded:
            self.mean_genes = self.mean_genes.to(*args, **kwargs)

        return self

    def add_batch(self, adata, batch_name):
        batches, _ = pd.factorize(adata.obs[batch_name])
        self.batch_keys = torch.tensor(
            batches,
            dtype=torch.int32,
        )
        self.n_batches = batches.max() + 1
        self.batch_corrected = True

    def _get_library_size_rhythmic_var_genes(
        self,
        adata: anndata.AnnData,
        layer_to_use: str,
        library_size_field=None,
    ):

        adata.layers[layer_to_use] = csr_matrix(adata.layers[layer_to_use])
        # Raise an error if some genes are not present in the anndata object
        if not set(self.context_genes).issubset(adata.var_names):
            raise ValueError(
                f"The genes {set(self.context_genes) - set(adata.var_names)} are not present in the anndata object."
            )
        # Extract library size
        if library_size_field is None:
            library_size = adata.layers[layer_to_use].sum(axis=1).A1
            library_size = torch.tensor(library_size, dtype=torch.float32)
        else:
            library_size = torch.tensor(
                adata.obs[library_size_field], dtype=torch.float32
            )
        # transform 0 library sizes to 1 to avoid NaNs
        library_size[library_size == 0] = 1.0
        # Extract rhythmic genes
        rhythmic_genes = (
            adata[:, self.rhythmic_gene_names].layers[layer_to_use].toarray()
        )
        rhythmic_genes = torch.tensor(rhythmic_genes, dtype=torch.float32)
        # variable genes
        variable_genes = adata[:, self.context_genes].layers[layer_to_use].toarray()
        variable_genes = torch.tensor(variable_genes, dtype=torch.float32)

        mean_genes = torch.log(
            (variable_genes.mean(axis=0) / library_size.mean())
        ).clamp_min(-50)

        return (library_size, rhythmic_genes, variable_genes, mean_genes)

    def load_anndata(
        self,
        adata: anndata.AnnData,
        layer_to_use: str,
        batch_name: str = None,
        library_size_field: str = None,
    ):
        """
        Load anndata object and extract the gene expression matrix, library size, for rhythmic genes and variable genes.

        If a batch_name is given the entropy is computed indep. for each batch, leading to longer training time.

        If the adata contains only a fraction of the genes use the library size field to give the total number of counts in each cell.

        ----------
        Parameters
        ----------
        adata: anndata object
            Anndata object containing the gene expression matrix.
        layer_to_use: str
            The layer to use for gene expression.
        batch: str
            The adata.obs column containing the batch, if any.
        """
        adata = adata.copy()
        (
            self.library_size,
            self.rhythmic_genes,
            self.variable_genes,
            self.mean_genes,
        ) = self._get_library_size_rhythmic_var_genes(
            adata, layer_to_use, library_size_field
        )

        # batch handling
        if batch_name is not None:
            self.add_batch(adata, batch_name)
        else:
            self.batch_corrected = False

        self.adata_loaded = True

    def prepare_context_training(
        self,
        adata: anndata.AnnData,
        layer_to_use: str,
        batch_name: str = None,
        freeze_mu_z_f_encoder=True,
    ):
        self.rhythmic_encoder.freeze_all_parameters()
        self.rhythmic_decoder.freeze_all_parameters()
        if freeze_mu_z_f_encoder:
            self.mu_z_encoder.freeze_all_parameters()
        self.load_anndata(adata, layer_to_use=layer_to_use, batch_name=batch_name)

    def load_circular_decoder(self, decoder_weights_path):
        self.rhythmic_decoder.load_rhythmic_weights(decoder_weights_path)
        self.rhythmic_decoder.freeze_rhythmic_weights()
        self.rhythmic_decoder_is_pretrained = True

    def latent_encoder_inference(
        self,
        x: torch.Tensor,
        library_size: torch.Tensor,
    ):

        x_norm = torch.log1p(x)
        qz_m = self.mean_encoder(x_norm)
        qz_v = self.var_encoder(x_norm)
        return qz_m, qz_v

    def generative(
        self,
        f: torch.Tensor,
        z: torch.Tensor,
        b_z: torch.Tensor,
        _lambda: torch.Tensor,
        library_size: torch.Tensor,
        f_g1: torch.Tensor,
    ):
        """
        Generative model of CoPhaser combining the rhythmic contribution decoded from f, the context mean shift decoded from z and the amplitude scaling factor _lambda.
        If b_z is 1, the rhythmic contribution is used, otherwise only the non-rhythmic contribution is used, and the rhythmic contribution is set to the value at f_g1.

        Parameters
        ----------
        f: torch.Tensor
            Rhythmic space representation of the cells.
        z: torch.Tensor
            Context space representation of the cells.
        b_z: torch.Tensor
            Bernoulli variable indicating if the cell is cycling (1) or not (0).
        _lambda: torch.Tensor
            Amplitude scaling factor for the rhythmic contribution.
            Identical to the one used in the rhythmic encoder, and therefore not recomputed here.
        library_size: torch.Tensor
            Library size of the cells.
        f_g1: torch.Tensor
            Values of the rhythmic decoder at the default angle (used for non-cycling cells).
        """
        # context mean shifts
        context_mean_shifts = (
            (
                self.decoder_non_rhythmic_contribution(z) * self.z_range
                - self.z_range / 2
            )
            * int(self._use_latent_z)
            * self.rhythmic_z_scale
        )

        # rhythmic contribution
        F = self.rhythmic_decoder(f)
        F_g1 = self.rhythmic_decoder(f_g1)
        F *= _lambda.view(-1, 1)
        F = F * b_z.view(-1, 1) + F_g1.view(1, -1) * (1 - b_z).view(-1, 1)

        # expand rhythmic term to all genes (zero padding)
        if self.rhythmic_decoder_to_all_genes:
            rhythmic_expanded = F
        else:
            rhythmic_expanded = torch.zeros(context_mean_shifts.size()).to(
                context_mean_shifts.device
            )
            rhythmic_expanded[:, self.rhythmic_gene_indices] = F

        library_size = library_size.unsqueeze(1)

        delta_mean = rhythmic_expanded + context_mean_shifts
        px_rate = torch.exp(delta_mean + self.mean_genes) * library_size

        # terms required for loss calculation
        log_theta_dispersion_expanded = self.log_theta_dispersion.expand(px_rate.size())
        theta_dispersion = torch.exp(log_theta_dispersion_expanded)

        return {
            "px_rate": px_rate,
            "theta_dispersion": theta_dispersion,
            "F": F,
            "Z": context_mean_shifts,
            "lambda": _lambda,
        }

    def forward(
        self,
        variable_genes: torch.Tensor,
        rhythmic_genes: torch.Tensor,
        library_size: torch.Tensor,
        epoch: int = -1,
        use_max_posterior: bool = False,
    ):
        """
        Forward pass of CoPhaser model.

        Parameters
        ----------
        variable_genes: torch.Tensor
            Gene expression matrix of the context genes.
        rhythmic_genes: torch.Tensor
            Gene expression matrix of the rhythmic genes.
        library_size: torch.Tensor
            Library size of the cells.
        epoch: int
            Current epoch of training. Used to adjust the concentration of the rhythmic encoder, and to enable the use of mu_z and _lambda after a certain number of epochs.
        use_max_posterior: bool
            If True, use the maximum of the variational posterior instead of sampling. Used for inference.
        """

        # cell specific latent space
        qz_m, qz_v = self.latent_encoder_inference(variable_genes, library_size)
        if not use_max_posterior:
            z = Normal(qz_m, torch.sqrt(qz_v)).rsample() * int(self._use_latent_z)
        else:
            z = qz_m * int(self._use_latent_z)

        # mean shift and amplitude scaling factor from z, for the rhythmic encoder
        mu_z = (self.mu_z_encoder(z) * self.z_range - self.z_range / 2) * int(
            self._use_latent_z
        )

        if self.use_lambda and (epoch > 100 or epoch == -1):
            _lambda = 2 ** (self.lambda_decoder(z) * 2 - 1)
            lambda_enc = _lambda
        else:
            _lambda = torch.ones_like(qz_m[:, 0])
            lambda_enc = torch.ones_like(qz_m[:, 0])

        # rhythmic space
        f, theta, mu, kappa, x_projected, f_g1 = self.rhythmic_encoder.forward(
            rhythmic_genes,
            library_size,
            mu_z,
            _lambda,
            self.use_mu_z_encoder and (epoch > 50 or epoch == -1),
            epoch,
            use_max_posterior=use_max_posterior,
        )
        if (epoch > 10) and self.cycling_status_prior < 1:
            b_z, logits = self.cycling_status_encoder.forward(
                rhythmic_genes, z, epoch, library_size
            )
        elif use_max_posterior and self.cycling_status_prior < 1:
            b_z, logits = self.cycling_status_encoder.forward(
                rhythmic_genes, z, 200, library_size
            )  # TODO: modify, the 200 is ugly
        else:
            b_z = torch.ones_like(theta)
            logits = torch.full_like(theta, np.inf)

        generative_outputs = self.generative(f, z, b_z, _lambda, library_size, f_g1)
        return generative_outputs, {
            "qz_m": qz_m,
            "qz_v": qz_v,
            "f": f,
            "z": z,
            "b_z": b_z,
            "cycling_logits": logits,
            "theta": theta,
            "mu_theta": mu,
            "mu_z": mu_z,
            "kappa_theta": kappa,
            "lambda_enc": lambda_enc,
            "x_projected": x_projected,
            "f_g1": f_g1,
        }

    def freeze_genes_rhythmic_VAE(self, gene_names):
        gene_indices_input = [g in self.rhythmic_gene_names for g in gene_names]
        self.rhythmic_encoder.freeze_weights_genes(gene_indices_input)
        if self.rhythmic_decoder_to_all_genes:
            gene_indices_output = [g in self.context_genes for g in gene_names]
            self.rhythmic_decoder.freeze_weights_genes(gene_indices_output)
        else:
            self.rhythmic_decoder.freeze_weights_genes(gene_indices_input)

    def unfreeze_genes_rhythmic_VAE(self, gene_names):
        gene_indices_input = [g in self.rhythmic_gene_names for g in gene_names]
        self.rhythmic_encoder.unfreeze_weights_genes(gene_indices_input)
        if self.rhythmic_decoder_to_all_genes:
            gene_indices_output = [g in self.context_genes for g in gene_names]
            self.rhythmic_decoder.unfreeze_weights_genes(gene_indices_output)
        else:
            self.rhythmic_decoder.unfreeze_weights_genes(gene_indices_input)

    def _get_gene_annotation(self):
        CCG_path = pkg_resources.resource_filename(
            __name__, f"../resources/CCG_annotated.csv"
        )
        df_gene = pd.read_csv(CCG_path, index_col="Primary name")["Peaktime"]
        model_gene_names = (
            self.context_genes
            if self.rhythmic_decoder_to_all_genes
            else self.rhythmic_gene_names
        )
        mask = np.isin([g.upper() for g in model_gene_names], df_gene.index)
        model_gene_names = np.array(model_gene_names)[mask]
        categories = df_gene[[g.upper() for g in model_gene_names]]
        return model_gene_names, categories, mask

    def _get_phase_amplitude_mean(self, mask, categories):
        fourier_coeffs = (
            self.rhythmic_decoder.fourier_coefficients.weight.detach()
            .cpu()
            .numpy()[mask][:, [0, 1]]
        )
        phases = np.arctan2(fourier_coeffs[:, 1], fourier_coeffs[:, 0])
        amplitudes = np.sqrt(fourier_coeffs[:, 0] ** 2 + fourier_coeffs[:, 1] ** 2)
        complex_values = amplitudes * np.exp(1j * phases)

        mean_phase = {}
        for cat in np.unique(categories):
            mask = categories == cat
            # mean phase weighted by amplitude
            mean_phase[cat] = np.angle(np.sum(complex_values[mask]))

        return mean_phase, amplitudes, phases

    @staticmethod
    def shift_phases(phases, origin, direction, offset=0):
        return utils.normalize_angles(
            utils.normalize_angles(phases - origin) * direction + offset
        )

    def _get_origine_direction(self, plot=True, offset=1 / 4 * np.pi):
        _, categories, mask = self._get_gene_annotation()
        mean_phase, amplitudes, phases = self._get_phase_amplitude_mean(
            mask, categories
        )
        _, _, best_direction = utils.best_order(mean_phase)
        if plot:
            mean_phase_shifted = {
                key: self.shift_phases(
                    value, mean_phase["G1/S"], best_direction, offset=offset
                )
                for key, value in mean_phase.items()
            }
            plotting.plot_phase_distribution(
                mean_phase_shifted,
                amplitudes,
                self.shift_phases(
                    phases, mean_phase["G1/S"], best_direction, offset=offset
                ),
                categories,
            )
        if not "G1/S" in categories.values:
            warnings.warn(
                "The 'G1/S' phase is required to set the origin, but no genes annotated as G1/S were used."
            )
            return 0, best_direction
        return mean_phase["G1/S"], best_direction

    def plot_fourier_coefficients(self, plot_all_genes=False):
        ab_coefficients = (
            self.rhythmic_decoder.fourier_coefficients.weight.detach().cpu().numpy()
        )
        if plot_all_genes and self.rhythmic_decoder_to_all_genes:
            gene_names = self.context_genes
        elif plot_all_genes:
            gene_names = self.rhythmic_gene_names
        else:
            ab_coefficients = ab_coefficients[self.rhythmic_gene_indices]
            gene_names = self.rhythmic_gene_names
        plotting.plot_fourrier_coefficients(ab_coefficients, gene_names)

    def orient_align_pseudotimes(self, thetas, offset=1 / 4 * np.pi, plot=True):
        origin, direction = self._get_origine_direction(plot=plot, offset=offset)
        # set the mean g1/s at offset
        thetas = self.shift_phases(thetas, origin, direction, offset=offset)
        return thetas

    def infer_pseudotimes(
        self,
        adata: anndata.AnnData,
        layer_to_use: str,
        isCellCycle=True,
        offset=1 / 4 * np.pi,
        plot=True,
    ):
        """
        Infer the pseudotimes of the cells in the adata. If isCellCycle, fixes the origin
        of the phase such as G1/S transition arrives at the given offset.
        """
        adata = adata.copy()
        library_size, rhythmic_genes, variable_genes, mean_genes = (
            self._get_library_size_rhythmic_var_genes(adata, layer_to_use)
        )
        self.to(device="cpu")
        with torch.no_grad():
            _, space_outputs = self.forward(
                variable_genes, rhythmic_genes, library_size, use_max_posterior=True
            )
        thetas = space_outputs["theta"]
        if not isCellCycle:
            return thetas
        return self.orient_align_pseudotimes(thetas, offset=offset, plot=plot)

    def get_posterior(
        self,
        adata: anndata.AnnData,
        layer_to_use: str,
        n_points: int,
        use_non_rhythmic=False,
        normalize=True,
        device="cuda",
    ):
        """
        Method to compute the exact posterior over a grid of angles for each cell.

        Parameters
        ----------
        adata: anndata.AnnData
            Anndata object containing the gene expression matrix.
        layer_to_use: str
            The layer to use for gene expression.
        n_points: int
            Number of points in the grid to compute the posterior.
        use_non_rhythmic: bool
            If True, includes the non-rhythmic genes in the likelihood computation.
        normalize: bool
            If True, normalizes the posterior to sum to 1.
        device: str
            Device to use for computation.

        Returns
        -------
        posterior: torch.Tensor
            Posterior over the grid of angles for each cell.
        theta_grid: torch.Tensor
            Grid of angles used for the posterior computation.
        """
        adata = adata.copy()
        self.to(device)
        library_size, rhythmic_genes, variable_genes, mean_genes = (
            self._get_library_size_rhythmic_var_genes(adata, layer_to_use)
        )
        library_size = library_size.to(device)
        variable_genes = variable_genes.to(device)
        rhythmic_genes = rhythmic_genes.to(device)

        with torch.no_grad():
            # use the maximum of the variational posterior of z
            generative, space = self.forward(
                variable_genes, rhythmic_genes, library_size, -1, use_max_posterior=True
            )
            z = space["z"]
            b_z = space["b_z"]
            _lambda = generative["lambda"]
            f_g1 = space["f_g1"]
            res = torch.zeros((z.shape[0], n_points), device=device)
            theta_grid = np.linspace(-np.pi, np.pi, n_points)
            for i, theta in tqdm.tqdm(enumerate(theta_grid), total=n_points):
                thetas = torch.full((z.shape[0],), theta, device=device)
                f = self.rhythmic_encoder.fourier_basis_expansion(thetas)
                pred = self.generative(f, z, b_z, _lambda, library_size, f_g1)
                log_lik = Loss.log_likelihood_NB_weighted(
                    x=variable_genes,
                    px_rate=pred["px_rate"],
                    theta_dispersion=pred["theta_dispersion"],
                    rhythmic_indices=self.rhythmic_gene_indices,
                    non_rhythmic_indices=self.non_rhythmic_gene_indices,
                    rhythmic_likelihood_weight=1,
                    non_rhythmic_likelihood_weight=int(use_non_rhythmic),
                )
                res[:, i] = log_lik
        theta_grid = self.orient_align_pseudotimes(theta_grid)
        if normalize:
            # numerical stability
            res_max, _ = res.max(dim=1, keepdim=True)
            res_stable = res - res_max
            likelihoods = torch.exp(res_stable)
            posterior = likelihoods / likelihoods.sum(dim=1, keepdim=True)
            return posterior.to("cpu"), theta_grid
        return res.to("cpu"), theta_grid

    @staticmethod
    def get_variational_posterior(thetas, kappas, n_points=100):
        """
        Compute the variational posterior over a grid of theta values given the parameters of the von Mises distribution.

        Parameters
        ----------
        thetas: torch.Tensor
            Mean directions of the von Mises distributions for each cell. Shape [n_cells].
        kappas: torch.Tensor
            Concentration parameters of the von Mises distributions for each cell. Shape [n_cells].
        n_points: int
            Number of points in the grid to compute the posterior.

        Returns
        -------
        variational_posteriors: torch.Tensor
            Variational posterior over the grid of theta values for each cell. Shape [n_cells, n_points].
        theta_grid: torch.Tensor
            Grid of theta values used for the posterior computation. Shape [n_points].
        """
        thetas = torch.tensor(thetas).clone().detach()
        kappas = torch.tensor(kappas).clone().detach()

        # Grid of theta values: shape [n_points]
        theta_grid = torch.linspace(-np.pi, np.pi, n_points, device=thetas.device)

        # Expand shapes for broadcasting
        mu = thetas.unsqueeze(1)
        theta = theta_grid.unsqueeze(0)

        angle_diff = theta - mu

        # Compute log probs: log P = kappa * log(1 + cos(θ - μ))
        cos_diff = torch.cos(angle_diff)
        log_probs = kappas * torch.log1p(cos_diff)

        # Normalize to get posterior
        log_probs = log_probs - log_probs.max(dim=1, keepdim=True).values  # stability
        probs = torch.exp(log_probs)
        variational_posteriors = probs / probs.sum(
            dim=1, keepdim=True
        )  # [n_cells, n_points]
        return variational_posteriors, theta_grid.flatten()

    def get_outputs(self):
        if not self.adata_loaded:
            raise RuntimeError(
                "Cannot get model output: no data has been loaded. Call load_anndata() first."
            )
        self.eval()
        with torch.no_grad():
            res = self.forward(
                self.variable_genes,
                self.rhythmic_genes,
                self.library_size,
                use_max_posterior=True,
            )
        return res

    def save(self, file_path: str):
        """Saves the model state and metadata."""
        init_params = {
            "rhythmic_gene_names": self.rhythmic_gene_names,
            "context_genes": self.context_genes,
            "n_latent": self.n_latent,
            "n_harm": self.n_harm,
            "rhythmic_decoder_to_all_genes": self.rhythmic_decoder_to_all_genes,
            "use_mu_z_encoder": self.use_mu_z_encoder,
            "use_lambda": self.use_lambda,
            "use_latent_z": self._use_latent_z,
            "rhythmic_encoder_weights": [],
            "z_range": self.z_range,
            "lambda_range": self.lambda_range,
            "rhythmic_z_scale": self.rhythmic_z_scale[
                self.rhythmic_gene_indices[0]
            ].item(),
            "non_cycling_cells_angle": self.rhythmic_encoder.default_angle,
            "force_context_genes_order": self.context_genes,
            "cycling_status_prior": (
                self.cycling_status_prior
                if hasattr(self, "cycling_status_prior")
                else 1
            ),
        }
        torch.save(
            {"state_dict": self.state_dict(), "init_params": init_params}, file_path
        )

    @classmethod
    def load(cls, file_path: str):
        """
        Loads the model state and metadata from a file.
        """
        checkpoint = torch.load(file_path, weights_only=False)
        cycling_status_prior = float(checkpoint["init_params"]["cycling_status_prior"])
        # drop cycling_status_prior, which is added later during training
        del checkpoint["init_params"]["cycling_status_prior"]
        if "init_params" in checkpoint:
            model = cls(**checkpoint["init_params"])
            model.load_state_dict(checkpoint["state_dict"])
            model.cycling_status_prior = cycling_status_prior
            return model
        else:
            raise ValueError(
                "Cannot load model: init_params not found in checkpoint. Ensure the model was saved with the save method."
            )
