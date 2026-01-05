from torch.distributions import NegativeBinomial, Normal, Poisson
from torch.distributions import kl_divergence as kl
from typing import Literal
import torch


class Loss:
    """Loss functions for CoPhaser."""

    @staticmethod
    def log_likelihood_ZINB(x, theta_dispersion, nb_logits, dropout_prob):
        zero_mask = (x == 0).float()
        nonzero_mask = (x > 0).float()
        nb_case_log_lik = NegativeBinomial(
            total_count=theta_dispersion, logits=nb_logits
        ).log_prob(x)
        log_lik_nonzero = ((1 - dropout_prob) * nb_case_log_lik * nonzero_mask).sum(
            dim=-1
        )
        log_lik_zero = (
            dropout_prob + (1 - dropout_prob) * torch.exp(nb_case_log_lik)
        ).log() * zero_mask
        return log_lik_nonzero + log_lik_zero.sum(dim=-1)

    @staticmethod
    def log_likelihood_ZINB_weighted(
        x,
        px_rate,
        theta_dispersion,
        dropout_prob,
        rhythmic_indices,
        non_rhythmic_indices,
        rhythmic_likelihood_weight,
        non_rhythmic_likelihood_weight,
    ):
        nb_logits = (px_rate + 1e-4).log() - (theta_dispersion + 1e-4).log()
        # rhythmic_genes
        nb_logits_rhythmic = nb_logits[:, rhythmic_indices]
        theta_dispersion_rhythmic = theta_dispersion[:, rhythmic_indices]
        x_rhythmic = x[:, rhythmic_indices]

        # non rhythmic genes
        nb_logits_non_rhythmic = nb_logits[:, non_rhythmic_indices]
        theta_dispersion_non_rhythmic = theta_dispersion[:, non_rhythmic_indices]
        x_non_rhythmic = x[:, non_rhythmic_indices]

        # log likelihoods
        log_lik_rhythmic = Loss.log_likelihood_ZINB(
            x=x_rhythmic,
            theta_dispersion=theta_dispersion_rhythmic,
            nb_logits=nb_logits_rhythmic,
            dropout_prob=dropout_prob,
        )
        log_lik_non_rhythmic = Loss.log_likelihood_ZINB(
            x=x_non_rhythmic,
            theta_dispersion=theta_dispersion_non_rhythmic,
            nb_logits=nb_logits_non_rhythmic,
            dropout_prob=dropout_prob,
        )
        return (
            rhythmic_likelihood_weight * log_lik_rhythmic
            + non_rhythmic_likelihood_weight * log_lik_non_rhythmic
        )

    @staticmethod
    def log_likelihood_NB(
        x,
        theta_dispersion,
        nb_logits,
        genes_weights: torch.Tensor = 1,
    ):
        log_lik = NegativeBinomial(
            total_count=theta_dispersion, logits=nb_logits
        ).log_prob(x)
        log_lik *= genes_weights
        return log_lik.sum(dim=-1)

    @staticmethod
    def log_likelihood_NB_weighted(
        x,
        px_rate,
        theta_dispersion,
        rhythmic_indices,
        non_rhythmic_indices,
        rhythmic_likelihood_weight,
        non_rhythmic_likelihood_weight,
    ):
        nb_logits = (px_rate + 1e-4).log() - (theta_dispersion + 1e-4).log()
        # rhythmic_genes
        nb_logits_rhythmic = nb_logits[:, rhythmic_indices]
        theta_dispersion_rhythmic = theta_dispersion[:, rhythmic_indices]
        x_rhythmic = x[:, rhythmic_indices]

        # non rhythmic genes
        nb_logits_non_rhythmic = nb_logits[:, non_rhythmic_indices]
        theta_dispersion_non_rhythmic = theta_dispersion[:, non_rhythmic_indices]
        x_non_rhythmic = x[:, non_rhythmic_indices]

        # log likelihoods
        log_lik_rhythmic = Loss.log_likelihood_NB(
            x=x_rhythmic,
            theta_dispersion=theta_dispersion_rhythmic,
            nb_logits=nb_logits_rhythmic,
        )
        log_lik_non_rhythmic = Loss.log_likelihood_NB(
            x=x_non_rhythmic,
            theta_dispersion=theta_dispersion_non_rhythmic,
            nb_logits=nb_logits_non_rhythmic,
        )
        return (
            rhythmic_likelihood_weight * log_lik_rhythmic
            + non_rhythmic_likelihood_weight * log_lik_non_rhythmic
        )

    @staticmethod
    def log_likelihood_poisson(
        x,
        px_rate,
        rhythmic_indices,
        non_rhythmic_indices,
        rhythmic_likelihood_weight,
        non_rhythmic_likelihood_weight,
    ):
        px_rate += 1e-6
        log_like_rhythmic = (
            Poisson(px_rate[:, rhythmic_indices])
            .log_prob(x[:, rhythmic_indices])
            .sum(dim=-1)
        )
        log_like_non_rhythmic = (
            Poisson(px_rate[:, non_rhythmic_indices])
            .log_prob(x[:, non_rhythmic_indices])
            .sum(dim=-1)
        )
        return (
            rhythmic_likelihood_weight * log_like_rhythmic
            + non_rhythmic_likelihood_weight * log_like_non_rhythmic
        )

    @staticmethod
    def kl_divergence(qz_m, qz_v):
        prior = Normal(torch.zeros_like(qz_m), torch.ones_like(qz_v))
        posterior = Normal(qz_m, torch.sqrt(qz_v))
        return kl(posterior, prior).sum(dim=1)

    @staticmethod
    def circular_batch_prior_loss(
        thetas: torch.Tensor,  # shape [N], angles in radians
        batch_keys: torch.Tensor,  # shape [N], int64 in [0, B-1]
        n_batches=1,
        compute_mean_diff=False,
        eps: float = 1e-8,
    ) -> torch.Tensor:
        """
        Adds a prior term encouraging within-batch concentration of thetas.

        Modes
        -----

        Surrogate encouraging high mean resultant length R̄_b:
        For each batch b, R̄_b = ||sum_j u_j|| / n_b where u_j = [cos θ_j, sin θ_j].
        loss = mean_b [ kappa_b * (1 - R̄_b) ]

        kappa broadcasting
        ------------------
        - scalar: same kappa for all samples
        - shape [B]: per-batch kappa
        - shape [N]: per-sample kappa

        Returns
        -------
        A scalar tensor suitable to add to your loss (smaller is better).
        """
        device = thetas.device

        sin_t = torch.sin(thetas)
        cos_t = torch.cos(thetas)

        # Per-batch sums using differentiable scatter_add_
        sum_sin = torch.zeros(n_batches, dtype=thetas.dtype, device=device)
        sum_cos = torch.zeros(n_batches, dtype=thetas.dtype, device=device)
        counts = torch.zeros(n_batches, dtype=thetas.dtype, device=device)

        batch_keys = batch_keys.to(torch.long)

        sum_sin.scatter_add_(0, batch_keys, sin_t)
        sum_cos.scatter_add_(0, batch_keys, cos_t)
        counts.scatter_add_(0, batch_keys, torch.ones_like(thetas, dtype=thetas.dtype))
        mask = counts != 0

        # Unit resultant length per batch: R̄_b = ||[sum_cos, sum_sin]|| / counts
        R = torch.sqrt(sum_cos * sum_cos + sum_sin * sum_sin) / (counts)  # [B]

        R *= mask  # np.inf*0=nan
        coherence_loss = R.nanmean()

        if compute_mean_diff:
            mean_vecs = torch.stack([sum_cos, sum_sin], dim=-1) / (
                counts.unsqueeze(-1) + eps
            )
            mean_vecs = mean_vecs / (
                mean_vecs.norm(dim=-1, keepdim=True) + eps
            )  # [B, 2]

            # pairwise cosine similarities between batch means
            sims = mean_vecs @ mean_vecs.T  # [B, B]
            mask = ~torch.eye(n_batches, dtype=torch.bool, device=device)
            pairwise_sims = sims[mask]  # exclude self-similarity

            # penalize similarity (prefer orthogonal/opposite means)
            repulsion_loss = (pairwise_sims**2).mean()
            return coherence_loss, repulsion_loss
        return coherence_loss, torch.tensor(0.0, device=device)

    @staticmethod
    def circular_entropy_loss(
        theta,
        n_bins=30,
        entropy_weight=10.0,
        batch_keys=None,
        entropy_per_batch=False,
        device="cuda",
    ):
        """
        Computes the circular entropy loss for theta.

        Parameters:
            theta (torch.Tensor): Tensor of angles (radians) with shape [batch_size].
            n_bins (int): Number of bins for the circular histogram.
            entropy_weight (float): Weight for the entropy term in the loss.
            batch_keys (torch.Tensor): Tensor of batch keys for grouping, shape [batch_size].
            entropy_per_batch (bool): If True, calculate entropy per batch group.
            device (str): Device to perform computations on ("cuda" or "cpu").

        Returns:
            torch.Tensor: Weighted negative entropy loss.
        """
        # Create bin edges and move to the correct device
        bin_edges = torch.linspace(-torch.pi, torch.pi, n_bins + 1, device=device)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

        # Compute distances and apply Gaussian kernel
        bin_width = (2 * torch.pi) / n_bins
        bin_centers_expanded = bin_centers.unsqueeze(0)  # Shape [1, n_bins]

        def compute_entropy(theta_group):
            # Compute distances for the group
            diff = (
                torch.remainder(
                    theta_group.unsqueeze(-1).to(device)
                    - bin_centers_expanded
                    + torch.pi,
                    2 * torch.pi,
                )
                - torch.pi
            )
            kernel = torch.exp(-0.5 * (diff / bin_width) ** 2)

            # Approximate histogram and normalize to probabilities
            hist = kernel.sum(dim=0) + 1e-10  # Adding epsilon to avoid division by zero
            prob = hist / hist.sum()

            # Compute entropy
            return -torch.sum(prob * torch.log(prob + 1e-10))

        if entropy_per_batch and batch_keys is not None:
            # Compute entropy per batch group
            unique_batches = batch_keys.unique(dim=0)
            entropies = []
            for batch in unique_batches:
                batch_mask = batch_keys == batch
                batch_entropy = compute_entropy(theta[batch_mask])
                entropies.append(batch_entropy)
            entropy = torch.stack(entropies).mean()  # Average entropies across batches
        else:
            # Compute entropy for the entire dataset
            entropy = compute_entropy(theta)

        return -entropy_weight * entropy

    @staticmethod
    def mutual_information_loss(
        MINE_model, f, z, MI_weight: float, to_detach: Literal["f", "z", "none"]
    ):
        if to_detach == "f":
            f = f.detach()
        elif to_detach == "z":
            z = z.detach()
        mi_loss = MINE_model.mutual_information_loss(f, z)
        mi_loss *= MI_weight
        return mi_loss

    @staticmethod
    def compute_loss(
        model,
        x,
        epoch,
        generative_outputs,
        inference_outputs,
        MINE_model,
        entropy_loss_weight,
        entropy_per_batch,
        L2_Z_decoder_loss_weight,
        MI_weight,
        rhythmic_likelihood_weight,
        non_rhythmic_likelihood_weight,
        closed_circle_weight,
        noise_model: Literal["poisson", "NB", "ZINB"],
        beta_kl_f,
        beta_kl_cycling_status,
        cycling_status_prior,
        batch_keys=None,
        MI_detach: Literal["f", "z", "none"] = "f",
    ):
        px_rate = generative_outputs["px_rate"]
        theta_dispersion = generative_outputs["theta_dispersion"]
        qz_m = inference_outputs["qz_m"]
        qz_v = inference_outputs["qz_v"]
        theta = inference_outputs["theta"]
        mu_theta = torch.arctan2(
            inference_outputs["mu_theta"][:, 1], inference_outputs["mu_theta"][:, 0]
        )
        dropout_prob = torch.clamp(
            torch.sigmoid(model.dropout_rate), min=0.01, max=0.99
        )

        # Likelihood of the data (Negative Binomial)
        if noise_model == "ZINB":
            log_lik = Loss.log_likelihood_ZINB_weighted(
                x=x,
                px_rate=px_rate,
                theta_dispersion=theta_dispersion,
                dropout_prob=dropout_prob,
                rhythmic_indices=model.rhythmic_gene_indices,
                non_rhythmic_indices=model.non_rhythmic_gene_indices,
                rhythmic_likelihood_weight=rhythmic_likelihood_weight,
                non_rhythmic_likelihood_weight=non_rhythmic_likelihood_weight,
            )
        elif noise_model == "NB":
            log_lik = Loss.log_likelihood_NB_weighted(
                x=x,
                px_rate=px_rate,
                theta_dispersion=theta_dispersion,
                rhythmic_indices=model.rhythmic_gene_indices,
                non_rhythmic_indices=model.non_rhythmic_gene_indices,
                rhythmic_likelihood_weight=rhythmic_likelihood_weight,
                non_rhythmic_likelihood_weight=non_rhythmic_likelihood_weight,
            )

        elif noise_model == "poisson":
            """
            log_lik = Loss.log_likelihood_poisson(
                x=x,
                px_rate=px_rate,
                rhythmic_indices=model.rhythmic_gene_indices,
                non_rhythmic_indices=model.non_rhythmic_gene_indices,
                rhythmic_likelihood_weight=rhythmic_likelihood_weight,
                non_rhythmic_likelihood_weight=non_rhythmic_likelihood_weight,
            )
            """
            px_rate += 1e-6
            log_like_rhythmic = (
                Poisson(px_rate[:, model.rhythmic_gene_indices])
                .log_prob(x[:, model.rhythmic_gene_indices])
                .sum(dim=-1)
            )
            log_like_non_rhythmic = (
                Poisson(px_rate[:, model.non_rhythmic_gene_indices])
                .log_prob(x[:, model.non_rhythmic_gene_indices])
                .sum(dim=-1)
            )
            log_lik = (
                rhythmic_likelihood_weight * log_like_rhythmic
                + non_rhythmic_likelihood_weight * log_like_non_rhythmic
            )
        else:
            raise NotImplementedError(
                f"{noise_model} is not supported as a noise model"
            )

        # KL divergence
        prior_dist = Normal(torch.zeros_like(qz_m), torch.ones_like(qz_v))
        var_post_dist = Normal(qz_m, torch.sqrt(qz_v))
        kl_divergence_z = kl(var_post_dist, prior_dist).sum(dim=1)

        kl_divergence_f = model.rhythmic_encoder.kl_divergence(
            inference_outputs["mu_theta"], inference_outputs["kappa_theta"]
        ).flatten()

        logits = inference_outputs["cycling_logits"]
        if epoch > 15:
            # remove cells from cycle metrics only when the bVAE is trained a bit
            cycling_cells = inference_outputs["b_z"] > 0.5
        else:
            cycling_cells = torch.ones_like(inference_outputs["b_z"], dtype=bool)
        if cycling_cells.sum() < 2:
            # avoid bug when no cells are classified as cycling
            cycling_cells[0:2] = True
        if not torch.isinf(logits[0]):
            kl_divergence_cycling_status = model.cycling_status_encoder.kl_divergence(
                logits, cycling_status_prior
            )
        else:
            kl_divergence_cycling_status = 0

        # remove non-cycling cells from theta-related losses
        kl_divergence_f[~cycling_cells] = 0
        # ELBO (Evidence Lower Bound)
        beta_z = 1  # min(1, epoch**3 / 2500)
        elbo = (
            log_lik
            - kl_divergence_z * beta_z
            - kl_divergence_f * beta_kl_f
            - kl_divergence_cycling_status * beta_kl_cycling_status
        )
        loss = torch.mean(-elbo)
        loss_dict = {"elbo_loss": loss.item()}
        loss_dict["kl_div_f"] = kl_divergence_f[cycling_cells].mean()
        loss_dict["kl_div_z"] = kl_divergence_z.mean()
        loss_dict["fraction_cycling_cells"] = cycling_cells.float().mean().item()

        # L2 loss decoder of Z
        l2_loss = torch.mean(
            torch.norm(
                model.decoder_non_rhythmic_contribution.neural_net[2].weight, dim=1
            )
        )
        l2_loss *= L2_Z_decoder_loss_weight
        loss += l2_loss
        loss_dict["l2_px_rate"] = l2_loss.item()

        # circular entropy loss
        H_loss = Loss.circular_entropy_loss(
            mu_theta[cycling_cells],
            entropy_weight=entropy_loss_weight,
            batch_keys=batch_keys[cycling_cells] if batch_keys is not None else None,
            entropy_per_batch=entropy_per_batch,
            device=theta.device,
        )
        loss += H_loss
        loss_dict["entropy_loss_unweighted"] = H_loss.item() / entropy_loss_weight
        loss_dict["entropy_loss"] = H_loss.item()

        radii = torch.norm(inference_outputs["x_projected"][cycling_cells], dim=1)
        radial_variance_loss = torch.var(radii) * closed_circle_weight
        circle_deviation_loss = torch.mean((radii - 1) ** 2) * closed_circle_weight

        loss_dict["radial_variance"] = radial_variance_loss.item()
        loss_dict["radius"] = circle_deviation_loss.item()
        loss += radial_variance_loss
        loss += circle_deviation_loss

        # L1 loss for mu_z
        L1_Z = inference_outputs["mu_z"].flatten().abs().mean() * 0  # 10
        loss_dict["L1_mu_z"] = L1_Z.item()
        loss += L1_Z

        # mutual information loss, let the MINE train first
        MI_loss = Loss.mutual_information_loss(
            MINE_model,
            inference_outputs["theta"][cycling_cells],
            inference_outputs["z"][cycling_cells],
            MI_weight,
            to_detach=MI_detach,
        ) * min(epoch / 20, 1)
        loss += MI_loss
        loss_dict["MI_loss"] = MI_loss.item() / MI_weight

        loss_dict["total_loss"] = loss
        return loss_dict
