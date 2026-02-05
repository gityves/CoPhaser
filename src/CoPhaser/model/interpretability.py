import torch
import numpy as np
from captum.attr import IntegratedGradients
from torch.utils.data import DataLoader
from CoPhaser.model import CoPhaser


class EncoderMeanWrapper(torch.nn.Module):
    def __init__(self, encoder, library_size, mu_Z, amp_z, use_mu_z_encoder):
        super().__init__()
        self.encoder = encoder
        self.library_size = library_size
        self.mu_Z = mu_Z
        self.amp_z = amp_z
        self.use_mu_z_encoder = use_mu_z_encoder

    def forward(self, x):
        lib = self.library_size.unsqueeze(-1)
        x_norm = self.encoder.normalize_expression(
            x, lib, self.mu_Z, self.amp_z, self.use_mu_z_encoder
        )
        mu, _ = self.encoder._mean_forward(x_norm)
        return mu  # shape: (batch_size, 2)


class SpaceWrapper(torch.nn.Module):
    def __init__(
        self,
        model: CoPhaser,
        lib_batch,
        space_field=None,
        generative_field=None,
        index_field=None,
    ):
        super().__init__()
        self.model = model
        self.space_field = space_field
        self.generative_field = generative_field
        self.index_field = index_field
        self.lib_batch = lib_batch

    def forward(
        self,
        variable_genes: torch.Tensor,
        epoch=200,
    ):
        rhythmic_genes = variable_genes[:, self.model.rhythmic_gene_indices]

        generative_outputs, space_outputs = self.model.forward(
            variable_genes, rhythmic_genes, self.lib_batch, epoch
        )
        if self.generative_field is not None:
            if self.index_field is not None:
                return generative_outputs[self.generative_field][:, self.index_field]
            return generative_outputs[self.generative_field]
        if self.index_field is not None:
            return space_outputs[self.space_field][:, self.index_field]
        return space_outputs[self.space_field]


def compute_feature_importance(
    model: CoPhaser,
    space_field=None,
    generative_field=None,
    index_field=None,
    batch_size=1024,
    device="cuda",
    mu_dim=1,
    mask_cells=None,
):
    """
    Computes average feature importance for the field provided.

    Returns:
        attributions: np.ndarray of shape (mu_dim, n_input)
    """
    if not model.rhythmic_decoder_to_all_genes:
        raise NotImplementedError()
    if space_field is None and generative_field is None:
        raise ValueError("Either space_field or generative_field must be provided.")
    model = model.to(device)
    x_variable = model.variable_genes.to(device)
    means = x_variable.mean(axis=0)
    means = means.unsqueeze(0).expand(batch_size, -1)
    library_size = model.library_size.to(device)

    n_input = x_variable.shape[1]
    running_attr = torch.zeros((mu_dim, n_input), device=device)
    if mask_cells is not None:
        x_variable = x_variable[mask_cells]
        library_size = library_size[mask_cells]
    dataloader = DataLoader(torch.arange(len(x_variable)), batch_size=batch_size)

    for batch_idxs in dataloader:

        x_variable_batch = x_variable[batch_idxs]
        lib_batch = library_size[batch_idxs]
        wrapper = SpaceWrapper(
            model,
            lib_batch,
            space_field=space_field,
            generative_field=generative_field,
            index_field=index_field,
        ).to(device)
        ig = IntegratedGradients(wrapper)
        for i in range(mu_dim):
            attributions = ig.attribute(
                inputs=x_variable_batch,
                baselines=means[
                    : len(batch_idxs)
                ],  # torch.zeros_like(x_variable_batch),
                target=i if mu_dim > 1 else None,
                internal_batch_size=len(batch_idxs),
            )  # shape: (batch_size, n_input)
            running_attr[i] += attributions.sum(dim=0)

    avg_attr = running_attr / len(x_variable_batch)  # shape: (mu_dim, n_input)
    return avg_attr.detach().cpu().numpy()


def compute_avg_feature_attributions(
    model: CoPhaser,
    batch_size=1024,
    device="cuda",
):
    """
    Computes average feature importance for each latent dim (μ_i) across the dataset.

    Returns:
        attributions: np.ndarray of shape (mu_dim, n_input)
    """
    model.to("cpu")
    _, latent_outputs = model.get_outputs()
    model = model.to(device)
    x_data = model.rhythmic_genes.to(device)
    library_size = model.library_size.to(device)
    mu_Z = latent_outputs["mu_z"].to(device)
    amp_z = latent_outputs["lambda_enc"].to(device)
    use_mu_z_encoder = model.use_mu_z_encoder

    mu_dim = 2
    n_input = x_data.shape[1]
    running_attr = torch.zeros((mu_dim, n_input), device=device)

    dataloader = DataLoader(torch.arange(len(x_data)), batch_size=batch_size)

    for batch_idxs in dataloader:
        x_batch = x_data[batch_idxs]
        lib_batch = library_size[batch_idxs]
        mu_z_batch = mu_Z[batch_idxs]
        amp_z_batch = amp_z[batch_idxs]

        wrapper = EncoderMeanWrapper(
            model.rhythmic_encoder,
            lib_batch,
            mu_z_batch,
            amp_z_batch,
            use_mu_z_encoder,
        ).to(device)
        ig = IntegratedGradients(wrapper)

        for i in range(mu_dim):
            attributions = ig.attribute(
                inputs=x_batch,
                baselines=torch.zeros_like(x_batch),
                target=i,
                internal_batch_size=batch_size,
            )  # shape: (batch_size, n_input)
            running_attr[i] += attributions.sum(dim=0)

    avg_attr = running_attr / len(x_data)  # shape: (mu_dim, n_input)
    return avg_attr.detach().cpu().numpy()
