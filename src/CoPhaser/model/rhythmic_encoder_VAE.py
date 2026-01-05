import torch
import torch.nn as nn
import torch.distributions as dist
from CoPhaser.powerSpherical import PowerSpherical, HypersphericalUniform
from CoPhaser.model.neuralNet import NeuralNet
from CoPhaser.model.freezableModule import FreezableModule


class RhythmicEncoderVAE(FreezableModule):
    def __init__(self, n_input, n_harm, preload_weights=[], default_angle=-1.5):
        super(RhythmicEncoderVAE, self).__init__()

        self.b = nn.Parameter(torch.randn(n_input) / 100)
        self.n_harm = n_harm
        self.SQRT2 = torch.sqrt(torch.tensor(2.0))
        self.default_angle = default_angle

        # Mean
        if len(preload_weights) == 0:
            self.fc1 = nn.Linear(n_input, 64, bias=False)
            self.fc2 = nn.Linear(64, 32, bias=False)
            self.fc_mu = nn.Linear(32, 2, bias=False)
            self.preloaded_weight = False
        else:
            self.fc1 = nn.Linear(n_input, 2, bias=False)
            self.fc2 = nn.Identity()
            self.fc_mu = nn.Identity()
            self.preloaded_weight = True
            with torch.no_grad():
                preload_weights = torch.tensor(
                    preload_weights.T, dtype=self.fc1.weight.dtype
                )
                self.fc1.weight.copy_(preload_weights)

        # Dispersion
        self.kappa_net = NeuralNet(n_input, 1, "softplus")

    def fourier_basis_expansion(self, theta):
        harmonics = []
        for harmonic in range(1, self.n_harm + 1):
            harmonics += [
                torch.cos(theta * harmonic).unsqueeze(1),
                torch.sin(theta * harmonic).unsqueeze(1),
            ]
        x_expanded = torch.cat(harmonics, dim=1)
        return x_expanded

    def reparameterize(self, mu, scale):
        q_mu = PowerSpherical(mu, scale.squeeze())
        return q_mu.rsample()  # Differentiable sampling

    def _mean_forward(self, x_norm):
        h = self.fc1(x_norm)
        h = self.fc2(h)
        x_projected = self.fc_mu(h)
        mu = x_projected / x_projected.norm(dim=-1, keepdim=True)
        return mu, x_projected

    def normalize_expression(
        self,
        counts,
        library_size,
        mu_z=None,
        _lambda=None,
        use_mu_z=True,
    ):
        if use_mu_z:
            counts_norm_center = (
                torch.log1p(counts / (library_size * torch.exp(mu_z)) * 1e4) - self.b
            ) / _lambda.view(-1, 1)
        else:
            counts_norm_center = torch.log1p(counts / library_size * 1e4) - self.b
        return counts_norm_center

    def forward(
        self,
        x,
        library_size,
        mu_Z,
        _lambda,
        use_mu_z_encoder,
        epoch=-1,
        use_max_posterior=False,
    ):
        library_size = library_size.view(-1, 1)

        x_norm = self.normalize_expression(
            x,
            library_size,
            mu_Z,
            _lambda,
            use_mu_z_encoder,
        )
        mu, x_projected = self._mean_forward(x_norm)
        if epoch != -1:
            # start training with highly concentrated distributions
            offset = 10000 * 10 ** -(epoch / 20)
            scale = self.kappa_net(x_norm, offset)
        else:
            scale = self.kappa_net(x_norm)
        # scale = torch.ones_like(x_norm[:,0])*10000

        if use_max_posterior:
            q_mu = mu
        else:
            q_mu = self.reparameterize(mu, scale)

        theta = torch.atan2(q_mu[:, 1] + 1e-4, q_mu[:, 0])
        f = self.fourier_basis_expansion(theta)
        # return values of f for theta = default_angle
        f_g1 = self.fourier_basis_expansion(
            torch.tensor([self.default_angle], device=x.device)
        )

        return f, theta, mu, scale, x_projected, f_g1

    def get_w_eff(self):
        if self.preloaded_weight:
            return self.fc1.weight
        return self.fc1.weight.T @ self.fc2.weight.T @ self.fc_mu.weight.T

    def get_importance_feature(self, feature_scales: torch.Tensor):
        W_eff = self.get_w_eff()
        W_eff_scaled = W_eff * feature_scales.unsqueeze(0)
        return torch.abs(W_eff_scaled)

    def kl_divergence(self, mu, scale):
        prior = HypersphericalUniform(dim=2)
        q_theta = PowerSpherical(mu, scale)
        kl = dist.kl.kl_divergence(q_theta, prior)
        return kl

    def freeze_weights_genes(self, genes_indices: list):
        self.freeze_indices("fc1.weight", genes_indices, dim=1)
        self.kappa_net.freeze_genes(genes_indices, True)

    def unfreeze_weights_genes(self, genes_indices: list):
        self.unfreeze_indices("fc1.weight", genes_indices, dim=1)
        self.kappa_net.unfreeze_genes(genes_indices, True)
