import torch
import torch.nn as nn


# Mutual Information Estimator Network
class MINE(nn.Module):
    def __init__(self, n_harm, z_dim, hidden_dim=128):
        super(MINE, self).__init__()
        self.n_harm = n_harm
        self.fc1 = nn.Linear(n_harm * 2 + z_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)  # Output score

    def fourier_basis_expansion(self, theta):
        harmonics = []
        for harmonic in range(1, self.n_harm + 1):
            harmonics += [
                torch.cos(theta * harmonic).unsqueeze(1),
                torch.sin(theta * harmonic).unsqueeze(1),
            ]
        x_expanded = torch.cat(harmonics, dim=1)
        return x_expanded

    def forward(self, theta, z):
        x = self.fourier_basis_expansion(theta)
        combined = torch.cat([x, z], dim=-1)  # Concatenate x and z along last dimension
        h = torch.relu(self.fc1(combined))
        h = torch.relu(self.fc2(h))
        score = self.fc3(h)
        return score

    def mutual_information_loss(self, x, z):
        # Joint distribution samples
        joint_scores = self(x, z).mean()

        # Marginal distribution samples: shuffle `z` to get independent samples from `p(x)p(z)`
        z_shuffled = z[torch.randperm(z.size(0))]
        # SMILE clip (T=50)
        marginal_scores = torch.exp(
            torch.clamp(self(x, z_shuffled), min=-50, max=50)
        ).mean()

        # Compute the mutual information estimate (Donsker-Varadhan bound)
        mi_estimate = joint_scores - torch.log(marginal_scores)
        return mi_estimate
