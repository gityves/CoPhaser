import torch
import torch.nn as nn
from torch.distributions import RelaxedBernoulli
from CoPhaser.model import neuralNet
import torch.nn.functional as F
import numpy as np


class RelaxedBernoulliEncoder(nn.Module):
    def __init__(self, n_input, size_z):
        super().__init__()
        # self.rhythmic_genes_embedding_net = torch.nn.Linear(n_input, 10)
        self.encoder_net = neuralNet.NeuralNet(n_input + size_z, 1, "none")

    @staticmethod
    def _get_temperature(epoch, final_temp=0.1, initial_temp=1, anneal_rate=0.1):
        # Exponential annealing, clipped at final_temp
        return max(final_temp, initial_temp * np.exp(-anneal_rate * max(epoch - 10, 0)))

    @staticmethod
    def generative(logits, epoch):
        temperature = RelaxedBernoulliEncoder._get_temperature(epoch)
        dist = RelaxedBernoulli(temperature=temperature, logits=logits)
        return dist.rsample()

    @staticmethod
    def get_mode(logits, temperature):
        return torch.sigmoid(logits / temperature)

    def forward(self, raw_counts, z, epoch, library_size):
        library_size = library_size.view(-1, 1)
        x = torch.log1p(raw_counts / library_size * 1e4)
        # rhythmic_genes_emb = self.rhythmic_genes_embedding_net(x)
        x = torch.cat([x, z], dim=-1)
        logits = self.encoder_net(x).squeeze(-1)
        # to avoid numerical issues
        logits = torch.clamp(logits, min=-20, max=20)
        b_z = self.generative(logits, epoch)
        return b_z, logits

    @staticmethod
    def kl_divergence(logits, prior_p=0.9):
        """
        KL divergence between Bernoulli(q) and Bernoulli(p), in logit space.
        q is parameterized by logits.
        p is a fixed prior probability.
        """
        prior_p = torch.tensor(prior_p, device=logits.device, dtype=logits.dtype)
        prior_logit = torch.log(prior_p) - torch.log1p(-prior_p)

        q = torch.sigmoid(logits)

        kl = -(
            F.softplus(-logits)
            + (1 - q) * (logits - prior_logit)
            - F.softplus(-prior_logit)
        )
        return kl
