import torch.nn as nn
from CoPhaser.model.freezableModule import FreezableModule
import warnings


class RhythmicDecoder(FreezableModule):
    def __init__(self, n_genes, n_harm):
        super(RhythmicDecoder, self).__init__()
        self.n_harm = n_harm
        self.fourier_coefficients = nn.Linear(n_harm * 2, n_genes, bias=False)
        self.tensor_hook_handles = []

    def forward(self, f):
        F = self.fourier_coefficients(f)
        return F

    def freeze_weights_genes(self, genes_indices: list):
        self.freeze_indices("fourier_coefficients.weight", genes_indices, 0)

    def unfreeze_weights_genes(self, genes_indices: list):
        self.unfreeze_indices("fourier_coefficients.weight", genes_indices, 0)

    def _get_harm_indices(self, harmonics):
        harm_indices = []
        for harm in harmonics:
            if harm < 0:
                continue
            if harm == 0:
                warnings.warn(
                    "F0 defined as log(mean counts) + b in main model, skipped"
                )
                continue
            harm -= 1
            harm_indices.extend([harm * 2, harm * 2 + 1])
        return harm_indices

    def freeze_weights_harm(self, harmonics: list):
        harm_indices = self._get_harm_indices(harmonics)
        self.freeze_indices("fourier_coefficients.weight", harm_indices, 1)

    def unfreeze_weights_harm(self, harmonics: list):
        harm_indices = self._get_harm_indices(harmonics)
        self.unfreeze_indices("fourier_coefficients.weight", harm_indices, 1)

    def freeze_fourier_coefficients(self):
        self.freeze_weights_harm(list(range(self.n_harm * 2)))
