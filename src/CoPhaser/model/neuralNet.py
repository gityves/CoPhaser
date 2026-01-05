import torch
import torch.nn.functional as F
from typing import Literal
from CoPhaser.model.freezableModule import FreezableModule


class NeuralNet(FreezableModule):
    def __init__(
        self,
        n_input: int,
        n_output: int,
        link_var: Literal["exp", "softplus", "sigmoid", "none"],
    ):
        """
        Basic neural network

        Args:
            n_input: Input dimension
            n_output: Output dimension
            link_var: Final non-linearity type
        """
        super().__init__()
        total_input = n_input

        self.neural_net = torch.nn.Sequential(
            torch.nn.Linear(total_input, 128),
            torch.nn.GELU(),
            torch.nn.Linear(128, n_output),
        )
        self.link_var = link_var

    def forward(self, x: torch.Tensor, offset=1e-4):
        output = self.neural_net(x)
        if self.link_var == "exp":
            output = torch.exp(torch.clamp(output, max=50)) + offset
        elif self.link_var == "softplus":
            output = F.softplus(output) + offset
        elif self.link_var == "sigmoid":
            output = F.sigmoid(output)
        elif self.link_var == "tanh":
            output = F.tanh(output)
        return output

    def freeze_genes(self, genes_indices, is_input: bool) -> None:
        """Freeze specific genes."""
        if is_input:
            self.freeze_indices("neural_net.0.weight", genes_indices, dim=1)
        else:
            self.freeze_indices("neural_net.2.weight", genes_indices, dim=0)

    def unfreeze_genes(self, genes_indices, is_input: bool) -> None:
        """Unfreeze specific genes."""
        if is_input:
            self.unfreeze_indices("neural_net.0.weight", genes_indices, dim=1)
        else:
            self.unfreeze_indices("neural_net.2.weight", genes_indices, dim=0)
