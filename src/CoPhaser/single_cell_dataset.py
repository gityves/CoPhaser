import torch
from torch.utils.data import Dataset


class SingleCellDataset(Dataset):
    def __init__(
        self,
        rhythmic_genes: torch.Tensor,
        variable_genes: torch.Tensor,
        library_size: torch.Tensor,
        batch_keys: torch.Tensor = None,
    ):
        """
        Initialize the dataset with the rhythmic genes, variable genes, and library size.

        ----------
        Parameters
        ----------
        rhythmic_genes : torch.Tensor
            The rhythmic genes counts for each cell.
        variable_genes : torch.Tensor
            The variable genes counts for each cell.
        library_size : torch.Tensor
            The library size for each cell.
        """
        self.rhythmic_genes = rhythmic_genes
        self.variable_genes = variable_genes
        self.library_size = library_size
        self.batch_keys = batch_keys

    def __len__(self):
        return len(self.rhythmic_genes)

    def __getitem__(self, idx):
        # Define placeholders for missing values
        empty_tensor = torch.tensor([], dtype=torch.float32)
        return (
            self.rhythmic_genes[idx],
            self.variable_genes[idx],
            self.library_size[idx],
            self.batch_keys[idx] if self.batch_keys is not None else empty_tensor,
        )


class SingleCellDatasetEncoder(Dataset):
    def __init__(
        self,
        variable_genes: torch.Tensor,
        f: torch.Tensor,
        qz_m: torch.Tensor,
        library_size: torch.Tensor,
        batch_keys: torch.Tensor = None,
    ):
        """
        Initialize the dataset with the rhythmic genes, variable genes, and library size.

        ----------
        Parameters
        ----------
        variable_genes : torch.Tensor
            The variable genes counts for each cell.

        library_size : torch.Tensor
            The library size for each cell.
        """
        self.variable_genes = variable_genes
        self.f = f
        self.qz_m = qz_m
        self.library_size = library_size
        self.batch_keys = batch_keys

    def __len__(self):
        return len(self.variable_genes)

    def __getitem__(self, idx):
        # Define placeholders for missing values
        empty_tensor = torch.tensor([], dtype=torch.float32)
        return (
            self.variable_genes[idx],
            self.f[idx],
            self.qz_m[idx],
            self.library_size[idx],
            self.batch_keys[idx] if self.batch_keys is not None else empty_tensor,
        )
