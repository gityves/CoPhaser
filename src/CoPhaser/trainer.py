from torch.utils.data import DataLoader
import torch
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from CoPhaser import SingleCellDataset
from CoPhaser.MINE import MINE
from CoPhaser.model.CoPhaser import CoPhaser
from typing import Literal, List, Tuple


class Trainer:
    def __init__(
        self,
        model: CoPhaser,
        loss_fn,
        # modifying these likely to have a large effect
        entropy_weight_factor=100,
        closed_circle_weight=1,
        MI_weight=100,
        # modifying these likely to have a moderate effect
        non_rhythmic_likelihood_weight=1,
        rhythmic_likelihood_weight=1,
        MI_detach: Literal["f", "z", "none"] = "f",
        # probably don't need to modify these
        beta_kl_f=0.1,
        beta_kl_cycling_status=10,
        # I never modified these
        L2_Z_decoder_loss_weight=0,
        calculate_entropy_per_batch=True,
        noise_model: Literal["poisson", "ZINB", "NB"] = "NB",
        # modify if expecting a lot of non-cycling cells
        cycling_status_prior=1,
        # modify this to unfreeze layers at specific epochs
        unfreeze_epoch_layer: List[Tuple[int, str]] = [],
    ):
        """
        Trainer class to handle training of CoPhaser model. Sorted by expected impact on results with:

        large effects: entropy_weight_factor, MI_weight and closed_circle_weight;
        moderate effects: non_rhythmic_likelihood_weight, rhythmic_likelihood_weight, MI_detach
        small effects: beta_kl_f, beta_kl_cycling_status;
        probably don't need to modify these: L2_Z_decoder_loss_weight, calculate_entropy_per_batch, noise_model

        Special cases:
        cycling_status_prior, modify if expecting a lot of non-cycling cells.
        unfreeze_epoch_layer, modify this to unfreeze layers at specific epochs, if you pretrained parts of the model.

        Parameters
        ----------
        model : CoPhaser
            The CoPhaser model to be trained.
        loss_fn : function
            The loss function to be used for training.
        entropy_weight_factor : float, optional
            Weight factor for the entropy loss, leading to a more uniform distribution of inferred phases, by default 100
        closed_circle_weight : float, optional
            Weight for the closed circle loss, leading to a projected f space shaped like an annulus, by default 1
        MI_weight : float, optional
            Weight for the mutual information loss between Z and the inferred phase, leading to an orthogonalization of the two spaces, by default 100
        non_rhythmic_likelihood_weight : float, optional
            Weight for the non-rhythmic genes likelihood in the loss function, by default 1
        rhythmic_likelihood_weight : float, optional
            Weight for the rhythmic genes likelihood in the loss function, by default 1
        MI_detach : Literal["f", "z", "none"], optional
            Whether to detach either f or z when calculating the mutual information loss, by default "f" helps finding the cyclic space.
        beta_kl_f : float, optional
            Weight for the KL divergence of the phase posterior to the prior, by default 0.1. Too high values leads to posterior collapse, leading to a low kl_div_f (<1).
        beta_kl_cycling_status : float, optional
            Weight for the KL divergence of the cycling status posterior to the prior, by default 10. Too low values leads to no cells being classified as non-cycling.
            NB: only used if cycling_status_prior < 1.
        L2_Z_decoder_loss_weight : float, optional
            Weight for the L2 loss between the latent variable Z and its reconstruction from the decoder, by default 0
        calculate_entropy_per_batch : bool, optional
            Whether to calculate entropy independently for each batch.
            NB, most of the time no batches are given to the model, so this has no effect, by default True
        noise_model : Literal["poisson", "ZINB", "NB"], optional
            The noise model to be used for the likelihood, by default "NB"
        cycling_status_prior : float, optional
            Prior probability of a cell being cycling. If you expect a lot of non-cycling cells, set this to a value < 1, by default 1.
        unfreeze_epoch_layer : List[Tuple[int, str]], optional
            List of tuples specifying at which epoch to unfreeze which layer, useful if you pretrained parts of the model.
            E.g., [(50, "rhythmic_decoder"), (100, "z_encoder")] would unfreeze the rhythmic decoder at epoch 50 and the z encoder at epoch 100, by default []
        """

        self.model = model
        self.loss_fn = loss_fn
        self.non_rhythmic_likelihood_weight = non_rhythmic_likelihood_weight
        self.rhythmic_likelihood_weight = rhythmic_likelihood_weight
        self.calculate_entropy_per_batch = calculate_entropy_per_batch
        self.entropy_weight_factor = entropy_weight_factor
        self.L2_Z_decoder_loss_weight = L2_Z_decoder_loss_weight
        self.MI_weight = MI_weight
        self.noise_model = noise_model
        self.unfreeze_epoch_layer = unfreeze_epoch_layer
        self.beta_kl_f = beta_kl_f
        self.closed_circle_weight = closed_circle_weight
        self.cycling_status_prior = cycling_status_prior
        self.model.cycling_status_prior = cycling_status_prior
        self.beta_kl_cycling_status = beta_kl_cycling_status
        self.MI_detach = MI_detach

    @staticmethod
    def print_loss(losses: dict, epoch, max_epoch, only_total=False):
        epoch_str = f"Epoch {epoch + 1}/{max_epoch}"
        if only_total:
            print(f"{epoch_str}, total_loss: {(np.mean(losses['total_loss'])):.4f}")
        else:
            for key, value in losses.items():
                epoch_str += f", {key}: {np.mean(value):.4f}"
            print(epoch_str)

    @staticmethod
    def record_loss_batches(losses_batch: dict, losses_epoch: dict):
        if losses_epoch:
            for key, value in losses_batch.items():
                # detach if tensor
                if type(value) == torch.Tensor:
                    losses_epoch[key].append(float(value.detach()))
                else:
                    losses_epoch[key].append(float(value))
        else:
            for key, value in losses_batch.items():
                if type(value) == torch.Tensor:
                    losses_epoch[key] = [float(value.detach())]
                else:
                    losses_epoch[key] = [float(value)]

    @staticmethod
    def record_losses_epochs(losses_epoch: dict, losses_training: dict, epoch: int):
        if losses_training:
            for key, value in losses_epoch.items():
                losses_training[key].append(np.mean(value))
            losses_training["epoch"].append(epoch)
        else:
            # skip first value since very variable between training
            for key, value in losses_epoch.items():
                losses_training[key] = []
            losses_training["epoch"] = []

    @staticmethod
    def plot_losses(losses_training: dict):
        df_losses = pd.DataFrame(losses_training)
        df_losses = df_losses.melt(
            id_vars="epoch", var_name="Metric", value_name="Value"
        )
        df_losses.loc[
            (df_losses["Metric"].isin(["kl_div_z", "elbo_loss"]))
            & (df_losses["Value"] > 1000),
            "Value",
        ] = np.nan
        sns.lineplot(data=df_losses, x="epoch", y="Value", hue="Metric")
        plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", borderaxespad=0)

    def train_model(
        self,
        n_epochs=200,
        lr=1e-2,
        device="cuda" if torch.cuda.is_available() else "cpu",
        batch_size=1024,
        print_only_total_loss=False,
        silent=False,
    ):
        self._check_data_loaded()

        self.model.to(device)
        data_loader = self._create_dataloader(batch_size)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        mine_net, mine_optimizer = self._init_mine_network(device)

        losses_training = {}
        for epoch in range(n_epochs):
            self.model.train()
            losses_epoch = {}

            self._maybe_unfreeze_layers(epoch)

            for batch in data_loader:
                if batch[0].size(0) < batch_size / 2:
                    continue

                inputs = self._prepare_batch(batch, device)
                entropy_loss_weight = self._get_entropy_weight(epoch)
                optimizer.zero_grad()

                generative_outputs, inference_outputs = self.model(
                    inputs["variable_genes"],
                    inputs["rhythmic_genes"],
                    inputs["library_size"],
                    epoch,
                )

                loss_dict = self.loss_fn(
                    model=self.model,
                    x=inputs["variable_genes"],
                    epoch=epoch,
                    generative_outputs=generative_outputs,
                    inference_outputs=inference_outputs,
                    MINE_model=mine_net,
                    entropy_loss_weight=entropy_loss_weight,
                    entropy_per_batch=self.calculate_entropy_per_batch,
                    L2_Z_decoder_loss_weight=self.L2_Z_decoder_loss_weight,
                    MI_weight=self.MI_weight,
                    rhythmic_likelihood_weight=self.rhythmic_likelihood_weight,
                    non_rhythmic_likelihood_weight=self.non_rhythmic_likelihood_weight,
                    closed_circle_weight=self.closed_circle_weight,
                    noise_model=self.noise_model,
                    beta_kl_f=self.beta_kl_f,
                    beta_kl_cycling_status=self.beta_kl_cycling_status,
                    batch_keys=inputs["batch_keys"],
                    cycling_status_prior=self.cycling_status_prior,
                    MI_detach=self.MI_detach,
                )
                loss = loss_dict["total_loss"]
                self.record_loss_batches(
                    losses_batch=loss_dict, losses_epoch=losses_epoch
                )
                loss.backward()

                if epoch > 20:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 4.0)
                optimizer.step()

                self._train_mine(mine_net, mine_optimizer, loss_dict, inference_outputs)

            self._loss_handling(
                epoch,
                n_epochs,
                losses_epoch,
                losses_training,
                print_only_total_loss,
                silent,
            )

        if not silent:
            self.plot_losses(losses_training)
        else:
            for k in losses_training.keys():
                losses_training[k] = losses_training[k][-1]
            return losses_training

    def _check_data_loaded(self):
        if not self.model.adata_loaded:
            raise ValueError("model.load_anndata needs to be called before training.")

    def _create_dataloader(self, batch_size):
        dataset = SingleCellDataset(
            self.model.rhythmic_genes,
            self.model.variable_genes,
            self.model.library_size,
            self.model.batch_keys if self.model.batch_corrected else None,
        )
        return DataLoader(dataset, batch_size=batch_size, shuffle=True)

    def _init_mine_network(self, device):
        mine_net = MINE(n_harm=max(self.model.n_harm, 5), z_dim=self.model.n_latent)
        mine_net.to(device)
        mine_optimizer = torch.optim.Adam(mine_net.parameters(), lr=1e-3)
        return mine_net, mine_optimizer

    def _maybe_unfreeze_layers(self, epoch):
        for epoch_unfreeze, layer in self.unfreeze_epoch_layer:
            if epoch_unfreeze == epoch:
                if layer == "rhythmic_decoder":
                    self.model.rhythmic_decoder.unfreeze_all_parameters()
                elif layer == "rhythmic_encoder":
                    self.model.rhythmic_encoder.unfreeze_all_parameters()
                elif layer == "z_encoder":
                    self.model.var_encoder.unfreeze_all_parameters()
                    self.model.mean_encoder.unfreeze_all_parameters()
                elif layer == "z_decoder":
                    self.model.decoder_non_rhythmic_contribution.unfreeze_all_parameters()
                else:
                    raise ValueError(f"Unknown layer {layer} to unfreeze.")

    def _prepare_batch(self, batch, device):
        rhythmic_genes, variable_genes, library_size, batch_keys = [
            x.to(device) for x in batch
        ]
        return {
            "rhythmic_genes": rhythmic_genes,
            "variable_genes": variable_genes,
            "library_size": library_size,
            "batch_keys": None if batch_keys.nelement() == 0 else batch_keys,
        }

    def _get_entropy_weight(self, epoch):
        return np.exp(-(max(epoch, 30) - 30) / 100) * self.entropy_weight_factor

    def _train_mine(self, mine_net, mine_optimizer, loss_dict, inference_outputs):
        if loss_dict["kl_div_z"] < 1000:
            mine_optimizer.zero_grad()
            mi_loss = mine_net.mutual_information_loss(
                inference_outputs["theta"].detach(),
                inference_outputs["z"].detach(),
            )
            (-mi_loss).backward()
            mine_optimizer.step()

    def _loss_handling(
        self,
        epoch,
        n_epochs,
        losses_epoch,
        losses_training,
        print_only_total_loss,
        silent,
    ):
        if not silent:
            self.print_loss(
                losses_epoch,
                epoch=epoch,
                max_epoch=n_epochs,
                only_total=print_only_total_loss,
            )
        self.record_losses_epochs(losses_epoch, losses_training, epoch + 1)
