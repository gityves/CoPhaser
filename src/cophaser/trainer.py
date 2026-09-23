from torch.utils.data import DataLoader, Subset
import torch
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from cophaser import SingleCellDataset
from cophaser.MINE import MINE
from cophaser.model.CoPhaser import CoPhaser
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
        cycle: Literal["cell_cycle", "circadian"] = None,
        # optional informative prior on the phase, annealed to a flat prior during training
        phase_prior_anneal_epochs=50,
        phase_prior_weight=2.0,
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
        cycle, name the biological cycle ("cell_cycle" or "circadian") to enable the
        unsupervised quality score used to pick between restarts (see quality_score() and
        cophaser.cycle_qc.fit_with_restarts). Default None keeps the previous behaviour.

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
        cycle : Literal["cell_cycle", "circadian"], optional
            Needed only for quality_score()/quality_metrics(), by default None
        phase_prior_anneal_epochs : int, optional
            Epoch at which the phase prior becomes flat, by default 50.
        phase_prior_weight : float, optional
            How strongly the prior counts in the loss, by default 2.
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
        self.cycle = cycle
        self.phase_prior_anneal_epochs = phase_prior_anneal_epochs
        self.phase_prior_weight = phase_prior_weight

    def quality_score(self):
        """
        Unsupervised quality of the current fit, for comparing runs on the same dataset.
        """
        if self.cycle is None:
            raise ValueError(
                "quality_score() needs the cycle: pass cycle='cell_cycle' (or 'circadian') "
                "to Trainer."
            )
        from cophaser import cycle_qc

        return cycle_qc.quality_score(self.model, cycle=self.cycle)

    def quality_metrics(self):
        """
        All unsupervised quality checks of the current fit, as a dict.
        """
        if self.cycle is None:
            raise ValueError(
                "quality_metrics() needs the cycle: pass cycle='cell_cycle' (or 'circadian') "
                "to Trainer."
            )
        from cophaser import cycle_qc

        return cycle_qc.quality_metrics(self.model, cycle=self.cycle)

    @staticmethod
    def print_loss(losses: dict, epoch, max_epoch, only_total=False, interpreted=True):
        epoch_str = f"Epoch {epoch + 1}/{max_epoch}"
        if only_total:
            print(f"{epoch_str}, total_loss: {(np.mean(losses['total_loss'])):.2f}")
        else:
            for key, value in losses.items():
                to_add = ""
                if interpreted:
                    if key == "entropy_loss_unweighted":
                        v = np.mean(value)
                        if v < -3.39:
                            bin = "uniform"
                        elif v < -3.38:
                            bin = "~uniform"
                        elif v < -3.36:
                            bin = "slightly non-uniform"
                        else:
                            bin = " ⚠ non-uniform"
                        to_add = f", phase distribution: {bin}"
                    elif key == "kl_div_f":
                        v = np.mean(value)
                        if v < 1:
                            bin = "too low"
                        elif v < 3:
                            bin = "low"
                        if v < 3:
                            to_add = f", ⚠ phase certainty: {bin}"
                    elif key == "fraction_cycling_cells":
                        v = np.mean(value)
                        if v != 1:
                            to_add = f", fraction cycling cells: {v:.2f}"
                    elif key == "kl_div_z":
                        v = np.mean(value)
                        if v > 1000:
                            to_add = f", ⚠ posterior diverged kl_div_z: {v:.2f}"
                    elif key == "elbo_loss":
                        to_add = f", {key}: {np.mean(value):.2f}"
                    elif key == "MI_loss":
                        if np.mean(value) > 1:
                            to_add = f", ⚠ high MI: {np.mean(value):.2f}"
                    elif key == "total_loss":
                        to_add = f", (summed losses: {np.mean(value):.2f})"
                else:
                    to_add = f", {key}: {np.mean(value):.2f}"
                epoch_str += to_add
            print(epoch_str)

    @staticmethod
    def record_loss_batches(losses_batch: dict, losses_epoch: dict):
        for key, value in losses_batch.items():
            if type(value) == torch.Tensor:
                value = float(value.detach())
            losses_epoch.setdefault(key, []).append(float(value))

    @staticmethod
    def record_losses_epochs(losses_epoch: dict, losses_training: dict, epoch: int):
        if losses_training:
            n = len(losses_training["epoch"])
            for key, value in losses_epoch.items():
                if key not in losses_training:
                    losses_training[key] = [np.nan] * n
                losses_training[key].append(np.mean(value))
            for key, series in losses_training.items():
                if key != "epoch" and len(series) == n:
                    series.append(np.nan)
            losses_training["epoch"].append(epoch)
        else:
            # skip first value since very variable between training
            for key, value in losses_epoch.items():
                losses_training[key] = []
            losses_training["epoch"] = []

    @staticmethod
    def plot_losses(losses_training: dict):
        longest = max((len(v) for v in losses_training.values()), default=0)
        losses_training = {
            key: list(value) + [np.nan] * (longest - len(value))
            for key, value in losses_training.items()
        }
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
        batch_size=None,
        print_only_total_loss=False,
        silent=False,
        helped_training=True,
        on_epoch=None,
        subsample=20_480,
        subsample_seed=42,
    ):
        """
        Train the model.

        Parameters
        ----------
        n_epochs : int, optional
            Number of epochs to train for, by default 200
        lr : float, optional
            Learning rate for the optimizer, by default 1e-2
        device : str, optional
            Device to train on, by default "cuda" if available, else "cpu"
        batch_size : int or None, optional
            Batch size for the DataLoader. If None, will be set to roughly 1/10 of the number of cells, rounded to a power of two.
        print_only_total_loss : bool, optional
            If True, only print the total loss at each epoch, by default False
        silent : bool, optional
            If True, do not print any loss information, by default False
        helped_training : bool, optional
            If True, print additional information about the training process, by default True
        on_epoch : callable or None, optional
            Function to call at the end of each epoch, by default None
        subsample : int or None, optional
            Number of cells drawn (without replacement) at the start of every epoch; a fresh
            draw each epoch, so all cells are seen over training. ``None``, or a value
            >= the number of cells, trains on all cells every epoch. By default 20,480.
        subsample_seed : int or None, optional
            Seed for the subsample draws (not controlled by ``np.random.seed``), by default 42.
        """

        self._check_data_loaded()
        n_cells = len(self.model.library_size)
        if subsample is not None and int(subsample) >= n_cells:
            subsample = None
        subsample_rng = np.random.default_rng(subsample_seed)
        # cells seen per epoch
        n_epoch_cells = int(subsample) if subsample is not None else n_cells
        if batch_size is None:
            # set batch size to roughly 1/10 of the number of cells
            batch_size = 2 ** int(np.log2(n_epoch_cells / 10))
        elif batch_size > n_epoch_cells:
            print(
                f"Warning: batch_size={batch_size} exceeds the {n_epoch_cells} cells used per "
                f"epoch; using batch_size={n_epoch_cells} instead. Pass batch_size=None to let "
                "the trainer choose (about a tenth of the cells, rounded to a power of two)."
            )
            batch_size = n_epoch_cells

        self.model.to(device)
        # with a subsample, the loader is rebuilt every epoch below
        data_loader = None if subsample is not None else self._create_dataloader(batch_size)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        mine_net, mine_optimizer = self._init_mine_network(device)

        n_skipped = 0  # optimizer steps skipped because the loss was not finite

        losses_training = {}

        for epoch in range(n_epochs):
            self.model.train()
            losses_epoch = {}

            self._maybe_unfreeze_layers(epoch)

            if subsample is not None:
                data_loader = self._create_dataloader(
                    batch_size,
                    indices=subsample_rng.choice(n_cells, size=int(subsample), replace=False),
                )

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
                    context_genes_raw_counts=inputs["variable_genes"],
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
                    phase_prior_direction=(
                        None
                        if inputs["phase_prior"] is None
                        else torch.stack(
                            [
                                torch.cos(inputs["phase_prior"]),
                                torch.sin(inputs["phase_prior"]),
                            ],
                            dim=-1,
                        )
                    ),
                    phase_prior_kappa=(
                        None
                        if inputs["phase_prior_kappa"] is None
                        else inputs["phase_prior_kappa"]
                        * self._phase_prior_anneal(epoch)
                    ),
                    phase_prior_weight=self.phase_prior_weight,
                )
                loss = loss_dict["total_loss"]
                self.record_loss_batches(
                    losses_batch=loss_dict, losses_epoch=losses_epoch
                )
                loss.backward()

                if epoch > 20:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 4.0)
                if self._step_is_finite(loss, self.model):
                    optimizer.step()
                else:
                    n_skipped += 1
                    optimizer.zero_grad()

                self._train_mine(mine_net, mine_optimizer, loss_dict, inference_outputs)

            self._loss_handling(
                epoch,
                n_epochs,
                losses_epoch,
                losses_training,
                print_only_total_loss,
                silent,
                helped_training,
                on_epoch,
            )
            if (
                epoch == 20
                and "elbo_loss" in losses_training
                and np.median(losses_training["elbo_loss"][15:]) < 900
            ):
                factor = 1000 / np.median(losses_training["elbo_loss"][15:])
                print(
                    f"Warning: the ELBO is {np.median(losses_training['elbo_loss'][15:]):.0f} "
                    f"at epoch 20, low enough that the reconstruction term may not dominate "
                    f"the regularisers. Training continues unchanged. To act on it, rerun "
                    f"with both likelihood weights multiplied by {factor:.2f} (rhythmic "
                    f"{self.rhythmic_likelihood_weight * factor:.2f}, non-rhythmic "
                    f"{self.non_rhythmic_likelihood_weight * factor:.2f}), or let "
                    f"cophaser.auto_hyperparameters set them from the counts."
                )
        if n_skipped:
            print(
                f"Warning: {n_skipped} optimizer step(s) were skipped because the loss "
                "was not finite. The run finished, but check it before trusting it."
            )
        if not silent:
            self.plot_losses(losses_training)
        else:
            for k in losses_training.keys():
                losses_training[k] = losses_training[k][-1]
            return losses_training

    @staticmethod
    def _step_is_finite(loss, module):
        """Whether this optimizer step is safe to apply."""
        if not torch.isfinite(loss).all():
            return False
        total = None
        for p in module.parameters():
            if p.grad is not None:
                total = p.grad.sum() if total is None else total + p.grad.sum()
        return total is None or bool(torch.isfinite(total))

    def _check_data_loaded(self):
        if not self.model.adata_loaded:
            raise ValueError("model.load_anndata needs to be called before training.")

    def _create_dataloader(self, batch_size, indices=None):
        """Training DataLoader; with `indices`, only those cells are visited."""
        dataset = SingleCellDataset(
            self.model.rhythmic_genes,
            self.model.variable_genes,
            self.model.library_size,
            self.model.batch_keys if self.model.batch_corrected else None,
            self.model.phase_prior if self._has_phase_prior() else None,
            self.model.phase_prior_kappa if self._has_phase_prior() else None,
        )
        if indices is not None:
            dataset = Subset(dataset, indices.tolist())
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
        (
            rhythmic_genes,
            variable_genes,
            library_size,
            batch_keys,
            phase_prior,
            phase_prior_kappa,
        ) = [x.to(device) for x in batch]
        return {
            "rhythmic_genes": rhythmic_genes,
            "variable_genes": variable_genes,
            "library_size": library_size,
            "batch_keys": None if batch_keys.nelement() == 0 else batch_keys,
            "phase_prior": None if phase_prior.nelement() == 0 else phase_prior,
            "phase_prior_kappa": (
                None if phase_prior_kappa.nelement() == 0 else phase_prior_kappa
            ),
        }

    def _has_phase_prior(self):
        return bool(getattr(self.model, "phase_prior_loaded", False))

    def _phase_prior_anneal(self, epoch):
        """How much of the prior's concentration survives at this epoch, in [0, 1].

        The per-cell concentration comes from the data (model.phase_prior_kappa); this only
        decides how fast it is annealed away. The prior is there to give early training a
        sensible starting arrangement of the cells; by the time this reaches zero the prior is
        flat and the phases are determined by the data alone, so a prior that is somewhat
        wrong cannot bias the final result.
        """
        if not self._has_phase_prior() or self.phase_prior_anneal_epochs <= 0:
            return 0.0
        return max(0.0, 1.0 - epoch / float(self.phase_prior_anneal_epochs))

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
            torch.nn.utils.clip_grad_norm_(mine_net.parameters(), 4.0)
            if self._step_is_finite(mi_loss, mine_net):
                mine_optimizer.step()
            else:
                mine_optimizer.zero_grad()

    def _loss_handling(
        self,
        epoch,
        n_epochs,
        losses_epoch,
        losses_training,
        print_only_total_loss,
        silent,
        helped_training=True,
        on_epoch=None,
    ):
        if not silent:
            self.print_loss(
                losses_epoch,
                epoch=epoch,
                max_epoch=n_epochs,
                only_total=print_only_total_loss,
                interpreted=helped_training,
            )
        self.record_losses_epochs(losses_epoch, losses_training, epoch + 1)
        if on_epoch is not None:
            on_epoch(epoch + 1, n_epochs, float(np.mean(losses_epoch["elbo_loss"])))
