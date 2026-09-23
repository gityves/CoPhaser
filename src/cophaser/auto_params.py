"""Hyperparameters derived from the data instead of tuned by hand."""

import warnings
import weakref
from typing import Literal

import numpy as np
import pandas as pd
import torch

from cophaser._resources import resource_path

Cycle = Literal["cell_cycle", "circadian"]

# ELBO at unit weights, as a fraction of the mean-rate null cost.
ELBO_PER_NULL_COST = 0.933

# ELBO target per cycling cell; the target scales as 1 / (cycling fraction).
ELBO_PER_CYCLING_CELL = 3200.0
MAX_G1_FOR_LAW = 0.80  # floor on g1; above this is unmeasured

DEFAULT_CLOSED_CIRCLE_WEIGHT = 10.0  # not the package default of 1

# Switch on the Bernoulli non-cycling branch above this G1 fraction.
G1_FRACTION_THRESHOLD = 0.45
NON_CYCLING_PRIOR = 0.1

CIRCADIAN_ELBO_BAND = (900.0, 7000.0)  # measured range; outside it, warn
CIRCADIAN_TYPICAL_ELBO = 1800.0


def suggest_n_latent(sample_type: Literal["cell_line", "tissue"] = "tissue"):
    """Size of the non-periodic latent space: 2 for a cell line, 10 for tissue."""
    if sample_type not in ("cell_line", "tissue"):
        raise ValueError("sample_type must be 'cell_line' or 'tissue'")
    return 2 if sample_type == "cell_line" else 10


def null_reconstruction_cost(counts, library_size):
    """Mean NLL per cell of the "every gene at its mean rate" model.

    A training-free proxy for the ELBO at unit likelihood weights.
    """
    counts = torch.as_tensor(counts, dtype=torch.float32)
    lib = torch.as_tensor(library_size, dtype=torch.float32).clamp(min=1.0)
    mean_frac = counts.mean(dim=0) / lib.mean()

    mu = mean_frac * lib.mean()
    var = counts.var(dim=0)
    # var = mu + mu^2/theta where overdispersed, else near-Poisson.
    theta = torch.where(
        var > mu * 1.05, mu**2 / (var - mu).clamp(min=1e-6), torch.full_like(mu, 1e4)
    ).clamp(1e-2, 1e4)[None, :]

    # Chunked to bound memory.
    total, n = 0.0, counts.shape[0]
    for start in range(0, n, 8192):
        block = counts[start : start + 8192]
        rate = torch.clamp(
            mean_frac[None, :] * lib[start : start + 8192, None], min=1e-8
        )
        nb = torch.distributions.NegativeBinomial(
            total_count=theta, logits=torch.log(rate) - torch.log(theta)
        )
        total += float(-nb.log_prob(block).sum())
    return total / n


def suggest_likelihood_weights(model, elbo_target=ELBO_PER_CYCLING_CELL, ratio=1.0):
    """Likelihood weights putting the ELBO near `elbo_target` from the first epoch.

    `ratio` is rhythmic:non-rhythmic, at fixed geometric mean. Returns Trainer kwargs.
    """
    cost = null_reconstruction_cost(model.variable_genes, model.library_size)
    weight = float(elbo_target / max(cost * ELBO_PER_NULL_COST, 1e-6))
    return dict(
        rhythmic_likelihood_weight=weight * float(np.sqrt(ratio)),
        non_rhythmic_likelihood_weight=weight / float(np.sqrt(ratio)),
    )


def _seurat_cell_cycle_gene_sets():
    """Seurat/Tirosh S and G2/M lists, used for scoring cells into phases."""
    df = pd.read_csv(resource_path("regev_lab_cell_cycle_genes.csv"))
    phase = df["phase"].astype(str).str.upper()
    names = df["gene"].astype(str).str.upper()
    return sorted(set(names[phase == "S"])), sorted(set(names[phase == "G2M"]))


def cycling_fractions(model=None, adata=None, layer=None, library_size_field=None):
    """Fraction of cells scored G1, S and G2/M, by scanpy's Tirosh-style scoring.

    Prefer `adata` over the model; NaN if too few cell-cycle genes are present.
    """
    scores, n_s, n_g2m = _score_cell_cycle(model, adata, layer, library_size_field)
    out = dict(
        g1_fraction=np.nan,
        s_fraction=np.nan,
        g2m_fraction=np.nan,
        n_s_genes=n_s,
        n_g2m_genes=n_g2m,
    )
    if scores is None:
        return out
    counts = scores["phase"].value_counts(normalize=True)
    out.update(
        g1_fraction=float(counts.get("G1", 0.0)),
        s_fraction=float(counts.get("S", 0.0)),
        g2m_fraction=float(counts.get("G2M", 0.0)),
    )
    return out


# id(adata) -> (weakref to adata, key, result); lets several models share one scoring.
_ADATA_SCORES = {}


def _score_cell_cycle(model=None, adata=None, layer=None, library_size_field=None):
    """Score every cell for S and G2/M phase; cached per AnnData, else on the model.

    Prefer `adata`: scoring off the model restricts the control-gene pool to the
    context genes, which distorts the G1 fraction.

    Returns (scores DataFrame or None, n_s_genes, n_g2m_genes).
    """
    if adata is not None:
        key = (layer, library_size_field, adata.shape)
        cached = _ADATA_SCORES.get(id(adata))
        if cached is not None and cached[0]() is adata and cached[1] == key:
            return cached[2]
        out = _compute_cell_cycle_scores(None, adata, layer, library_size_field)
        adata_id = id(adata)
        ref = weakref.ref(adata, lambda _: _ADATA_SCORES.pop(adata_id, None))
        _ADATA_SCORES[adata_id] = (ref, key, out)
        return out

    key = ("model", id(model))
    cached = getattr(model, "_cophaser_cc_scores", None) if model is not None else None
    if cached is not None and cached[0] == key:
        return cached[1:]
    out = _compute_cell_cycle_scores(model, None, layer, library_size_field)
    if model is not None:
        model._cophaser_cc_scores = (key,) + out
    return out


def _compute_cell_cycle_scores(model, adata, layer, library_size_field):

    import anndata as ad
    import scanpy as sc
    import scipy.sparse as sp

    if adata is not None:
        counts = adata.layers[layer] if layer is not None else adata.X
        names = [str(v) for v in adata.var_names]
        if library_size_field is not None:
            lib = np.asarray(adata.obs[library_size_field], dtype=np.float64)
        else:
            lib = np.asarray(counts.sum(axis=1), dtype=np.float64).ravel()
    elif model is not None:
        counts = model.variable_genes.detach().cpu().numpy()
        names = [str(g) for g in model.context_genes]
        lib = model.library_size.detach().cpu().numpy().astype(np.float64).ravel()
    else:
        raise ValueError("pass an AnnData or a loaded model")

    s_genes, g2m_genes = _seurat_cell_cycle_gene_sets()
    upper = {}
    for name in names:
        upper.setdefault(name.upper(), name)
    s = [upper[g] for g in s_genes if g in upper]
    g2m = [upper[g] for g in g2m_genes if g in upper]
    if len(s) < 5 or len(g2m) < 5:
        return None, len(s), len(g2m)

    # Normalise by the true library size, keeping sparse input sparse.
    lib = np.where(lib > 0, lib, 1.0)
    scale = (1e4 / lib).astype(np.float32)
    if sp.issparse(counts):
        X = sp.diags(scale) @ counts.tocsr().astype(np.float32)
    else:
        # a single float32 copy, scaled in place
        X = np.array(counts, dtype=np.float32)
        X *= scale[:, None]

    tmp = ad.AnnData(X=X, var=pd.DataFrame(index=pd.Index(names)))
    sc.pp.log1p(tmp)
    sc.tl.score_genes_cell_cycle(tmp, s_genes=s, g2m_genes=g2m)
    return tmp.obs[["S_score", "G2M_score", "phase"]].copy(), len(s), len(g2m)


def _standardise(x):
    x = np.asarray(x, dtype=float)
    sd = x.std()
    return (x - x.mean()) / (sd if sd > 0 else 1.0)


def seurat_phase_prior(model=None, adata=None, layer=None, library_size_field=None):
    """Rough per-cell phase angle from the standardised (S, G2/M) scores, for use as a prior.

    Orders cells G1 -> S -> G2/M but the spacing is wrong, so Trainer anneals it away.
    Returns angles in radians, or None if the cells cannot be scored.
    """
    scores, _, _ = _score_cell_cycle(model, adata, layer, library_size_field)
    if scores is None:
        return None

    return np.arctan2(
        _standardise(scores["G2M_score"]), _standardise(scores["S_score"])
    )


def seurat_phase_prior_confidence(
    model=None, adata=None, layer=None, library_size_field=None, max_ratio=3.0
):
    """Per-cell weight for `seurat_phase_prior`: radius in the (S, G2/M) plane,
    normalised to mean 1 and clipped at `max_ratio`.

    Returns weights, or None if the cells cannot be scored.
    """
    scores, _, _ = _score_cell_cycle(model, adata, layer, library_size_field)
    if scores is None:
        return None

    radius = np.hypot(
        _standardise(scores["S_score"]), _standardise(scores["G2M_score"])
    )
    mean = radius.mean()
    if not np.isfinite(mean) or mean <= 0:
        return np.ones_like(radius)
    return np.clip(radius / mean, 0.0, float(max_ratio))


def suggest_cycling_status_prior(
    model=None,
    cycle: Cycle = "cell_cycle",
    adata=None,
    layer=None,
    library_size_field=None,
):
    """Whether to switch on the Bernoulli non-cycling branch, and with what prior.

    Returns (prior, diagnostics). A prior of 1 disables the branch, which is right when
    essentially every cell cycles. Only defined for the cell cycle; other cycles get 1.
    """
    if cycle != "cell_cycle":
        return 1.0, dict(reason=f"no rule for cycle={cycle}, leaving the branch off")
    diag = cycling_fractions(model, adata, layer, library_size_field)
    g1 = diag["g1_fraction"]
    if not np.isfinite(g1):
        diag["reason"] = "too few cell-cycle genes among the context genes to score"
        return 1.0, diag
    prior = NON_CYCLING_PRIOR if g1 > G1_FRACTION_THRESHOLD else 1.0
    diag["reason"] = (
        f"{g1:.0%} of cells scored G1, "
        f"{'above' if g1 > G1_FRACTION_THRESHOLD else 'below'} the "
        f"{G1_FRACTION_THRESHOLD:.0%} threshold"
    )
    return prior, diag


def suggest_elbo_target(
    model,
    cycling_status_prior=1.0,
    adata=None,
    layer=None,
    library_size_field=None,
    cycling_fraction=None,
):
    """`ELBO_PER_CYCLING_CELL / cycling fraction`.

    The fraction is `cycling_fraction` if given, else 1 with the Bernoulli branch off and
    `1 - g1` with it on.
    """
    if cycling_fraction is not None:
        return float(ELBO_PER_CYCLING_CELL / float(cycling_fraction))
    if cycling_status_prior >= 1:
        return float(ELBO_PER_CYCLING_CELL)
    g1 = cycling_fractions(model, adata, layer, library_size_field)["g1_fraction"]
    if not np.isfinite(g1):
        return float(ELBO_PER_CYCLING_CELL / (1.0 - MAX_G1_FOR_LAW))
    return float(ELBO_PER_CYCLING_CELL / (1.0 - min(g1, MAX_G1_FOR_LAW)))


def _effective_likelihood_weight(model, rhythmic_weight, non_rhythmic_weight):
    """Cost-weighted mean of the rhythmic / non-rhythmic likelihood weights."""
    idx = getattr(model, "rhythmic_gene_indices", None)
    counts, lib = model.variable_genes, model.library_size
    if idx is None or not len(idx):
        return float(non_rhythmic_weight)
    idx = np.asarray(list(idx), dtype=int)
    mask = np.zeros(int(counts.shape[1]), dtype=bool)
    mask[idx] = True
    cost_r = null_reconstruction_cost(counts[:, mask], lib)
    cost_n = null_reconstruction_cost(counts[:, ~mask], lib)
    total = cost_r + cost_n
    if not np.isfinite(total) or total <= 0:
        return float(non_rhythmic_weight)
    return float((rhythmic_weight * cost_r + non_rhythmic_weight * cost_n) / total)


def _paper_config_hyperparameters(model, cycle="circadian", verbose=True):
    """Paper configuration from `cycle_configs`, with only the likelihood weights
    re-derived from this dataset (cycling fraction 1)."""
    from cophaser import cycle_configs

    spec = cycle_configs.CYCLE_TRAINER_DEFAULTS[cycle]
    trainer_args = dict(spec["trainer"])
    trainer_args.update(
        suggest_likelihood_weights(
            model,
            elbo_target=suggest_elbo_target(model, cycling_fraction=1.0),
            ratio=1.0,
        )
    )

    predicted = (
        null_reconstruction_cost(model.variable_genes, model.library_size)
        * ELBO_PER_NULL_COST
    )
    expected = predicted * _effective_likelihood_weight(
        model,
        float(trainer_args["rhythmic_likelihood_weight"]),
        float(trainer_args["non_rhythmic_likelihood_weight"]),
    )
    diag = dict(
        reason=f"{cycle}: likelihood weights from the cycling-fraction law at fraction 1; "
        "the paper's other knobs",
        predicted_elbo_at_unit_weights=predicted,
        expected_elbo=expected,
        n_cells=int(len(model.library_size)),
    )
    # Only circadian has a measured band.
    if cycle == "circadian":
        diag["elbo_band"] = CIRCADIAN_ELBO_BAND
        low, high = CIRCADIAN_ELBO_BAND
        if not (low <= expected <= high):
            diag["warning"] = (
                f"this dataset's expected ELBO is {expected:.0f}, outside the "
                f"{low:.0f}-{high:.0f} band the configuration was measured in. Consider "
                "scaling both likelihood weights by "
                f"{CIRCADIAN_TYPICAL_ELBO / max(expected, 1e-6):.2f} and comparing the two with "
                "cycle_qc.fit_with_restarts."
            )
    if verbose:
        print(
            f"auto hyperparameters ({cycle}): likelihood weights "
            f"{trainer_args['rhythmic_likelihood_weight']:.2f} (rhythmic) / "
            f"{trainer_args['non_rhythmic_likelihood_weight']:.2f} (non-rhythmic), expected "
            f"ELBO {expected:.0f}. Use several restarts with cycle_qc.fit_with_restarts "
            "rather than tuning.",
            flush=True,
        )
        if "warning" in diag:
            print(f"  Warning: {diag['warning']}", flush=True)
    return dict(trainer=trainer_args, diagnostics=diag)


def auto_hyperparameters(
    model,
    cycle: Cycle = "cell_cycle",
    adata=None,
    layer=None,
    library_size_field=None,
    elbo_target=None,
    cycling_fraction=None,
    ratio=1.0,
    verbose=True,
    cycling_status_prior=None,
):
    """Data-derived hyperparameters for a loaded model.

    Returns {"trainer": {...}, "diagnostics": {...}}; `trainer` goes straight to Trainer().
    `elbo_target`, `cycling_fraction` and `cycling_status_prior` override what is derived
    (the likelihood weights follow the given prior). n_latent is excluded - it must be
    chosen before the model is built, see suggest_n_latent.
    """
    if cycle != "cell_cycle":
        ignored = [
            name
            for name, value in (
                ("elbo_target", elbo_target),
                ("cycling_fraction", cycling_fraction),
                ("cycling_status_prior", cycling_status_prior),
            )
            if value is not None
        ] + (["ratio"] if ratio != 1.0 else [])
        if ignored:
            warnings.warn(f"{', '.join(ignored)} ignored for cycle={cycle!r}.", stacklevel=2)
        return _paper_config_hyperparameters(model, cycle=cycle, verbose=verbose)

    if cycling_status_prior is not None:
        prior = float(cycling_status_prior)
        diag = dict(reason="set by the user")
    else:
        prior, diag = suggest_cycling_status_prior(
            model,
            cycle=cycle,
            adata=adata,
            layer=layer,
            library_size_field=library_size_field,
        )
    if elbo_target is None:
        elbo_target = suggest_elbo_target(
            model,
            cycling_status_prior=prior,
            adata=adata,
            layer=layer,
            library_size_field=library_size_field,
            cycling_fraction=cycling_fraction,
        )
    weights = suggest_likelihood_weights(model, elbo_target=elbo_target, ratio=ratio)
    trainer_args = dict(weights)
    trainer_args["cycling_status_prior"] = prior
    trainer_args["closed_circle_weight"] = DEFAULT_CLOSED_CIRCLE_WEIGHT
    if prior < 1:
        # Keep the warm-started decoder frozen for the first epochs (the CLI and
        # fit_with_restarts apply the warm start when cycling_status_prior < 1).
        trainer_args["unfreeze_epoch_layer"] = [(20, "rhythmic_decoder")]
    diag["predicted_elbo_at_unit_weights"] = (
        null_reconstruction_cost(model.variable_genes, model.library_size)
        * ELBO_PER_NULL_COST
    )
    diag["elbo_target"] = float(elbo_target)
    if verbose:
        print(
            f"auto hyperparameters: likelihood weights "
            f"{trainer_args['rhythmic_likelihood_weight']:.2f} (rhythmic) / "
            f"{trainer_args['non_rhythmic_likelihood_weight']:.2f} (non-rhythmic) "
            f"to target an ELBO of {elbo_target:.0f}; "
            f"cycling_status_prior {prior} - {diag['reason']}"
            + (
                "; a warm-started rhythmic decoder is released at epoch 20"
                if prior < 1
                else ""
            ),
            flush=True,
        )
    return dict(trainer=trainer_args, diagnostics=diag)


def candidate_configs(
    model,
    cycle: Cycle = "cell_cycle",
    closed_circle_weights=(3.0, 10.0, 30.0),
    elbo_target=None,
    ratio=1.0,
    verbose=True,
):
    """Trainer configs varying only `closed_circle_weight`, for
    cycle_qc.fit_with_restarts(configs=...).

    The likelihood weights stay fixed so quality scores remain comparable across configs.
    Use at least two seeds per config (n_restarts >= 2 * len(configs)). Cell cycle only.
    """
    if cycle != "cell_cycle":
        raise ValueError(
            f"candidate_configs is only defined for the cell cycle; for cycle={cycle!r} use "
            "auto_hyperparameters, which keeps the paper's closed_circle_weight."
        )
    base = auto_hyperparameters(
        model, cycle=cycle, elbo_target=elbo_target, ratio=ratio, verbose=verbose
    )
    configs = []
    for weight in closed_circle_weights:
        args = dict(base["trainer"])
        args["closed_circle_weight"] = float(weight)
        configs.append(args)
    return configs
