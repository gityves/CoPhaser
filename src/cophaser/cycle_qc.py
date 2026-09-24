"""Unsupervised quality signals for a fitted CoPhaser model, and restart-and-select.

Scores are comparable only *within* a dataset: use them to rank candidates, never to accept
or reject a single run.
"""

import copy
import time
from typing import Literal

import numpy as np
import torch

from cophaser import gene_sets
from cophaser._resources import resource_path
from tqdm import tqdm

Cycle = Literal["cell_cycle", "circadian"]

# Genes that should be sharply phase-locked.
_MARKER_GENES = {
    "cell_cycle": [str(g).upper() for g in gene_sets.human_canonical_histones if g],
    "circadian": [
        "ARNTL",
        "BMAL1",
        "PER1",
        "PER2",
        "PER3",
        "CRY1",
        "CRY2",
        "NR1D1",
        "NR1D2",
        "DBP",
        "TEF",
        "HLF",
        "CIART",
        "NPAS2",
    ],
}

# (early genes, late genes, expected peak separation in radians).
_PHASE_PAIRS = {
    # S vs G2/M
    "cell_cycle": (["PCNA", "MCM6"], ["TOP2A", "MKI67"], np.pi / 2),
    # PER/NR1D1 vs BMAL1/NPAS2, antiphase
    "circadian": (
        ["PER1", "PER2", "PER3", "NR1D1"],
        ["ARNTL", "BMAL1", "NPAS2"],
        np.pi,
    ),
}

# Expected S-to-G2/M gap = scale * pi * (S + G2/M fraction).
PHASE_SEPARATION_SCALE = 0.794

MIN_MARKER_GENES = 1
MARKER_GENES_WARN = 3

# Marker genes detected in fewer cells than this are ignored when scoring from an AnnData.
MARKER_MIN_DETECTED = 0.05

MIN_CLOCK_GENES_FOR_ORDERING = 5

RANKING_SIGNALS = (
    "marker_ratio",
    "marker_resultant",
    "phase_separation",
    "profile_reproducibility",
    "cycling_fraction",
)

# profile_reproducibility misranks circadian runs (few rhythmic genes vs many others).
RANKING_SIGNALS_BY_CYCLE: dict[str, tuple[str, ...]] = {
    "cell_cycle": RANKING_SIGNALS,
    "circadian": ("clock_acrophase_agreement", "phase_uniformity"),
}
DIAGNOSTIC_SIGNALS = ("library_size_ratio", "rhythmic_amplitude", "marker_pair_spread")

# Signals comparable across likelihood weights; the others are ranked within a weight group.
SCALE_FREE_SIGNALS = ("phase_separation", "clock_acrophase_agreement")

# Vetoes. Library size should swing ~2x over the cycle; much more means phase tracks depth.
LIBRARY_SIZE_VETO_RATIO = 3.0

# Normalised phase entropy (1 = uniform); low means reconstruction overpowered the entropy loss.
PHASE_UNIFORMITY_VETO = 0.88
ENTROPY_N_BINS = 30  # same as Loss.circular_entropy_loss
# Cycling fraction below this multiple of the prior means the branch learnt nothing.
CYCLING_COLLAPSE_RATIO = 1.5

# Peak-to-trough ratios ignore phase bins with fewer cells than this (lowered on small data).
MIN_CELLS_PER_BIN = 10


def _gene_indices(model, wanted):
    wanted = {str(g).upper() for g in wanted}
    return [i for i, g in enumerate(model.context_genes) if str(g).upper() in wanted]


def _theta_and_fractions(model):
    """Inferred phase, and each context gene's share of its cell's library."""
    model.eval()
    was_on = next(model.parameters()).device
    model.to("cpu")
    with torch.no_grad():
        _, inference_outputs = model.get_outputs()
    theta = inference_outputs["theta"].detach().cpu().numpy().ravel()
    model.to(was_on)

    # kept float32: this dense matrix can be GBs
    counts = model.variable_genes.detach().cpu().numpy()
    lib = model.library_size.detach().cpu().numpy()
    lib = np.where(lib <= 0, np.float32(1.0), lib)
    theta = np.arctan2(np.sin(theta), np.cos(theta))
    return theta, counts / lib[:, None], lib.astype(float), inference_outputs


def _phase_uniformity(theta, cycling=None, n_bins=ENTROPY_N_BINS):
    """Soft-binned phase entropy over log(n_bins); 1 = uniform. Restricted to `cycling` cells."""
    theta = np.asarray(theta, dtype=float)
    if cycling is not None:
        cycling = np.asarray(cycling, dtype=bool).ravel()
        if cycling.shape == theta.shape and cycling.any():
            theta = theta[cycling]
    if theta.size == 0:
        return np.nan
    edges = np.linspace(-np.pi, np.pi, n_bins + 1)
    centres = (edges[:-1] + edges[1:]) / 2
    width = 2 * np.pi / n_bins
    diff = np.mod(theta[:, None] - centres[None, :] + np.pi, 2 * np.pi) - np.pi
    hist = np.exp(-0.5 * (diff / width) ** 2).sum(axis=0) + 1e-10
    prob = hist / hist.sum()
    entropy = -np.sum(prob * np.log(prob + 1e-10))
    return float(entropy / np.log(n_bins))


def _cycling_mask(model, inference_outputs):
    """Cells with b_z > 0.5 (as in Loss.compute_elbo), or None when the branch is off."""
    prior = getattr(model, "cycling_status_prior", 1)
    if not isinstance(prior, (int, float)) or prior >= 1 or inference_outputs is None:
        return None
    b_z = inference_outputs.get("b_z")
    if b_z is None:
        return None
    return b_z.detach().cpu().numpy().ravel() > 0.5


def _cycling_fraction(model, inference_outputs):
    """Mean posterior cycling probability, or NaN when the Bernoulli branch is off."""
    prior = getattr(model, "cycling_status_prior", 1)
    if not isinstance(prior, (int, float)) or prior >= 1:
        return np.nan
    logits = inference_outputs.get("cycling_logits")
    if logits is None:
        return np.nan
    logits = logits.detach().cpu().numpy().ravel().astype(float)
    if not np.isfinite(logits).all():
        return np.nan
    return float(np.mean(1.0 / (1.0 + np.exp(-logits))))


def _binned_mean(theta, values, n_bins, min_cells=1):
    edges = np.linspace(-np.pi, np.pi, n_bins + 1)
    which = np.clip(np.digitize(theta, edges) - 1, 0, n_bins - 1)
    return np.array(
        [
            values[which == b].mean() if (which == b).sum() >= min_cells else np.nan
            for b in range(n_bins)
        ]
    )


def _peak_trough(theta, values, n_bins):
    min_cells = min(MIN_CELLS_PER_BIN, max(2, len(theta) // (4 * n_bins)))
    prof = _binned_mean(theta, values, n_bins, min_cells=min_cells)
    if np.isfinite(prof).sum() < 3:
        return np.nan
    # pseudocount (1% of the mean) so an empty trough bin gives a large ratio, not NaN
    eps = 0.01 * np.nanmean(prof)
    if not np.isfinite(eps) or eps <= 0:
        return np.nan
    return float((np.nanmax(prof) + eps) / (np.nanmin(prof) + eps))


def _resultants(theta, frac):
    """Per-gene (resultant length, total fraction): |sum_c frac_gc e^{i theta_c}| / sum_c frac_gc."""
    tot = frac.sum(axis=0)
    ok = tot > 0
    if not ok.any():
        return np.array([]), np.array([])
    cos = np.cos(theta).astype(frac.dtype, copy=False)
    sin = np.sin(theta).astype(frac.dtype, copy=False)
    re, im = cos @ frac[:, ok], sin @ frac[:, ok]
    return np.hypot(re, im) / tot[ok], tot[ok]


def _binned_profiles(theta, frac, n_bins):
    """(n_bins x n_genes) mean library-fraction per phase bin, and the cells per bin."""
    edges = np.linspace(-np.pi, np.pi, n_bins + 1)
    which = np.clip(np.digitize(theta, edges) - 1, 0, n_bins - 1)
    prof = np.full((n_bins, frac.shape[1]), np.nan, dtype=np.float64)
    per_bin = np.zeros(n_bins)
    for b in range(n_bins):
        m = which == b
        per_bin[b] = m.sum()
        if per_bin[b]:
            prof[b] = frac[m].sum(axis=0) / per_bin[b]
    return prof, per_bin


def _profile_reproducibility(theta, frac, idx, n_bins, seed=0):
    """Expression-weighted correlation of phase profiles between two random halves of cells."""
    if not len(idx):
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(theta))
    a, b = perm[: len(theta) // 2], perm[len(theta) // 2 :]
    sub = frac[:, idx]
    pa, _ = _binned_profiles(theta[a], sub[a], n_bins)
    pb, _ = _binned_profiles(theta[b], sub[b], n_bins)

    bins_ok = np.isfinite(pa).all(axis=1) & np.isfinite(pb).all(axis=1)
    if bins_ok.sum() < 8:
        return np.nan, np.nan
    pa, pb = pa[bins_ok], pb[bins_ok]

    keep = (pa.std(axis=0) > 0) & (pb.std(axis=0) > 0)
    if keep.sum() < 2:
        return np.nan, np.nan
    za = (pa[:, keep] - pa[:, keep].mean(axis=0)) / pa[:, keep].std(axis=0)
    zb = (pb[:, keep] - pb[:, keep].mean(axis=0)) / pb[:, keep].std(axis=0)
    corr = (za * zb).mean(axis=0)
    w = sub[:, keep].mean(axis=0).astype(np.float64)
    if w.sum() <= 0:
        return float(np.mean(corr)), float(keep.sum())
    return float((corr * w).sum() / w.sum()), float(keep.sum())


def _acrophase(theta, values):
    v = values - values.mean()
    if not np.any(v):
        return np.nan
    return float(np.arctan2((v * np.sin(theta)).mean(), (v * np.cos(theta)).mean()))


def _circ_mean(phases):
    phases = [p for p in phases if np.isfinite(p)]
    if not phases:
        return np.nan
    return float(np.arctan2(np.mean(np.sin(phases)), np.mean(np.cos(phases))))


def _clock_acrophase_agreement(theta, frac, model):
    """Mean acrophase error (hours, negated) vs gene_sets.amp_phase_circadian after the best rotation/reflection.

    The rotation is amplitude-weighted; the residual is an unweighted mean over genes.
    Returns (score, n_genes_used).
    """
    refs, phases, amps = [], [], []
    for gene, (_, ref_phase) in gene_sets.amp_phase_circadian.items():
        j = _gene_indices(model, [gene])
        if not j:
            continue
        values = frac[:, j[0]]
        v = values - values.mean()
        if not np.any(v):
            continue
        re = float((v * np.cos(theta)).mean())
        im = float((v * np.sin(theta)).mean())
        amp = float(np.hypot(re, im))
        if not np.isfinite(amp) or amp <= 0:
            continue
        refs.append(ref_phase)
        phases.append(float(np.arctan2(im, re)))
        amps.append(amp)

    if len(phases) < MIN_CLOCK_GENES_FOR_ORDERING:
        return np.nan, len(phases)

    fitted = np.asarray(phases)
    ref = np.asarray(refs)
    weights = np.asarray(amps)

    best = np.inf
    for direction in (1.0, -1.0):
        d = direction * fitted - ref
        shift = np.arctan2((weights * np.sin(d)).sum(), (weights * np.cos(d)).sum())
        resid = direction * fitted - shift - ref
        err = np.abs(np.arctan2(np.sin(resid), np.cos(resid)))
        best = min(best, float(err.mean()))
    return -float(best * 24.0 / (2 * np.pi)), len(phases)


def expected_phase_separation(model, cycle: Cycle = "cell_cycle"):
    """Expected peak separation (radians) of the two marker sets, from the S and G2/M fractions.

    Falls back to the fixed value in _PHASE_PAIRS. Diagnostic only, not used for ranking.
    """
    fallback = _PHASE_PAIRS[cycle][2]
    if cycle != "cell_cycle":
        return fallback
    fractions = getattr(model, "_cycle_qc_fractions", None)
    if fractions is None:
        from cophaser import auto_params

        fractions = auto_params.cycling_fractions(model)
        model._cycle_qc_fractions = fractions
    total = fractions["s_fraction"] + fractions["g2m_fraction"]
    if not np.isfinite(total) or total <= 0:
        return fallback
    return float(PHASE_SEPARATION_SCALE * np.pi * total)


def _marker_fraction_from_adata(
    adata, cycle, layer, library_size_field, min_detected_fraction=MARKER_MIN_DETECTED
):
    """Per-cell library fraction of all marker genes in `adata`, not only the model's genes.

    Returns (fraction per cell, n_genes_used, n_genes_available).
    """
    wanted = {str(g).upper() for g in _MARKER_GENES[cycle]}
    present = [g for g in adata.var_names if str(g).upper() in wanted]
    if not present:
        return None, 0, 0

    sub = adata[:, present]
    X = sub.layers[layer] if layer is not None else sub.X
    if hasattr(X, "tocsc"):
        Xc = X.tocsc()
        detected = np.diff(Xc.indptr) / max(adata.n_obs, 1)
    else:
        X = np.asarray(X)
        detected = (X > 0).mean(axis=0)
    keep = detected >= min_detected_fraction
    if not keep.any():
        # fall back to the most widely detected gene
        keep = np.zeros(len(present), dtype=bool)
        keep[int(np.argmax(detected))] = True

    counts = np.asarray(X[:, keep].sum(axis=1), dtype=float).ravel()

    if library_size_field is not None:
        lib = np.asarray(adata.obs[library_size_field], dtype=float)
    else:
        full = adata.layers[layer] if layer is not None else adata.X
        lib = np.asarray(full.sum(axis=1), dtype=float).ravel()
    lib = np.where(lib <= 0, 1.0, lib)
    return counts / lib, int(keep.sum()), len(present)


def preflight(
    adata,
    layer=None,
    rhythmic_gene_names=None,
    context_genes=None,
    cycle: Cycle = "cell_cycle",
    library_size_field=None,
    warn: bool = True,
):
    """Check sequencing depth, rhythmic gene count and marker-gene coverage before training.

    Returns the measurements with a `warnings` list.
    """
    out = {}
    if library_size_field is not None:
        lib = np.asarray(adata.obs[library_size_field], dtype=float)
    else:
        full = adata.layers[layer] if layer is not None else adata.X
        lib = np.asarray(full.sum(axis=1), dtype=float).ravel()
    out["n_cells"] = int(adata.n_obs)
    out["median_umi"] = float(np.median(lib)) if lib.size else np.nan
    out["n_rhythmic_genes"] = (
        None
        if rhythmic_gene_names is None
        else len(
            {str(g).upper() for g in rhythmic_gene_names}
            & {str(v).upper() for v in adata.var_names}
        )
    )
    coverage = marker_gene_coverage(
        adata, cycle=cycle, context_genes=context_genes, warn=False
    )
    out["marker_genes"] = coverage

    msgs = []
    umi = out["median_umi"]
    if np.isfinite(umi):
        if umi < 1000:
            msgs.append(
                f"median library size is {umi:.0f} UMI. Below ~1000 the inferred phase was "
                "unreliable at every hyperparameter setting tested; treat the result as "
                "exploratory."
            )
        elif umi < 3000:
            msgs.append(
                f"median library size is {umi:.0f} UMI. About 3000 is where best-of-N "
                "agreement reaches ~0.9; below it expect a ceiling well under that whatever "
                "the hyperparameters, and prefer more restarts to more tuning."
            )
    n_rhy = out["n_rhythmic_genes"]
    if n_rhy is not None and n_rhy < 50:
        msgs.append(
            f"only {n_rhy} of the annotated rhythmic genes are present; below ~50 the phase "
            "is materially weaker."
        )
    if coverage["n_selected"] < MARKER_GENES_WARN:
        msgs.append(
            "few marker genes survive gene selection, so the quality score that chooses "
            "between restarts will be noisy here - see marker_gene_coverage."
        )
    out["warnings"] = msgs
    if warn:
        for m in msgs:
            print(f"CoPhaser preflight: {m}", flush=True)
    return out


def marker_gene_coverage(
    source, cycle: Cycle = "cell_cycle", context_genes=None, warn: bool = True
):
    """How many marker genes the model will see; restart selection depends on them.

    Parameters
    ----------
    source: an AnnData, a CoPhaser model, or a list of gene names.
    context_genes: the genes the model will be given, when `source` is an AnnData. The
        report then lists marker genes present in the data but not selected.

    Returns
    -------
    dict with `n_selected` (marker genes the model will see), `n_in_data`, and `missing`
    (marker genes present in the data but not selected).
    """
    wanted = {str(g).upper() for g in _MARKER_GENES[cycle]}

    in_data = None
    if hasattr(source, "var_names"):
        in_data = [g for g in source.var_names if str(g).upper() in wanted]
        selected = (
            [g for g in context_genes if str(g).upper() in wanted]
            if context_genes is not None
            else in_data
        )
    elif hasattr(source, "context_genes"):
        selected = [g for g in source.context_genes if str(g).upper() in wanted]
    else:
        selected = [g for g in source if str(g).upper() in wanted]

    missing = []
    if in_data is not None:
        chosen = {str(g).upper() for g in selected}
        missing = [g for g in in_data if str(g).upper() not in chosen]

    out = dict(
        n_selected=len(selected),
        n_in_data=len(in_data) if in_data is not None else None,
        selected=list(selected),
        missing=missing,
    )
    if warn and len(selected) < MARKER_GENES_WARN:
        label = "histone" if cycle == "cell_cycle" else "core clock"
        if len(selected) < MIN_MARKER_GENES:
            head = (
                f"Warning: no {label} gene is among the genes given to the model. Restart "
                "selection then rests on `profile_reproducibility` alone, which is markedly "
                "weaker than the full set of signals - measured on datasets in this "
                "situation it recovers about a third of the gap between an arbitrary run "
                "and the best one, where the marker signals recover most of it."
            )
        else:
            head = (
                f"Note: only {len(selected)} {label} gene(s) among the genes given to the "
                "model. Restart selection still works but is noticeably noisier."
            )
        print(head)
        if missing:
            shown = ", ".join(map(str, missing[:10]))
            more = f", and {len(missing) - 10} more" if len(missing) > 10 else ""
            print(
                f"  {len(missing)} {label} gene(s) are present in the data but were not "
                f"selected: {shown}{more}."
            )
            print(
                "  Passing them in `context_genes` alongside your variable genes costs a "
                "few extra columns and makes the selection markedly more reliable."
            )
        elif in_data is not None and not in_data:
            print(
                f"  The data contains no {label} gene at all, so this cannot be fixed by "
                "changing gene selection. Judge the fit by eye "
                "(tutorials/1_general_tutorial.ipynb) rather than trusting the score."
            )
        elif in_data is not None:
            print(
                f"  Every {label} gene present in the data is already selected, so this is "
                "as good as gene selection can do here."
            )
    return out


def quality_metrics(
    model,
    cycle: Cycle = "cell_cycle",
    n_bins: int = 24,
    warn: bool = True,
    adata=None,
    layer=None,
    library_size_field=None,
):
    """All unsupervised quality signals for a fitted model, as a dict.

    Pass `adata` to score markers over every marker gene in the data, not only the model's.
    NaN where the needed genes are missing.
    """
    idx = _gene_indices(model, _MARKER_GENES[cycle])
    out = {k: np.nan for k in RANKING_SIGNALS + DIAGNOSTIC_SIGNALS}
    out["n_marker_genes"] = len(idx)
    if warn:
        marker_gene_coverage(model, cycle=cycle, warn=True)

    theta, frac, lib, inference_outputs = _theta_and_fractions(model)

    marker_frac = None
    if adata is not None:
        if adata.n_obs != len(theta):
            raise ValueError(
                f"adata has {adata.n_obs} cells but the model was fitted on {len(theta)}. "
                "Pass the same object, with the same cells in the same order, that the "
                "model was loaded from."
            )
        model_obs = getattr(model, "obs_names", None)
        if model_obs is not None and not np.array_equal(
            np.asarray(model_obs), np.asarray(adata.obs_names)
        ):
            raise ValueError(
                "adata.obs_names differ from the cells the model was loaded with. Pass the "
                "same cells in the same order."
            )
        marker_frac, n_used, n_available = _marker_fraction_from_adata(
            adata, cycle, layer, library_size_field
        )
        if n_available:
            out["n_marker_genes"] = n_used
            out["n_marker_genes_available"] = n_available
    if marker_frac is not None:
        # resultant of the summed marker fraction
        out["marker_ratio"] = _peak_trough(theta, marker_frac, n_bins)
        res, _ = _resultants(theta, marker_frac[:, None].astype(frac.dtype, copy=False))
        if res.size:
            out["marker_resultant"] = float(res[0])
    elif len(idx) >= MIN_MARKER_GENES:
        # expression-weighted mean of per-gene resultants
        out["marker_ratio"] = _peak_trough(theta, frac[:, idx].sum(axis=1), n_bins)
        res, tot = _resultants(theta, frac[:, idx])
        if res.size:
            out["marker_resultant"] = float((res * tot).sum() / tot.sum())

    early, late, expected = _PHASE_PAIRS[cycle]
    acro = {}
    for group in (early, late):
        for g in group:
            j = _gene_indices(model, [g])
            if j:
                acro[g] = _acrophase(theta, frac[:, j[0]])
    early_ph = [acro[g] for g in early if g in acro]
    late_ph = [acro[g] for g in late if g in acro]
    if early_ph and late_ph:
        gap = _circ_mean(late_ph) - _circ_mean(early_ph)
        gap = np.arctan2(np.sin(gap), np.cos(gap))
        # |gap|: the direction of theta is arbitrary
        out["phase_separation"] = -float(abs(abs(gap) - expected))
    if len(early_ph) > 1 or len(late_ph) > 1:
        spreads = []
        for phases in (early_ph, late_ph):
            for a, b in zip(phases, phases[1:]):
                d = a - b
                spreads.append(abs(np.arctan2(np.sin(d), np.cos(d))))
        out["marker_pair_spread"] = -float(np.mean(spreads))

    if cycle == "circadian":
        agreement, n_clock = _clock_acrophase_agreement(theta, frac, model)
        out["clock_acrophase_agreement"] = agreement
        out["n_clock_reference_genes"] = n_clock

    out["expected_phase_separation"] = expected_phase_separation(model, cycle)
    out["library_size_ratio"] = _peak_trough(theta, lib, n_bins)
    out["phase_uniformity"] = _phase_uniformity(
        theta, _cycling_mask(model, inference_outputs)
    )
    out["cycling_fraction"] = _cycling_fraction(model, inference_outputs)
    out["cycling_status_prior"] = float(getattr(model, "cycling_status_prior", 1) or 1)
    rhythmic_idx = getattr(model, "rhythmic_gene_indices", None)
    if rhythmic_idx is not None and len(rhythmic_idx):
        rhythmic_idx = list(rhythmic_idx)
        res, _ = _resultants(theta, frac[:, rhythmic_idx])
        if res.size:
            out["rhythmic_amplitude"] = float(np.mean(res))

        # rhythmic minus others: a phase tracking library size makes every profile reproducible
        others = np.setdiff1d(np.arange(frac.shape[1]), rhythmic_idx)
        rhythmic_rep, _ = _profile_reproducibility(theta, frac, rhythmic_idx, n_bins)
        other_rep, _ = _profile_reproducibility(theta, frac, others, n_bins)
        if np.isfinite(rhythmic_rep) and np.isfinite(other_rep):
            out["profile_reproducibility"] = rhythmic_rep - other_rep
    return out


def quality_score(model, cycle: Cycle = "cell_cycle", warn: bool = True):
    """Peak-to-trough ratio of the marker-gene fraction across inferred phase.

    Higher is better; NaN without marker genes. To compare runs prefer rank_candidates()
    over a list of quality_metrics().
    """
    return quality_metrics(model, cycle=cycle, warn=warn)["marker_ratio"]


def rank_candidates(
    metrics_list, signals=None, groups=None, cycle: Cycle = "cell_cycle", _veto=True
):
    """Score runs of one dataset by their mean percentile rank over the defined signals.

    Returns one score per run in [0, 1], higher better; vetoed runs get 1 subtracted.
    `signals=None` uses RANKING_SIGNALS_BY_CYCLE[cycle]. `groups` labels runs trained with
    different likelihood weights; signals not in SCALE_FREE_SIGNALS are ranked within a group.
    """
    if signals is None:
        signals = RANKING_SIGNALS_BY_CYCLE.get(cycle, RANKING_SIGNALS)
    n = len(metrics_list)
    groups = [None] * n if groups is None else list(groups)
    if len(set(groups)) > 1 and max(groups.count(g) for g in set(groups)) < 2:
        print(
            "Warning: every config group has a single run, so signals cannot be ranked "
            "within groups; ranking all runs together instead."
        )
        groups = [None] * n
    totals, counts = np.zeros(n), np.zeros(n)
    for sig in signals:
        vals = np.array([float(m.get(sig, np.nan)) for m in metrics_list])
        if sig not in SCALE_FREE_SIGNALS and len(set(groups)) > 1:
            for g in set(groups):
                mask = np.array([x == g for x in groups])
                if mask.sum() == 1:
                    # a lone run has nothing to rank against in its group; rank it among all
                    # runs so it is scored on the same signals as the others
                    sub = rank_candidates(metrics_list, signals=(sig,), _veto=False)[mask]
                else:
                    sub = rank_candidates(
                        [m for m, keep in zip(metrics_list, mask) if keep],
                        signals=(sig,),
                        _veto=False,
                    )
                ok_sub = np.isfinite(sub)
                idx = np.flatnonzero(mask)[ok_sub]
                totals[idx] += sub[ok_sub]
                counts[idx] += 1
            continue
        ok = np.isfinite(vals)
        if ok.sum() < 2:
            continue
        # average ranks for ties, scaled to [0, 1]
        order = vals[ok].argsort()
        ranks = np.empty(ok.sum())
        ranks[order] = np.arange(ok.sum())
        v = vals[ok]
        for value in np.unique(v):
            tie = v == value
            if tie.sum() > 1:
                ranks[tie] = ranks[tie].mean()
        denom = max(ok.sum() - 1, 1)
        totals[ok] += ranks / denom
        counts[ok] += 1
    with np.errstate(invalid="ignore"):
        scores = np.where(counts > 0, totals / np.where(counts == 0, 1, counts), np.nan)
    if not _veto:
        return scores

    vetoed = np.array(
        [
            float(m.get("library_size_ratio", np.nan)) > LIBRARY_SIZE_VETO_RATIO
            or float(m.get("phase_uniformity", np.nan)) < PHASE_UNIFORMITY_VETO
            or float(m.get("cycling_fraction", np.nan))
            < _collapse_threshold(float(m.get("cycling_status_prior", 1) or 1))
            for m in metrics_list
        ]
    )
    return scores - vetoed.astype(float)


def _collapse_threshold(prior):
    # capped halfway to 1, otherwise a prior above 1/1.5 vetoes every run
    return min(CYCLING_COLLAPSE_RATIO * prior, prior + (1 - prior) / 2)


class RestartResults(list):
    """List of restart dicts (`model`, `trainer`, `seed`, `score`, metrics...), best first."""

    @property
    def best(self):
        return self[0] if self else None

    def __repr__(self):
        if not self:
            return "RestartResults([])"
        rows = ", ".join(
            f"seed {e.get('seed')}: {e.get('score', float('nan')):.2f}" for e in self
        )
        return f"RestartResults({len(self)} runs, best first - {rows})"


def fit_with_restarts(
    adata,
    model_params,
    trainer_params,
    layer,
    n_restarts: int = 3,
    cycle: Cycle = "cell_cycle",
    library_size_field=None,
    seeds=None,
    configs=None,
    on_restart=None,
    verbose: bool = False,
    keep_models: bool = True,
    fourier_warmstart=None,
    **train_kwargs,
):
    """Train several times from independent initialisations and rank the runs.

    Parameters
    ----------
    adata: data to train on.
    model_params: CoPhaser arguments.
    trainer_params: Trainer arguments, or the full auto_hyperparameters result.
    layer: layer of adata containing counts.
    n_restarts: number of independent runs.
    cycle: marker-gene cycle used for quality signals.
    seeds: optional seed for each restart. Defaults to range(n_restarts).
    verbose: show training loss.
    on_restart: optional callback run after each restart.
    keep_models: keep models/trainers in the results. If False, keep state_dict only.
    library_size_field: passed to load_anndata when needed.
    configs: optional trainer configurations cycled across restarts.
    train_kwargs: forwarded to Trainer.train_model.
    """
    if adata is None or model_params is None or trainer_params is None or layer is None:
        raise ValueError(
            "fit_with_restarts needs adata, model_params (CoPhaser arguments), "
            "trainer_params (Trainer arguments or auto_hyperparameters output) and layer."
        )
    if seeds is None:
        seeds = list(range(n_restarts))
    trainer_params = trainer_params.get("trainer", trainer_params)
    if fourier_warmstart is None:
        fourier_warmstart = (
            cycle == "cell_cycle"
            and float(trainer_params.get("cycling_status_prior", 1) or 1) < 1
        )
    elif fourier_warmstart and cycle != "cell_cycle":
        print("Warning: the Fourier warm start is RPE cell-cycle data; skipped for", cycle)
        fourier_warmstart = False

    from cophaser.model.CoPhaser import CoPhaser as _CoPhaser
    from cophaser.model import DecoderPrior
    from cophaser.loss import Loss as _Loss
    from cophaser.trainer import Trainer as _Trainer

    model_params = dict(model_params)
    trainer_params = dict(trainer_params)
    if fourier_warmstart:
        # otherwise the warm-started decoder stays frozen for the whole run
        trainer_params.setdefault("unfreeze_epoch_layer", [(20, "rhythmic_decoder")])

    def model_factory():
        model = _CoPhaser(**model_params)
        model.load_anndata(
            adata, layer_to_use=layer, library_size_field=library_size_field
        )
        nonlocal fourier_warmstart
        if fourier_warmstart:
            try:
                DecoderPrior.load_fourier_coefficients_prior(
                    resource_path("fourier_coefficients_RPE.csv"), model
                )
            except ValueError as e:
                print(f"Warning: Fourier warm start skipped ({e})")
                fourier_warmstart = False
        # lets quality_metrics check the adata it is given has the same cells
        model.obs_names = adata.obs_names.copy()
        return model

    n_epochs_each = int(train_kwargs.get("n_epochs", 200))
    user_on_epoch = train_kwargs.get("on_epoch")
    progress = tqdm(total=len(seeds) * n_epochs_each, desc="Training", unit="epoch")

    def _on_epoch(epoch, total, elbo):
        progress.update(1)
        if user_on_epoch is not None:
            user_on_epoch(epoch, total, elbo)

    train_kwargs = dict(train_kwargs, on_epoch=_on_epoch)
    if not verbose and train_kwargs.get("silent") is None:
        train_kwargs["silent"] = True

    report = []
    try:
        for i, seed in enumerate(seeds):
            torch.manual_seed(seed)
            np.random.seed(seed)
            run_started = time.time()
            model = model_factory()

            if i == 0:
                preflight(
                    adata,
                    layer=layer,
                    rhythmic_gene_names=getattr(model, "rhythmic_gene_names", None),
                    context_genes=getattr(model, "context_genes", None),
                    cycle=cycle,
                    library_size_field=library_size_field,
                    warn=True,
                )
                marker_gene_coverage(adata, cycle=cycle, context_genes=None, warn=True)

            config = configs[i % len(configs)] if configs else None
            trainer = _Trainer(
                model,
                _Loss.compute_loss,
                cycle=cycle,
                **{**trainer_params, **(config or {})},
            )
            trainer.train_model(**train_kwargs)

            metrics = quality_metrics(
                model,
                cycle=cycle,
                warn=False,
                adata=adata,
                layer=layer,
                library_size_field=library_size_field,
            )

            metrics["seed"] = seed
            metrics["minutes"] = (time.time() - run_started) / 60.0
            if config is not None:
                metrics["config"] = config
            if on_restart is not None:
                extra = on_restart(model, trainer, metrics)
                if isinstance(extra, dict):
                    metrics.update(extra)

            model.to("cpu")
            if keep_models:
                metrics["model"] = model
                metrics["trainer"] = trainer
            else:
                metrics["state_dict"] = copy.deepcopy(model.state_dict())
                metrics["model"] = None
                metrics["trainer"] = None
                metrics["rhythmic_likelihood_weight"] = (
                    trainer.rhythmic_likelihood_weight
                )
                metrics["non_rhythmic_likelihood_weight"] = (
                    trainer.non_rhythmic_likelihood_weight
                )
                del model, trainer
            report.append(metrics)
    finally:
        progress.close()

    groups = [i % len(configs) for i in range(len(report))] if configs else None
    scores = rank_candidates(report, groups=groups, cycle=cycle)
    for metrics, score in zip(report, scores):
        metrics["score"] = float(score)

    if not np.isfinite(scores).any():
        if len(report) < 2:
            print(
                "Note: a single run cannot be scored. "
                "Use n_restarts >= 2 to rank runs."
            )
        else:
            print(
                "Warning: no usable quality signal on any restart, so the "
                "runs are returned in seed order."
            )

        return RestartResults(report)

    order = sorted(
        range(len(report)),
        key=lambda i: (np.isnan(scores[i]), 0.0 if np.isnan(scores[i]) else -scores[i]),
    )
    ranked = RestartResults(report[i] for i in order)
    best = ranked[0]
    tied = sum(1 for entry in ranked if entry["score"] == best["score"])
    note = f" - {tied} runs tied" if tied > 1 else ""
    print(f"Best restart: seed {best['seed']} (score {best['score']:.2f}){note}")
    return ranked
