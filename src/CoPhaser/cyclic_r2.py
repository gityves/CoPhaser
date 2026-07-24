import anndata
import numpy as np
import pandas as pd
from scipy.sparse import issparse
from sklearn.preprocessing import LabelEncoder
import scanpy as sc


def _phase_coverage_weight(theta_cluster: np.ndarray) -> float:
    """
    Returns a weight in [0, 1] measuring how uniformly a cluster covers the
    circle [0, 2π).  A cluster whose cells all pile up in one phase arc gets
    a low weight; a uniformly covered cluster gets weight ≈ 1.

    Uses the circular uniformity score:  1 - |mean resultant length R|
    where R = 0 for a perfectly uniform distribution and R = 1 for all mass
    at a single point.
    """
    cos_mean = np.cos(theta_cluster).mean()
    sin_mean = np.sin(theta_cluster).mean()
    R = np.sqrt(cos_mean**2 + sin_mean**2)  # mean resultant length ∈ [0, 1]
    return float(1.0 - R)  # 1 = uniform, 0 = all same phase


def infer_cell_types(
    z: np.ndarray,
    n_neighbors: int = 15,
    resolution: float = 0.2,
    random_state: int = 0,
    min_cells_per_cluster: int = 50,
) -> np.ndarray:
    adata = anndata.AnnData(X=z)
    adata.obsm["X_latent"] = z
    sc.pp.neighbors(
        adata, use_rep="X_latent", n_neighbors=n_neighbors, random_state=random_state
    )
    sc.tl.leiden(
        adata,
        resolution=resolution,
        random_state=random_state,
        flavor="igraph",
        n_iterations=2,
    )
    labels = adata.obs["leiden"].astype(int).values
    labels = _merge_small_clusters_via_graph(
        labels, adata, min_cells_per_cluster=min_cells_per_cluster
    )  # merge tiny clusters into their neighbors
    return labels


def _merge_small_clusters_via_graph(
    labels: np.ndarray,
    adata,
    min_cells_per_cluster: int,
) -> np.ndarray:
    """
    Merge clusters smaller than `min_cells_per_cluster` into the cluster they are
    most strongly connected to in the kNN graph (i.e. the neighbor graph that
    Leiden itself used), rather than by Euclidean centroid distance.

    This is done iteratively: after each merge, cluster sizes and connectivity
    are recomputed, since a merge can change which cluster is "most connected"
    for the next tiny cluster, or leave a merged cluster still under threshold.
    """
    labels = labels.copy()
    # scanpy stores the kNN connectivity graph here after sc.pp.neighbors
    conn = adata.obsp["connectivities"]  # sparse (N, N), symmetric, weighted

    max_iters = len(np.unique(labels)) + 1  # safety bound
    for _ in range(max_iters):
        unique, counts = np.unique(labels, return_counts=True)
        if len(unique) <= 1:
            break
        small = unique[counts < min_cells_per_cluster]
        if len(small) == 0:
            break
        if len(small) >= len(unique):
            # everything is "small" -- nothing larger to merge into, stop
            break

        # merge the single smallest cluster first, then re-evaluate
        sizes = dict(zip(unique, counts))
        target = min(small, key=lambda c: sizes[c])

        mask_t = labels == target
        # sum connectivity weights from cells in `target` to cells in every
        # other cluster; merge into whichever cluster has the strongest total
        # connection (most graph "edge weight" reaching out of this cluster)
        rows = conn[mask_t]
        weight_by_cluster: dict[int, float] = {}
        # rows.indices / rows.data give us, per nonzero entry, which column
        # (cell) and what weight -- map columns back to their cluster label
        coo = rows.tocoo()
        for col, val in zip(coo.col, coo.data):
            lbl = labels[col]
            if lbl == target:
                continue
            weight_by_cluster[lbl] = weight_by_cluster.get(lbl, 0.0) + val

        if not weight_by_cluster:
            # isolated cluster with no outgoing edges (rare) -- fall back to
            # merging into the largest remaining cluster
            largest = unique[np.argmax(counts)]
            if largest == target:
                continue
            labels[mask_t] = largest
        else:
            best = max(weight_by_cluster, key=weight_by_cluster.get)
            labels[mask_t] = best

    return labels


def fit_cyclic_r2_celltypes(
    X,
    theta: np.ndarray,
    z: np.ndarray,
    gene_names: list[str] | None = None,
    K: int = 1,
    labels: np.ndarray | None = None,
    lambda_ridge: float = 1e-1,
    n_folds: int = 5,
    max_cells: int = 10_000,
    min_cells_per_cluster: int = 100,
    min_phase_coverage: float = 0.15,
    aggregation: str = "mean",
    top_n_genes: int = 100,
    return_beta: bool = False,
    return_cluster_r2: bool = False,
) -> pd.DataFrame | tuple:
    """
    Cyclic pseudo-R^2 that is robust to cell-type composition by computing
    per-cluster pseudo-R^2 and aggregating across clusters.

    For each cluster, the cyclic fit's out-of-fold "signal" and "noise" are
    computed (see `_cluster_r2`), and combined as a pseudo coefficient of
    determination:

        R^2 = signal / (signal + noise)

    A gene scores highly only if it shows a strong cyclic signal within
    multiple cell types.

    Parameters
    ----------
    X : (N, G) array-like
        Raw scRNA-seq counts.
    theta : (N,) array-like
        Fitted phase values for each cell (radians, [0, 2π)).
    z : (N, d) array-like
        Low-dimensional embedding used to identify cell populations
        (e.g. PCA, scVI, UMAP — anything that separates cell types).
    gene_names : list of str or None
    K : int
        Number of Fourier harmonics.
    labels : (N,) array-like of int or None
        Cell-type labels.  If None, will be inferred from z.
    lambda_ridge : float
        L2 regularisation for Poisson regression.
    n_folds : int
        Cross-validation folds within each cluster.
    max_cells : int
        Global cell cap (applied before clustering).
    min_cells_per_cluster : int
        Clusters with fewer cells are excluded from R^2 aggregation.
        Also used as the minimum cluster size floor in `infer_cell_types`.
        Must be > 2 * K + 1 (need enough cells per fold for phase fitting).
    min_phase_coverage : float in (0, 1]
        Clusters whose circular coverage weight (1 - R) falls below this
        threshold are excluded.  Increase this to demand more phase diversity.
    aggregation : {"weighted_harmonic", "weighted_mean", "min", "mean"}
        How to combine per-cluster R^2 values into a single gene-level score.

        "weighted_harmonic"  (recommended)
            Harmonic mean, weighted by effective cluster weight
            w_k = n_k x phase_coverage_k.
            Dominated by the worst well-sampled cluster — a gene must be
            cyclic in *all* major populations to score well.

        "weighted_mean"
            Arithmetic weighted mean.  More lenient: a very cyclic gene in
            a large cluster can compensate for low R^2 elsewhere.

        "min"
            Hard minimum across clusters.  Most conservative.

        "mean"
            Unweighted arithmetic mean.  Can be dominated by small clusters.
    top_n_genes : int
        For each cluster, only the top N genes by R^2 are considered being
        rhythmic in that cluster.  The final R^2 table includes a column
        "intersection" counting how many clusters each gene was present in.
        This is useful for filtering out genes that are only cyclic in a
        single cell type.

    return_beta : bool
        If True, also return full-data regression coefficients (G, p, n_clusters).
    return_cluster_r2 : bool
        If True, also return a DataFrame with per-cluster R^2 values.

    Returns
    -------
    r2_df : DataFrame with columns ["gene", "r2", "signal", "noise",
                                     "peak_phase", "n_clusters_used",
                                     "mean_cluster_size", "intersection"]
        "peak_phase" is the phase in [0, 2π) at which the gene peaks,
        aggregated across clusters as an R^2-weighted circular mean.
    cluster_r2 : DataFrame (optional, if return_cluster_r2=True)
        Per-cluster R^2 with columns ["gene", "cluster", "r2", "signal",
                                       "noise", "peak_phase", "n_cells",
                                       "phase_coverage"].
    beta : (G, p, n_clusters) array (optional, if return_beta=True)
    """
    theta = np.asarray(theta, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64)
    if labels is not None:
        labels = LabelEncoder().fit_transform(labels)

    N, G = X.shape
    assert len(theta) == N, "theta must have one entry per cell"
    assert z.shape[0] == N, "z must have one row per cell"

    if N > max_cells:
        idx = np.random.choice(N, max_cells, replace=False)
        X, theta, z = X[idx], theta[idx], z[idx]
        if labels is not None:
            labels = np.asarray(labels, dtype=int)[idx]
        N = max_cells

    if issparse(X):
        X = X.toarray()
        X = np.asarray(X, dtype=np.float64)

    s = X.sum(axis=1) / 1e4
    s = np.asarray(s, dtype=np.float64)
    log_s = np.log(np.maximum(s, 1e-8))

    if gene_names is None:
        gene_names = [f"gene_{i}" for i in range(G)]

    D = _make_design_matrix(theta, K)  # (N, 2K+1)

    if labels is None:
        labels = infer_cell_types(
            z,
            n_neighbors=15,
            resolution=0.2,
            random_state=0,
            min_cells_per_cluster=min_cells_per_cluster,
        )
    cluster_ids = np.unique(labels)

    cluster_records = []  # list of dicts, one per valid cluster
    beta_per_cluster = []  # only filled if return_beta

    for cid in cluster_ids:
        mask = labels == cid
        n_k = mask.sum()

        # ---- quality gates ----
        if n_k < min_cells_per_cluster:
            continue

        cov = _phase_coverage_weight(theta[mask])
        if cov < min_phase_coverage:
            continue

        # ---- compute pseudo R^2 for this cluster ----
        r2_k, sig_k, noise_k = _cluster_r2(
            X[mask],
            D[mask],
            log_s[mask],
            lambda_ridge=lambda_ridge,
            n_folds=min(n_folds, n_k // (2 * K + 2)),
        )

        # full-cluster fit → per-gene peaking phase (acrophase)
        beta_k = _fit_beta_irls(X[mask], D[mask], log_s[mask], lambda_ridge)
        peak_k = _peak_phase_from_beta(beta_k, K)  # (G,)

        cluster_records.append(
            {
                "cluster": cid,
                "n_cells": n_k,
                "phase_coverage": cov,
                "r2": r2_k,  # (G,)
                "signal": sig_k,  # (G,)
                "noise": noise_k,  # (G,)
                "peak_phase": peak_k,  # (G,)
            }
        )

        if return_beta:
            beta_per_cluster.append(beta_k)  # (G, p)

    if len(cluster_records) == 0:
        raise ValueError(
            "No clusters passed the quality gates.  "
            "Try lowering min_cells_per_cluster or min_phase_coverage."
        )

    # ------------------------------------------------------------------
    # Aggregate across clusters
    # ------------------------------------------------------------------
    n_used = len(cluster_records)
    ns = np.array([r["n_cells"] for r in cluster_records], dtype=float)  # (C,)
    covs = np.array([r["phase_coverage"] for r in cluster_records], dtype=float)  # (C,)
    r2s = np.stack([r["r2"] for r in cluster_records], axis=0)  # (C, G)
    sigs = np.stack([r["signal"] for r in cluster_records], axis=0)  # (C, G)
    noises = np.stack([r["noise"] for r in cluster_records], axis=0)  # (C, G)

    # effective weight = n_cells × phase_coverage
    w = ns * covs  # (C,)
    w = w / w.sum()  # normalise to sum to 1

    if aggregation == "weighted_harmonic":
        # harmonic mean:  1 / Σ(w_k / r2_k)
        # guard against r2 = 0 → treat as very small positive
        safe_r2 = np.maximum(r2s, 1e-12)
        agg_r2 = 1.0 / (w[:, None] / safe_r2).sum(0)
    elif aggregation == "weighted_mean":
        agg_r2 = (w[:, None] * r2s).sum(0)
    elif aggregation == "mean":
        agg_r2 = r2s.mean(0)
    elif aggregation == "min":
        agg_r2 = r2s.min(0)
    else:
        raise ValueError(f"Unknown aggregation: {aggregation!r}")

    # intersection: return the number of clusters in which each gene was in the top_n_genes
    top_n_mask = np.zeros_like(r2s, dtype=bool)  # (C, G)
    for c in range(n_used):
        top_n_mask[c, np.argsort(-r2s[c])[:top_n_genes]] = True
    intersection = top_n_mask.sum(0)
    # signal & noise: always use weighted arithmetic mean for interpretability
    agg_sig = (w[:, None] * sigs).sum(0)
    agg_noise = (w[:, None] * noises).sum(0)

    # peaking phase: weighted circular mean of per-cluster acrophases.
    # Weight each cluster's phase by w_k and by its R^2 so that clusters where
    # the gene is barely cyclic (noisy, meaningless phase) contribute little.
    phases = np.stack([r["peak_phase"] for r in cluster_records], axis=0)  # (C, G)
    phase_w = w[:, None] * np.maximum(r2s, 0.0)  # (C, G)
    cos_sum = (phase_w * np.cos(phases)).sum(0)
    sin_sum = (phase_w * np.sin(phases)).sum(0)
    agg_phase = np.mod(np.arctan2(sin_sum, cos_sum), 2 * np.pi)  # (G,) in [0, 2π)

    r2_df = pd.DataFrame(
        {
            "gene": gene_names,
            "r2": agg_r2,
            "signal": agg_sig,
            "noise": agg_noise,
            "peak_phase": agg_phase,
            "n_clusters_used": n_used,
            "mean_cluster_size": int(ns.mean()),
            "intersection": intersection,
        }
    )

    # ------------------------------------------------------------------
    # 4. Optional extras
    # ------------------------------------------------------------------
    extras = []

    if return_cluster_r2:
        rows = []
        for r in cluster_records:
            for g, gn in enumerate(gene_names):
                rows.append(
                    {
                        "gene": gn,
                        "cluster": r["cluster"],
                        "r2": r["r2"][g],
                        "signal": r["signal"][g],
                        "noise": r["noise"][g],
                        "peak_phase": r["peak_phase"][g],
                        "n_cells": r["n_cells"],
                        "phase_coverage": r["phase_coverage"],
                    }
                )
        cluster_r2_df = pd.DataFrame(rows)
        extras.append(cluster_r2_df)

    if return_beta:
        beta_arr = np.stack(beta_per_cluster, axis=-1)  # (G, p, n_clusters)
        extras.append(beta_arr)

    if extras:
        return (r2_df, *extras)
    return r2_df


def _make_design_matrix(theta: np.ndarray, K: int) -> np.ndarray:
    """Intercept + K sine/cosine pairs → shape (N, 2K+1)."""
    cols = [np.ones(len(theta))]
    for k in range(1, K + 1):
        cols.append(np.cos(k * theta))
        cols.append(np.sin(k * theta))
    return np.column_stack(cols)


def _peak_phase_from_beta(beta: np.ndarray, K: int, n_grid: int = 256) -> np.ndarray:
    """
    Peaking phase (acrophase) per gene: the phase θ ∈ [0, 2π) at which the
    fitted cyclic log-mean is maximal.

    For K = 1 this is the closed form atan2(β_sin, β_cos), but evaluating the
    reconstructed curve on a dense grid handles any number of harmonics
    (multi-harmonic curves can peak away from the first-harmonic acrophase).

    Parameters
    ----------
    beta : (G, 2K+1) array
        Regression coefficients [intercept, cos1, sin1, cos2, sin2, ...].
    K : int
        Number of Fourier harmonics.
    n_grid : int
        Resolution of the phase grid used to locate the maximum.

    Returns
    -------
    peak_phase : (G,) array in [0, 2π)
    """
    grid = np.linspace(0.0, 2.0 * np.pi, n_grid, endpoint=False)
    D_grid = _make_design_matrix(grid, K)  # (n_grid, 2K+1)
    # cyclic part of the log-mean (intercept is constant → irrelevant to argmax)
    curve = D_grid @ beta.T  # (n_grid, G)
    return grid[np.argmax(curve, axis=0)]  # (G,)


def _fit_beta_irls(
    X_sub: np.ndarray,
    D_sub: np.ndarray,
    log_s_sub: np.ndarray,
    lambda_ridge: float,
) -> np.ndarray:
    """
    Log-OLS initialisation + one Poisson IRLS step, batched over genes.
    Returns beta of shape (G, p).
    """
    p = D_sub.shape[1]
    G = X_sub.shape[1]

    # OLS init on log scale
    Y = np.log(X_sub + 0.1) - log_s_sub[:, None]
    DTD = D_sub.T @ D_sub
    beta = np.linalg.solve(
        DTD + lambda_ridge * np.diag(np.diag(DTD)), D_sub.T @ Y
    ).T  # (G, p)

    # one IRLS step
    eta = log_s_sub[:, None] + D_sub @ beta.T  # (N, G)
    mu = np.exp(np.clip(eta, -30, 30))
    t = (D_sub @ beta.T) + (X_sub - mu) / np.maximum(mu, 1e-8)
    w = mu  # (N, G) Poisson weights

    DTWD = np.empty((G, p, p))
    for i in range(p):
        for j in range(i, p):
            v = (D_sub[:, i] * D_sub[:, j]) @ w  # (G,)
            DTWD[:, i, j] = v
            DTWD[:, j, i] = v

    DTWz = (D_sub.T @ (w * t)).T  # (G, p)
    d = np.arange(p)
    DTWD[:, d, d] += lambda_ridge * DTWD[:, d, d]
    beta = np.linalg.solve(DTWD, DTWz[..., None])[..., 0]  # (G, p)
    return beta


def _cluster_r2(
    X_c: np.ndarray,
    D_c: np.ndarray,
    log_s_c: np.ndarray,
    lambda_ridge: float,
    n_folds: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Cross-validated cyclic pseudo-R^2 for a single cluster.

    "signal" is the (cross-fitted) covariance between the residual-from-null
    counts and the cyclic model's improvement over the null; "noise" is the
    held-out residual variance of the cyclic fit itself. The pseudo-R^2 is

        R^2 = signal / (signal + noise)

    which sits in roughly [0, 1): 0 means the cyclic model explains none of
    the variance beyond the flat (non-cyclic) null, values approaching 1
    mean the cyclic fit accounts for nearly all of the variance relative to
    its own residual noise.

    Returns (r2, signal_var, noise_var) each of shape (G,).
    """
    N_c, G = X_c.shape
    n_folds = max(n_folds, 2)

    fold = np.arange(N_c) % n_folds
    signal_num = np.zeros(G)
    noise_ss = np.zeros(G)

    for f in range(n_folds):
        te = fold == f
        tr = ~te
        if tr.sum() < D_c.shape[1] + 1:
            continue  # skip degenerate folds

        beta = _fit_beta_irls(X_c[tr], D_c[tr], log_s_c[tr], lambda_ridge)

        mu_hat_te = np.exp(np.clip(log_s_c[te, None] + D_c[te] @ beta.T, -30, 30))
        c = X_c[tr].sum(0) / max(np.exp(log_s_c[tr]).sum(), 1e-8)
        mu_null_te = np.exp(log_s_c[te, None]) * c

        dd = X_c[te] - mu_null_te
        ff = mu_hat_te - mu_null_te
        signal_num += (dd * ff).sum(0)
        noise_ss += ((X_c[te] - mu_hat_te) ** 2).sum(0)

    signal_var = np.maximum(signal_num / N_c, 0.0)
    noise_var = noise_ss / N_c
    r2 = signal_var / np.maximum(signal_var + noise_var, 1e-12)
    return r2, signal_var, noise_var
