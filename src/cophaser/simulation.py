import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix


def get_mu(A0, A, B, lambda_, n_cells, thetas=100):
    """
    Compute mean expression fractions from Fourier coefficients.

    Parameters
    ----------
    A0 : np.ndarray
        (n_cells, n_genes) array of A_0 coefficients or (1, n_genes) if shared across cells.
    A : np.ndarray
        (n_genes, harmonics) array of A_k coefficients (shared across cells).
    B : np.ndarray
        (n_genes, harmonics) array of B_k coefficients (shared across cells).
    lambda_ : np.ndarray
        (n_cells,) array of scaling factors for harmonic amplitude, or scalar if shared across cells.
    thetas : int or np.ndarray
        Number of theta points to compute, or array of theta values.

    Returns
    -------
    mu : np.ndarray
        (n_cells, n_genes, n_theta_differents) array of mean expression fractions.
    theta_values : np.ndarray
        (n_theta_differents,) array of theta values.
    """
    n_genes, harmonics = A.shape
    assert B.shape == (n_genes, harmonics), "A and B must have the same shape"
    if A0.shape[0] == 1:
        A0 = A0.repeat(n_cells, axis=0)
    elif len(A0.shape) == 1:
        A0 = A0[None, :].repeat(n_cells, axis=0)
    else:
        assert (
            A0.shape[0] == n_cells
        ), "A0 must have shape (n_cells, n_genes) or (1, n_genes)"
    assert A0.shape[1] == n_genes, "A0 must have shape (n_cells, n_genes)"
    if np.isscalar(lambda_):
        lambda_ = np.full(n_cells, lambda_)
    else:
        assert len(lambda_) == n_cells, "lambda_ must have length n_cells"
        lambda_ = np.array(lambda_).squeeze()

    # Create theta values
    if isinstance(thetas, int):
        theta_values = np.linspace(-np.pi, np.pi, thetas, endpoint=False)
    else:
        theta_values = np.array(thetas).squeeze()

    # Precompute trigonometric bases
    k_vals = np.arange(1, harmonics + 1)
    cos_terms = np.cos(np.outer(k_vals, theta_values))  # (harmonics, n_theta)
    sin_terms = np.sin(np.outer(k_vals, theta_values))  # (harmonics, n_theta)

    # Compute harmonic contributions per gene and theta
    # Shape: (n_genes, n_theta)
    expr_theta = np.einsum("gh,ht->gt", A, cos_terms) + np.einsum(
        "gh,ht->gt", B, sin_terms
    )

    # Add constant term A0 (per cell, per gene) and scale by lambda_ (per cell)
    # A0[..., None]: (n_cells, n_genes, 1)
    # lambda_[:, None, None]: (n_cells, 1, 1)
    expr = A0[..., None] + lambda_[:, None, None] * expr_theta[None, :, :]

    mu = np.exp(expr)  # Exponentiate to get mean fractions

    return mu, theta_values


def align_mu_pred(mu_pred, results_shifts):
    """Shift and flip mu_pred according to results_shifts dictionary."""
    if results_shifts["direction"] == -1:
        mu_pred = mu_pred[:, :, ::-1]
    # shift mu_pred
    n_theta = mu_pred.shape[1]
    shift_amount = int(
        (results_shifts["shift"] / (2 * np.pi)) * n_theta
    )  # convert shift from radians to index
    mu_pred = np.roll(mu_pred, -shift_amount, axis=-1)
    return mu_pred


def get_mu_pred(model, generative_outputs, space_outputs, results_shifts, n_theta=100):
    A0 = (
        model.mean_genes[model.rhythmic_gene_indices].detach().numpy()
        + generative_outputs["Z"].detach().numpy()[:, model.rhythmic_gene_indices]
    )
    A = (
        model.rhythmic_decoder.fourier_coefficients.weight[
            model.rhythmic_gene_indices, 0::2
        ]
        .detach()
        .numpy()
    )
    B = (
        model.rhythmic_decoder.fourier_coefficients.weight[
            model.rhythmic_gene_indices, 1::2
        ]
        .detach()
        .numpy()
    )
    lambda_ = generative_outputs["lambda"].detach().numpy()
    theta_values = np.linspace(-np.pi, np.pi, n_theta, endpoint=False)
    mu_pred, theta_pred = get_mu(
        A0=A0, A=A, B=B, lambda_=lambda_, n_cells=A0.shape[0], thetas=theta_values
    )
    mu_pred = align_mu_pred(mu_pred, results_shifts)
    # set mu_pred of non cycling cells to A0 only
    non_cycling_mask = space_outputs["b_z"].detach().numpy() < 0.5
    mu_pred[non_cycling_mask, :, :] = np.exp(A0[non_cycling_mask, :, None])
    return mu_pred, theta_pred


def simulate_counts_rhythmic(
    fourrier_coefficients: pd.DataFrame,
    n_cells: int,
    alphas=0.1,
    lambda_=1.0,
    library_size=1e4,
    n_theta_differents=100,
) -> csr_matrix:
    """
    Generate scRNA-seq-like counts from Fourier-based gene expression profiles.

    Parameters
    ----------
    fourrier_coefficients : pd.DataFrame
        Rows = genes, columns = Fourier coefficients (A_0, A_1, B_1, A_2, B_2, ...), from the log space fractions model.
    n_cells : int
        Number of cells to simulate.
    alphas : float or array-like
        Dispersion parameter(s) of the negative binomial.
        - If float: shared across all genes
        - If array-like: one value per gene (matching index order).
    lambda_ : float
        Amplitude of the oscillations.
    library_size : float
        Total counts per cell (mean library size).
    n_theta_differents : int
        Number of different theta points to simulate.

    Returns
    -------
    csr_matrix
        Sparse counts matrix of shape (n_cells, n_genes).
    np.ndarray
        Array of assigned times for each cell, shape (n_cells,).
    np.ndarray
        Array of total fractions of counts per cell, shape (n_cells,).
    """
    genes = fourrier_coefficients.index
    n_genes = len(genes)

    alphas = _handle_alphas(alphas, n_genes)

    mu, theta_values = get_mu(
        A0=fourrier_coefficients["A_0"].values,
        A=fourrier_coefficients.filter(like="A_").sort_index(axis=1).values[:, 1:],
        B=fourrier_coefficients.filter(like="B_").sort_index(axis=1),
        lambda_=lambda_,
        n_cells=n_cells,
        thetas=n_theta_differents,
    )
    # Assign each cell a random time point
    assigned_times_i = np.random.choice(n_theta_differents, size=n_cells, replace=True)
    assigned_times = theta_values[assigned_times_i]

    # for all cells simulate counts only for their assigned time
    mu_simulate = mu[np.arange(n_cells), :, assigned_times_i].squeeze()
    counts = simulate_counts(
        mean_fractions=mu_simulate,
        alphas=alphas,
        library_size=library_size,
    )

    return counts, assigned_times, mu_simulate.sum(axis=1), mu


def simulate_counts_rhythmic_shifting(
    fourrier_coefficients: pd.DataFrame,
    n_cells: int,
    pseudotimes: np.ndarray,
    alphas=0.1,
    lambda_=(1.0, 1.0),
    library_size=1e4,
    n_theta_differents=100,
):
    """
    Generate scRNA-seq-like counts from Fourier-based gene expression profiles,
    with gradual change in lambda and A_0 according to pseudotimes.

    Parameters
    ----------
    fourrier_coefficients : pd.DataFrame
        Rows = genes, columns = Fourier coefficients (A_0, A_0_DIFF, A_1, B_1, A_2, B_2, ...),
        from the log space fractions model.
    n_cells : int
        Number of cells to simulate.
    pseudotimes : array-like, shape (n_cells,)
        Values between 0 and 1 indicating progression from baseline (0) to endpoint (1).
    alphas : float or array-like
        Dispersion parameter(s) of the negative binomial.
        - If float: shared across all genes
        - If array-like: one value per gene (matching index order).
    lambda_ : list or tuple of two floats
        [lambda_start, lambda_end], interpolated according to pseudotimes.
    library_size : float
        Total counts per cell (mean library size).

    Returns
    -------
    csr_matrix
        Sparse counts matrix of shape (n_cells, n_genes).
    np.ndarray
        Array of assigned times for each cell, shape (n_cells,).
    np.ndarray
        Array of total fractions of counts per cell, shape (n_cells,).
    """
    genes = fourrier_coefficients.index
    n_genes = len(genes)

    alphas = _handle_alphas(alphas, n_genes)

    if len(pseudotimes) != n_cells:
        raise ValueError("pseudotimes must have length n_cells")
    if not (isinstance(lambda_, (list, tuple)) and len(lambda_) == 2):
        raise ValueError("lambda_ must be a list or tuple of length 2")

    lambda_start, lambda_end = lambda_
    lambda_vals = (1 - pseudotimes) * lambda_start + pseudotimes * lambda_end
    A0_interp = (
        pseudotimes[:, None] * fourrier_coefficients["A_0"].values[None, :]
        + (1 - pseudotimes[:, None]) * fourrier_coefficients["A_0_DIFF"].values[None, :]
    )

    mu, theta_values = get_mu(
        A0=A0_interp,
        A=fourrier_coefficients.filter(like="A_").sort_index(axis=1).values[:, 2:],
        B=fourrier_coefficients.filter(like="B_").sort_index(axis=1),
        lambda_=lambda_vals,
        n_cells=n_cells,
        thetas=n_theta_differents,
    )

    # Assign each cell a random time point
    assigned_times_i = np.random.choice(n_theta_differents, size=n_cells, replace=True)
    assigned_times = theta_values[assigned_times_i]

    # for all cells simulate counts only for their assigned time
    mu_simulate = mu[np.arange(n_cells), :, assigned_times_i].squeeze()
    counts = simulate_counts(
        mean_fractions=mu_simulate,
        alphas=alphas,
        library_size=library_size,
    )

    return counts, assigned_times, mu_simulate.sum(axis=1), mu


def _handle_alphas(alphas, n_genes):
    if np.isscalar(alphas):
        return np.full(n_genes, alphas)
    else:
        alphas = np.asarray(alphas)
        if len(alphas) != n_genes:
            raise ValueError("Length of alphas must match number of genes")
        return alphas


def simulate_counts(
    mean_fractions: np.ndarray,
    alphas=0.1,
    library_size=1e4,
) -> csr_matrix:
    """
    Generate scRNA-seq-like counts from mean expression fractions, using a negative binomial model.
    Parameters
    ----------
    mean_fractions : np.ndarray
        Matrix of shape (n_cells, n_genes) with mean expression fractions for each gene at each time point.
    alphas : float or array-like
        Dispersion parameter(s) of the negative binomial.
        - If float: shared across all genes
        - If array-like: one value per gene (matching index order).
    library_size : float
        Total counts per cell (mean library size).
    Returns
    -------
    csr_matrix
        Sparse counts matrix of shape (n_cells, n_genes).
    np.ndarray
        Array of assigned times for each cell, shape (n_cells,).
    """
    n_genes = mean_fractions.shape[1]

    alphas = _handle_alphas(alphas, n_genes)
    # Scale mean fractions by library size
    mu = mean_fractions * library_size
    # Generate counts using negative binomial
    counts = np.zeros_like(mu, dtype=int)
    for g in range(n_genes):
        p = 1 / (1 + alphas[g] * mu[:, g])
        r = 1 / alphas[g]
        counts[:, g] = np.random.negative_binomial(r, p)
    return csr_matrix(counts)


def scale_fractions(
    fractions: np.ndarray, target: float | np.ndarray = 1
) -> np.ndarray:
    """
    Scale mean expression fractions so that they sum to a target value for each cell.

    Parameters
    ----------
    fractions : np.ndarray
        Matrix of shape (n_cells, n_genes) with mean expression fractions.

    Returns
    -------
    np.ndarray
        Scaled fractions matrix of the same shape.
    """
    if np.isscalar(target):
        target = np.full(fractions.shape[0], target)
    elif len(target) != fractions.shape[0]:
        raise ValueError(
            f"Length of target must match number of cells ({len(target)} != {fractions.shape[0]})"
        )
    print(fractions.shape, target.shape)
    row_sums = fractions.sum(axis=1, keepdims=True)
    # Avoid division by zero
    row_sums[row_sums == 0] = 1
    return fractions / row_sums * target[:, None]


def generate_differently_expressed_counts(
    n_genes: int,
    n_cells: int,
    fold_changes: np.ndarray,
    target_fractions: np.ndarray,
    alphas=0.1,
    library_size=1e4,
    base_fractions_unscaled=None,
) -> csr_matrix:
    """
    Generate scRNA-seq-like counts with differentially expressed genes using a negative binomial model.

    Parameters
    ----------
    n_genes : int
        Total number of genes.
    n_cells : int
        Number of cells to simulate.
    fold_changes : list
        List of log2 fold changes for the differentially expressed genes.
    alphas : float or array-like
        Dispersion parameter(s) of the negative binomial.
        - If float: shared across all genes
        - If array-like: one value per gene (matching index order).
    library_size : float
        Total counts per cell (mean library size).
    Returns
    -------
    csr_matrix
        Sparse counts matrix of shape (n_cells, n_genes).
    """
    # ensure fold_changes length does not exceed n_genes
    if len(fold_changes) > n_genes:
        raise ValueError("Length of fold_changes cannot exceed n_genes")
    if base_fractions_unscaled is None:
        simulate_base_condition = True
        n_cells = n_cells // 2  # Half cells for each condition
        # Base mean fractions log uniformly distributed between -7 and -4
        base_log_fractions = np.random.uniform(-7, -4, size=n_genes)
        base_fractions_unscaled = 10 ** (base_log_fractions)
        # Scale to target fractions
        base_fractions = np.tile(base_fractions_unscaled, (n_cells, 1))
        base_fractions = scale_fractions(
            base_fractions, target=target_fractions[:n_cells]
        )
        # base counts
        base_counts = simulate_counts(
            mean_fractions=base_fractions,
            alphas=alphas,
            library_size=library_size,
        )
    else:
        simulate_base_condition = False
    # Convert log fold changes to linear scale
    fold_changes_linear = 2 ** np.array(fold_changes)
    # Apply fold changes to the mean fractions for the specified genes
    base_fractions_unscaled[: len(fold_changes)] *= fold_changes_linear
    # Scale to target fractions
    altered_fractions = np.tile(base_fractions_unscaled, (n_cells, 1))
    if simulate_base_condition:
        altered_fractions = scale_fractions(
            altered_fractions, target=target_fractions[n_cells:]
        )
    else:
        altered_fractions = scale_fractions(altered_fractions, target=target_fractions)
    # altered counts
    altered_counts = simulate_counts(
        mean_fractions=altered_fractions,
        alphas=alphas,
        library_size=library_size,
    )
    if simulate_base_condition:
        # Combine counts from both conditions by stacking
        combined_counts = np.zeros((n_cells * 2, n_genes), dtype=int)
        combined_counts[:n_cells, :] = base_counts.toarray()
        combined_counts[n_cells:, :] = altered_counts.toarray()
        # Convert to sparse matrix
        combined_counts = csr_matrix(combined_counts)
        return combined_counts
    return csr_matrix(altered_counts)


def generate_gradual_shifting_counts(
    n_genes: int,
    n_cells: int,
    fold_changes: np.ndarray,
    target_fractions: np.ndarray,
    pseudotimes: np.ndarray,
    alphas=0.1,
    library_size=1e4,
) -> csr_matrix:
    """
    Generate scRNA-seq-like counts with gradual differential expression
    using a negative binomial model, controlled by pseudotimes.

    Parameters
    ----------
    n_genes : int
        Total number of genes.
    n_cells : int
        Number of cells to simulate.
    fold_changes : list or array-like
        List of log2 fold changes for the differentially expressed genes.
    target_fractions : array-like, shape (n_cells,)
        Target total fractions for each cell.
    pseudotimes : array-like, shape (n_cells,)
        Values between 0 and 1 indicating progression from baseline (0)
        to fully differentially expressed (1).
    alphas : float or array-like
        Dispersion parameter(s) of the negative binomial.
        - If float: shared across all genes
        - If array-like: one value per gene (matching index order).
    library_size : float
        Total counts per cell (mean library size).

    Returns
    -------
    csr_matrix
        Sparse counts matrix of shape (n_cells, n_genes).
    """
    if len(fold_changes) > n_genes:
        raise ValueError("Length of fold_changes cannot exceed n_genes")
    if len(pseudotimes) != n_cells:
        raise ValueError("pseudotimes must have length n_cells")

    # Base mean fractions log uniformly distributed between -7 and -4
    base_log_fractions = np.random.uniform(-7, -4, size=n_genes)
    base_fractions_unscaled = 10**base_log_fractions

    # Convert log fold changes to linear scale
    fold_changes_linear = 2 ** np.array(fold_changes)

    # Expand base fractions to all cells
    fractions = np.tile(base_fractions_unscaled, (n_cells, 1))

    # For DE genes, interpolate between baseline (1x) and altered (fold_changes_linear)
    for g, fc in enumerate(fold_changes_linear):
        fractions[:, g] *= (1 - pseudotimes) + pseudotimes * fc

    # Scale to match target fractions for each cell
    fractions = scale_fractions(fractions, target=target_fractions)

    # Simulate counts
    counts = simulate_counts(
        mean_fractions=fractions,
        alphas=alphas,
        library_size=library_size,
    )

    return csr_matrix(counts), base_fractions_unscaled
