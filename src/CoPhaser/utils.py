import scanpy as sc
import pandas as pd
import numpy as np
from scipy.spatial.distance import jensenshannon
from scipy.optimize import differential_evolution
from sklearn.metrics import mutual_info_score
from scipy.stats import circmean
import anndata
import matplotlib.pyplot as plt
import torch
from CoPhaser import gene_sets


def get_variable_genes(adata, n_variable_genes=2000, layer=None, min_cells=3):
    adata = adata.copy()
    if layer:
        adata.X = adata.layers[layer]
    sc.pp.filter_genes(adata, min_cells=min_cells)
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata, n_top_genes=n_variable_genes)
    variable_genes_name = adata.var_names[adata.var["highly_variable"]].tolist()
    return variable_genes_name


def add_histones_fraction(adata, layer="spliced", use_only_clustered=False):
    if use_only_clustered:
        histones_genes = gene_sets.human_canonical_histones
        histones_genes_in_data = list(
            set(histones_genes).intersection(set(adata.var_names))
        )
        adata.obs["histones_fraction"] = (
            adata[:, histones_genes_in_data].layers[layer].sum(axis=1)
        ) / adata.layers[layer].sum(axis=1)
    else:
        query = "Hist"
        if adata.var_names[int(len(adata.var_names) / 2)].isupper():
            query = query.upper()
        adata.obs["histones_fraction"] = (
            adata[:, adata.var_names.str.contains(query)].layers[layer].sum(axis=1)
        ) / adata.layers[layer].sum(axis=1)


def get_best_shift_direction(thetas, phases):
    """
    Determine the optimal shift and direction to align predicted phases with true phases.
    """
    results_shifts = {"l1": [], "shift": [], "direction": []}
    valid_indices = ~np.isnan(phases)
    thetas = thetas[valid_indices]
    phases = phases[valid_indices]
    for direction_theta in [1, -1]:
        theta_shift = circmean(
            normalize_angles(direction_theta * thetas - phases), high=np.pi, low=-np.pi
        )
        results_shifts["l1"].append(
            np.abs(
                normalize_angles(direction_theta * thetas - phases - theta_shift)
            ).mean()
        )
        results_shifts["shift"].append(theta_shift)
        results_shifts["direction"].append(direction_theta)
    results_shifts = pd.DataFrame(results_shifts)
    results_shifts = results_shifts.sort_values(by="l1").iloc[0]
    return results_shifts


def align_phases(thetas, phases):
    """
    Align predicted phases (thetas) with true phases by finding the best shift and direction.
    """
    results_shifts = get_best_shift_direction(thetas, phases)
    aligned_thetas = normalize_angles(
        results_shifts["direction"] * thetas - results_shifts["shift"]
    )
    return aligned_thetas


def circular_std(data, axis=1):
    """
    Computes the circular standard deviation of angle in rad.
    """
    # Ensure input is a NumPy array
    data = np.asarray(data)

    # Convert angles to unit vectors
    sin_sum = np.sum(np.sin(data), axis=axis)
    cos_sum = np.sum(np.cos(data), axis=axis)

    # Compute mean resultant length
    R = np.sqrt(sin_sum**2 + cos_sum**2) / data.shape[axis]

    # Circular standard deviation
    circ_std = np.sqrt(-2 * np.log(R))
    return circ_std


def get_ptp_phase(
    df_rhythmic,
    columns_not_gene=["inferred_theta", "pca_theta"],
    column_theta="inferred_theta",
):
    # Separate genes and inferred_theta
    genes = df_rhythmic.drop(columns=columns_not_gene, errors="ignore")
    theta = df_rhythmic[column_theta]

    # Calculate peak-to-peak (max - min) for all genes
    peak_to_peak = genes.max() - genes.min()

    # Find the index of the maximum value for each gene
    max_indices = genes.idxmax()

    # Get the corresponding phase (theta) for each gene's max value
    phase = theta[max_indices.values].values

    # Create the results DataFrame
    return pd.DataFrame(
        {"peak_to_peak": peak_to_peak.values, "phase": phase}, index=genes.columns
    )


def q3_minus_q1(X: torch.Tensor) -> torch.Tensor:
    """
    Computes Q3 - Q1 for each feature (column) in the input tensor.

    Parameters:
    -----------
    X : torch.Tensor
        A 2D tensor of shape (n_cells, n_genes), where each column is a feature.

    Returns:
    --------
    torch.Tensor
        A 1D tensor of shape (n_genes,) containing Q3 - Q1 for each gene.
    """
    # Compute 75th percentile (Q3) for each gene
    q3 = torch.quantile(X, 0.75, dim=0)

    # Compute Q1 for each gene
    q1 = torch.quantile(X, 0.25, dim=0)

    # Return Q3 - Q1
    return q3 - q1


def pca_torch(X, n_components):
    # Center the data
    X_centered = X - X.mean(dim=0, keepdim=True)

    # Compute covariance matrix
    covariance_matrix = torch.matmul(X_centered.T, X_centered) / (
        X_centered.shape[0] - 1
    )

    # Eigendecomposition
    eigvals, eigvecs = torch.linalg.eigh(covariance_matrix)

    # Sort by descending eigenvalue
    sorted_indices = torch.argsort(eigvals, descending=True)
    eigvecs = eigvecs[:, sorted_indices]

    # Project onto top components
    components = eigvecs[:, :n_components]  # shape: (n_genes, n_components)
    X_pca = torch.matmul(X_centered, components)  # shape: (n_cells, n_components)

    return X_pca, components, X.mean(dim=0)


def get_jensenshannon_raw(pseudotime_values, hue_values, n_bins=50):
    # Remove NaN values
    valid_idx = ~pd.isna(hue_values)
    pseudotime_values = pseudotime_values[valid_idx]
    hue_values = hue_values[valid_idx]

    # Check that there are exactly two unique values in hue
    unique_hue_values = np.unique(hue_values)
    if len(unique_hue_values) != 2:
        raise ValueError(
            f"Expected exactly 2 unique values in hue column, found {len(unique_hue_values)}: {unique_hue_values}"
        )

    hue_values = hue_values == unique_hue_values[0]

    # Separate the two categories of cells
    p_pseudotimes = pseudotime_values[hue_values]
    q_pseudotimes = pseudotime_values[~hue_values]

    # Discretize pseudotimes into bins
    bins = np.linspace(-np.pi, np.pi, n_bins + 1)
    p_pseudotimes = (p_pseudotimes + np.pi) % (2 * np.pi) - np.pi
    q_pseudotimes = (q_pseudotimes + np.pi) % (2 * np.pi) - np.pi

    # Compute histograms (normalized to probabilities)
    p, _ = np.histogram(p_pseudotimes, bins=bins, density=True)
    q, _ = np.histogram(q_pseudotimes, bins=bins, density=True)

    # Compute Jensen-Shannon divergence using scipy
    js_divergence = jensenshannon(p, q, 2)

    return js_divergence


def get_jensenshannon(adata, pseudotime_column, hue, n_bins=50):
    """
    Compute the Jensen-Shannon divergence between two categories of cells
    based on their pseudotime_column distribution.

    Parameters:
    - adata: AnnData object containing single-cell RNA-seq data
    - pseudotime_column: Name of the pseudotime column in adata.obs
    - hue: Name of the column in adata separating the two type of cells

    Returns:
    - js_divergence: Jensen-Shannon divergence between the two distribution of cells
    """
    pseudotime_values = adata.obs[pseudotime_column].values
    hue_values = adata.obs[hue].values

    # Remove NaN values
    valid_idx = ~pd.isna(hue_values)
    pseudotime_values = pseudotime_values[valid_idx]
    hue_values = hue_values[valid_idx]

    # Check that there are exactly two unique values in hue
    unique_hue_values = np.unique(hue_values)
    if len(unique_hue_values) != 2:
        raise ValueError(
            f"Expected exactly 2 unique values in hue column, found {len(unique_hue_values)}: {unique_hue_values}"
        )

    hue_values = hue_values == unique_hue_values[0]

    # Separate the two categories of cells
    p_pseudotimes = pseudotime_values[hue_values]
    q_pseudotimes = pseudotime_values[~hue_values]

    # Discretize pseudotimes into bins
    bins = np.linspace(-np.pi, np.pi, n_bins + 1)
    p_pseudotimes = (p_pseudotimes + np.pi) % (2 * np.pi) - np.pi
    q_pseudotimes = (q_pseudotimes + np.pi) % (2 * np.pi) - np.pi

    # Compute histograms (normalized to probabilities)
    p, _ = np.histogram(p_pseudotimes, bins=bins, density=True)
    q, _ = np.histogram(q_pseudotimes, bins=bins, density=True)

    # Compute Jensen-Shannon divergence using scipy
    js_divergence = jensenshannon(p, q, 2)

    return js_divergence


def pseudotime_mutual_information(category, pseudotime, n_bins=50):
    # Discretize pseudotime
    pt_bins = pd.qcut(pseudotime, q=n_bins, duplicates="drop")
    return mutual_info_score(pt_bins, category)


def mean_angle_rad(theta1, theta2):
    """
    Compute the circular mean of two angles in radians.

    Parameters:
    - theta1, theta2: float or np.array of angles in radians (in [-π, π] or any range)

    Returns:
    - mean_angle: circular mean angle in radians, in [-π, π]
    """
    # Convert to complex representation
    z1 = np.exp(1j * theta1)
    z2 = np.exp(1j * theta2)

    # Average the complex numbers and take the angle
    mean_complex = (z1 + z2) / 2
    mean_angle = np.angle(mean_complex)

    return mean_angle


def normalize_angles(x):
    """Normalize angles to [-π, π] range."""
    return np.mod(x + np.pi, 2 * np.pi) - np.pi


def normalize_angles_torch(x):
    return torch.fmod(x + torch.pi, 2 * torch.pi) - torch.pi


def fit_piecewise_linear(df, xlabel, ylabel, n_pieces, figsize=(8, 8)):
    x = df[xlabel].values
    y = df[ylabel].values

    # Bounds for parameters
    breakpoint_bounds = [(-np.pi, np.pi)] * (n_pieces - 1)  # Bounds for breakpoints
    slope_bounds = [(-5, 5)] * n_pieces  # Bounds for slopes
    intercept_bounds = [(-np.pi, np.pi)]  # Bounds for first intercept only
    bounds = breakpoint_bounds + slope_bounds + intercept_bounds

    def compute_intercepts(breakpoints, slopes, first_intercept):
        """Compute dependent intercepts to ensure continuity."""
        n_pieces = len(slopes)
        intercepts = np.zeros(n_pieces)
        intercepts[0] = first_intercept

        # Compute each subsequent intercept based on the previous segment
        for i in range(1, n_pieces):
            # At breakpoint[i], both segments should give same value
            y_at_break = slopes[i - 1] * breakpoints[i] + intercepts[i - 1]
            # Solve for intercept[i]
            intercepts[i] = y_at_break - slopes[i] * breakpoints[i]

        return intercepts

    def create_objective_function(x, y, n_pieces):
        y = normalize_angles(y)

        def objective(params):
            # First n_pieces-1 parameters are breakpoints (-π and π are fixed)
            # Next n_pieces parameters are slopes
            # Last parameter is the first intercept
            breakpoints = sorted(
                np.concatenate([[-np.pi], params[: n_pieces - 1], [np.pi]])
            )
            slopes = params[n_pieces - 1 : 2 * n_pieces - 1]
            first_intercept = params[-1]

            # Compute dependent intercepts
            intercepts = compute_intercepts(breakpoints, slopes, first_intercept)

            y_pred = np.zeros_like(y)
            x_norm = normalize_angles(x)

            for i in range(n_pieces):
                mask = (x_norm >= breakpoints[i]) & (x_norm <= breakpoints[i + 1])
                y_pred[mask] = normalize_angles(
                    slopes[i] * x_norm[mask] + intercepts[i]
                )

            return np.sqrt(np.mean((normalize_angles(y - y_pred)) ** 2))

        return objective

    def create_prediction_function(optimization_result, n_pieces):
        def predict(x_new):
            x_norm = normalize_angles(x_new)
            breakpoints = sorted(
                np.concatenate(
                    [[-np.pi], optimization_result.x[: n_pieces - 1], [np.pi]]
                )
            )
            slopes = optimization_result.x[n_pieces - 1 : 2 * n_pieces - 1]
            first_intercept = optimization_result.x[-1]

            # Compute dependent intercepts
            intercepts = compute_intercepts(breakpoints, slopes, first_intercept)

            y_pred = np.zeros_like(x_norm)
            for i in range(n_pieces):
                mask = (x_norm >= breakpoints[i]) & (x_norm <= breakpoints[i + 1])
                y_pred[mask] = normalize_angles(
                    slopes[i] * x_norm[mask] + intercepts[i]
                )

            return y_pred, breakpoints

        return predict

    def plot_fit(x, y, fitted_func, n_pieces, xlabel, ylabel, rmse, mean_slope):
        x_norm = normalize_angles(x)
        y_norm = normalize_angles(y)
        plt.scatter(x_norm, y_norm, color="blue", alpha=0.5, label="Cells")

        x_smooth = np.linspace(-np.pi, np.pi, 1000)
        y_smooth, breakpoints = fitted_func(x_smooth)
        breakpoints = sorted(breakpoints)
        plt.plot(x_smooth[:-1], y_smooth[:-1], "r-", label="Fitted function")

        # Add vertical lines for breakpoints
        plt.vlines(
            breakpoints,
            plt.ylim()[0],
            plt.ylim()[1],
            colors="gray",
            linestyles="dashed",
            alpha=0.5,
        )

        # Annotate slopes at the midpoint of each segment
        slopes = result.x[n_pieces - 1 : 2 * n_pieces - 1]  # Extract slopes from result
        for i in range(n_pieces):
            midpoint_x = (breakpoints[i] + breakpoints[i + 1]) / 2
            plt.text(
                midpoint_x,
                plt.ylim()[1] - 0.3,  # Position text slightly below the top of the plot
                f"{slopes[i]:.2f}",
                color="black",
                fontsize=18,
                ha="center",
            )

        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(f"Mean slope: {mean_slope:.3f}, RMSE: {rmse:.3f}")
        plt.legend()
        plt.show()

        print("\nOptimized breakpoints:")
        for i, bp in enumerate(breakpoints):
            print(f"Breakpoint {i}: {bp:.3f} rad = {np.degrees(bp):.1f}°")

    objective = create_objective_function(x, y, n_pieces)

    result = differential_evolution(objective, bounds=bounds, maxiter=10000)

    fitted_func = create_prediction_function(result, n_pieces)
    breakpoints = sorted(np.concatenate([[-np.pi], result.x[: n_pieces - 1], [np.pi]]))
    mean_slope = np.average(
        result.x[n_pieces - 1 : 2 * n_pieces - 1], weights=np.diff(breakpoints)
    )

    plot_fit(x, y, fitted_func, n_pieces, xlabel, ylabel, result.fun, mean_slope)
    print(f"RMSE: {result.fun:.3f}")
    print(f"Mean slope: {mean_slope:.3f}")


def count_inversions(order, expected_order):
    """
    Count the number of inversions required to transform the given order
    into the expected order.
    """
    # Create a mapping of the expected order to indices
    expected_indices = {phase: i for i, phase in enumerate(expected_order)}

    # Convert the order to a list of indices based on the expected order
    order_indices = [expected_indices[phase] for phase in order]

    # Count inversions in order_indices
    inversions = 0
    n = len(order_indices)

    for i in range(n):
        for j in range(i + 1, n):
            if order_indices[i] > order_indices[j]:
                inversions += 1

    return inversions


def best_order(phases_dict, expected_order=["G1", "G1/S", "S", "G2", "G2/M", "M"]):
    """
    Determine which order (original or flipped) has fewer inversions, considering circular shifts.
    """
    rotated_order = sorted(phases_dict, key=phases_dict.get)
    # Count inversions for all circular shifts
    min_inversions = float("inf")
    best_rotation = None
    best_direction = 1

    for _ in range(len(expected_order)):
        # Count inversions for the original order
        original_inversions = count_inversions(rotated_order, expected_order)

        # Count inversions for the flipped order
        flipped_inversions = count_inversions(rotated_order[::-1], expected_order)

        # Choose the best one (with fewer inversions)
        if original_inversions < min_inversions:
            min_inversions = original_inversions
            best_rotation = rotated_order
            best_direction = 1

        if flipped_inversions < min_inversions:
            min_inversions = flipped_inversions
            best_rotation = rotated_order[::-1]
            best_direction = -1
        rotated_order = rotated_order[1:] + [rotated_order[0]]
        if min_inversions == 0:
            break

    return best_rotation, min_inversions, best_direction


def calculate_var(data):
    # Convert data to a numpy array if it's not already
    data = np.array(data)

    # Calculate the mean of all values
    mean_values = np.mean(data, axis=0)

    # Calculate the var for each f (along axis 0, for each value in the list)
    mse = np.mean(np.mean((data - mean_values) ** 2, axis=1))

    return mse


def create_synthetic_cells(adata: anndata.AnnData, n_bins=16, n_cells=50):
    adata = adata.copy()
    adata.obs["is_synthetic"] = False
    adata.obs["bin"] = pd.cut(
        adata.obs["inferred_theta"], n_bins, labels=list(range(n_bins))
    )
    for bin in range(n_bins // 2):
        i_min = adata.obs[adata.obs["bin"] == bin].sample(n_cells, replace=False).index
        i_max = (
            adata.obs[adata.obs["bin"] == bin + n_bins // 2]
            .sample(n_cells, replace=False)
            .index
        )
        synthetic_cells = adata[i_min].copy()
        for layer in adata.layers.keys():
            synthetic_cells.layers[layer] = (
                (synthetic_cells.layers[layer] + adata[i_max].layers[layer]) / 2
            ).ceil()
        synthetic_cells.obs_names = (
            synthetic_cells.obs_names + "_" + adata[i_max].obs_names
        )
        synthetic_cells.obs["is_synthetic"] = True
        adata = anndata.concat([adata, synthetic_cells])
    return adata


def compute_smoothed_variance_and_range(
    adata,
    phase_layer,
    bin_size=0.3,
    counts_sum_field="library_size",
    hue=None,
):
    def get_median_data(adata_obs, min_=-np.pi, max_=np.pi):
        adata_obs["phase_bin"] = pd.cut(
            adata_obs[phase_layer],
            bins=np.arange(min_, max_ + bin_size, bin_size),
        )
        median_data = (
            adata_obs.groupby("phase_bin", observed=False)[counts_sum_field]
            .median()
            .reset_index()
        )
        median_data["phase_center"] = median_data["phase_bin"].apply(lambda x: x.mid)
        return median_data

    # Handle adata or plain dataframe
    adata_obs_tot = adata if isinstance(adata, pd.DataFrame) else adata.obs

    if hue is None:
        raise ValueError("The 'hue' parameter must be provided for group comparison.")

    # Gather smoothed values for each group
    smoothed_group_dfs = []
    group_values = adata_obs_tot[hue].dropna().unique()

    for group_val in group_values:
        group_data = adata_obs_tot[adata_obs_tot[hue] == group_val].copy()
        median_df = get_median_data(group_data)
        median_df[hue] = group_val
        smoothed_group_dfs.append(median_df)

    # Combine all group data into one dataframe
    combined_df = pd.concat(smoothed_group_dfs)

    # Pivot: rows = phase_center, columns = hue, values = median
    pivot_df = combined_df.pivot_table(
        index="phase_center", columns=hue, values=counts_sum_field, observed=False
    ).fillna(0)

    per_bin_variance = pivot_df.var(axis=1)
    mean_variance_across_bins = per_bin_variance.mean()

    mean_per_group = pivot_df.mean(axis=1)
    range_across_groups = mean_per_group.max() / mean_per_group.min()

    return mean_variance_across_bins, range_across_groups
