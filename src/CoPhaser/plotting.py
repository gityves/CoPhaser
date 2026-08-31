from matplotlib.gridspec import GridSpecFromSubplotSpec
import matplotlib.transforms as transforms
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from typing import Literal

from CoPhaser import utils
import pkg_resources


import numpy as np
import matplotlib.pyplot as plt

import umap

TITLE_FONT_SIZE = 18


def modify_axis_labels(
    figsize=(8, 8),
    nrows=1,
    ncols=1,
    axis: Literal["x", "y", "both"] = "x",
    ax=None,
    step=0.25,
    offset=0,
):
    """
    Creates subplots and modifies axis labels for x, y, or both axes.

    Parameters:
        figsize (tuple): Size of the figure.
        nrows (int): Number of rows for subplots.
        ncols (int): Number of columns for subplots.
        axis (str): The axis to modify labels for. Options: "x", "y", "both".
        ax: An optional axis object. If provided, modifies this axis instead of creating new subplots.

    Returns:
        fig, axs: The figure and axes objects with modified labels.
    """
    if ax is None:
        fig, axs = plt.subplots(figsize=figsize, nrows=nrows, ncols=ncols)
        if nrows + ncols != 2:
            axs = axs.flatten()
        else:
            axs = [axs]
    else:
        axs = [ax]
        fig = ax.figure

    # Define ticks and labels for the x-axis and y-axis
    ticks = np.arange(-np.pi + offset, np.pi + step * np.pi + offset, step * np.pi)
    labels = [f"{round(t / np.pi, 2)}π" if t != 0 else "0" for t in ticks]

    for ax in axs:
        if axis in ["x", "both"]:
            ax.set_xticks(ticks)
            ax.set_xticklabels(labels)
        if axis in ["y", "both"]:
            ax.set_yticks(ticks)
            ax.set_yticklabels(labels)
    if nrows + ncols == 2:
        axs = axs[0]
    return fig, axs


def switch_legend_axes(ax, legend_prop_ax=0.2, split_orientation="h"):
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    divider = make_axes_locatable(ax)
    if split_orientation == "h":
        cax = divider.append_axes("right", size=f"{legend_prop_ax*100}%", pad=0.05)
    elif split_orientation == "v":
        cax = divider.append_axes("top", size=f"{legend_prop_ax*100}%", pad=0.05)

    handles, labels = ax.get_legend_handles_labels()
    legend_title = ax.get_legend().get_title().get_text()
    # get legend title
    if len(handles) == 0:
        # try to get legend from sns
        legend = ax.legend_
        handles = legend.legend_handles
        labels = [text.get_text() for text in legend.texts]
        legend_title = legend.get_title().get_text()
    cax.legend(handles, labels, loc="center", frameon=False, title=legend_title)
    cax.axis("off")
    if ax.get_legend() is not None:
        ax.get_legend().remove()


def plot_z_space_paper(
    z_1,
    z_2,
    ax,
    hue=None,
    alpha=0.1,
    size=5,
    title="z Space",
    xlabel="z1",
    ylabel="z2",
    cmap=None,
    legend=None,
    legend_prop_ax=None,
    split_orientation="h",
    legend_title=None,
):
    sns.scatterplot(
        x=z_1,
        y=z_2,
        hue=hue,
        alpha=alpha,
        edgecolor=None,
        s=size,
        ax=ax,
        palette=cmap,
        legend=legend,
    )
    if legend_title is not None:
        ax.legend(title=legend_title)

    if legend_prop_ax is not None and legend:
        switch_legend_axes(
            ax, legend_prop_ax=legend_prop_ax, split_orientation=split_orientation
        )

    ax.set_title(title)
    ax.set_yticks([])
    ax.set_xticks([])
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)


def plot_z_space(
    z_space: torch.Tensor,
    hue_cell_identity: np.ndarray = None,
    hue_time_component: np.ndarray = None,
    cell_identity_label="cell identity",
    alpha: float = 0.9,
    colors: np.ndarray = None,
):
    fig, axs = plt.subplots(2, 1, figsize=(8, 16))
    if z_space.shape[1] == 2:
        embedding = z_space.numpy()
        x_label = "Z 1"
        y_label = "Z 2"
    else:
        import umap

        reducer = umap.UMAP()
        embedding = reducer.fit_transform(z_space.numpy())
        x_label = "UMAP 1"
        y_label = "UMAP 2"
    if colors is not None:
        hue_to_color = {hue: color for hue, color in zip(hue_cell_identity, colors)}
    else:
        hue_to_color = None
    sns.scatterplot(
        x=embedding[:, 0],
        y=embedding[:, 1],
        hue=hue_cell_identity,
        palette=hue_to_color,
        ax=axs[0],
        alpha=alpha,
        s=10,
    ).set(xlabel=x_label, ylabel=y_label)
    axs[0].set_title(
        f"Z space colored by {cell_identity_label}",
        fontdict={"fontsize": TITLE_FONT_SIZE},
    )
    if hue_cell_identity is not None:
        sns.move_legend(axs[0], "upper left", bbox_to_anchor=(1, 1))

    sns.scatterplot(
        x=embedding[:, 0],
        y=embedding[:, 1],
        hue=hue_time_component,
        ax=axs[1],
        alpha=alpha,
        s=10,
    ).set(xlabel=x_label, ylabel=y_label)
    axs[1].set_title(
        "Z space colored by thetas",
        fontdict={"fontsize": TITLE_FONT_SIZE},
    )


def plot_cell_cycle(
    df_rhythmic,
    CCG_path="CCG_annotated.csv",
    ax=None,
    shift=0,
    direction=1,
    shrink=1.5,
):
    if CCG_path == "CCG_annotated.csv":
        CCG_path = pkg_resources.resource_filename(
            __name__, f"resources/CCG_annotated.csv"
        )
    df_ccg = pd.read_csv(CCG_path, index_col=0)
    df_phase_ptp = utils.get_ptp_phase(df_rhythmic)
    df_phase_ptp.index = df_phase_ptp.index.str.upper()
    df_phase_ptp = df_ccg.merge(
        df_phase_ptp, how="left", left_on="Primary name", right_index=True
    )
    df_phase_ptp = df_phase_ptp.dropna(subset=["peak_to_peak"])
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6), subplot_kw={"projection": "polar"})

    label_order = ["G1", "G1/S", "S", "G2", "G2/M", "M"]

    # Create the plot with the specified hue order
    df_phase_ptp["phase"] = utils.normalize_angles(
        utils.normalize_angles(df_phase_ptp["phase"]) * direction - shift
    )
    sns.histplot(
        data=df_phase_ptp[df_phase_ptp["peak_to_peak"] > 0.5],
        x="phase",
        hue="Peaktime",
        stat="density",
        common_norm=False,
        multiple="dodge",
        hue_order=label_order,
        shrink=shrink,
        ax=ax,
    ).set(ylabel="", xlabel="")

    # Modify axis annotation to show radians from -π to π
    labels_position = list(ax.get_xticks())
    labels = [
        str(((l + np.pi) % (2 * np.pi) - np.pi) / np.pi)[:5] + "π"
        for l in labels_position
    ]
    # remove 0.25pi label
    try:
        labels[labels_position.index(0.25 * np.pi)] = ""
    except:
        pass
    ax.set_xticklabels(labels)
    ax.set_yticklabels([])
    ax.set_title("CCG Acrophase Distribution")
    # divide by 2 the number of yticks
    yticks = ax.get_yticks()
    ax.set_yticks(yticks[::2])
    # move legend on the upper right partially outside of the plot
    sns.move_legend(ax, "upper left", bbox_to_anchor=(0.82, 1.1))


def plot_smoothed_size_evolution(
    adata,
    phase_layer,
    bin_size=0.3,
    counts_sum_field="library_size",
    hue=None,
    add_coherence_score=False,
    get_median_field=False,
    use_rad_axis=True,
    ax=None,
):
    def get_median_data(adata_obs, min_=-np.pi, max_=np.pi):
        adata_obs["phase_bin"] = pd.cut(
            adata_obs[phase_layer],
            bins=np.arange(
                min_,
                max_ + bin_size,
                bin_size,
            ),
        )

        # Group by bins and calculate median
        median_data = (
            adata_obs.groupby("phase_bin", observed=False)[counts_sum_field]
            .median()
            .reset_index()
        )

        # Convert bin intervals to their center points for plotting
        median_data["phase_center"] = median_data["phase_bin"].apply(lambda x: x.mid)
        return median_data

    if isinstance(adata, pd.DataFrame):
        adata_obs_tot = adata
    else:
        adata_obs_tot = adata.obs
    if hue is not None:
        frames = []
        for x in adata_obs_tot[hue].dropna().unique():
            mask = adata_obs_tot[hue] == x
            adata_obs = adata_obs_tot.loc[mask].copy()
            df_median = get_median_data(adata_obs)
            df_median[hue] = x
            df_median[counts_sum_field] = df_median[counts_sum_field].astype("float")
            df_median[counts_sum_field] = df_median[counts_sum_field].fillna(0)
            frames.append(df_median)
        median_data = pd.concat(frames)
        if add_coherence_score:
            all_functions = [
                median_data[median_data[hue] == h][counts_sum_field].values
                for h in adata_obs_tot[hue].dropna().unique()
            ]
            score = utils.calculate_var(all_functions)
            to_add = f", C={score}"
        else:
            to_add = ""

    else:
        median_data = get_median_data(adata_obs_tot)
        to_add = ""

    # Plot the median values
    if ax is None:
        pass
    elif use_rad_axis:
        fig, ax = modify_axis_labels()
    else:
        fig, ax = plt.subplot()
    sns.lineplot(data=median_data, x="phase_center", y=counts_sum_field, hue=hue, ax=ax)
    plt.xlabel(f"{phase_layer}")
    plt.ylabel(f"Median {counts_sum_field}")
    plt.title(f"Median {counts_sum_field} per {phase_layer}" + to_add)
    if get_median_field:
        return median_data


def plot_gene_profile(
    df_mean,
    adata,
    genes=["Top2a", "Pcna", "Mki67", "Mcm6"],
    layer_to_use="total",
    ncols=2,
    gene_to_upper=True,
    library_size=None,
    hue=None,
    theta_col="inferred_theta",
    alpha=0.2,
):
    nrows = np.ceil(len(genes) / ncols).astype(int)
    fig, axs = modify_axis_labels(
        figsize=(5 * ncols + 3, 5 * nrows), ncols=ncols, nrows=nrows, axis="x"
    )
    axs = axs.flatten()
    if gene_to_upper:
        genes = [gene.upper() for gene in genes]
    if library_size is None:
        library_size = adata.layers[layer_to_use].sum(axis=1).A1

    for i in range(len(genes)):
        sns.scatterplot(
            x=df_mean[theta_col],
            y=np.log(
                adata[:, genes[i]].layers[layer_to_use].toarray().flatten()
                / library_size
                * 10**4
                + 1
            ),
            alpha=alpha,
            label="Observed",
            ax=axs[i],
            color="red",
            edgecolor=None,
        ).set(ylabel="Normalized Counts", xlabel="Inferred Phase", title=genes[i])
        if hue is None:
            sns.scatterplot(
                x=df_mean[theta_col],
                y=np.log(df_mean[genes[i]] / library_size * 10**4 + 1),
                alpha=alpha,
                label="Inferred Means",
                ax=axs[i],
                edgecolor=None,
            ).set(ylabel="Normalized Counts", xlabel="Inferred Phase", title=genes[i])
        else:
            sns.scatterplot(
                x=df_mean[theta_col],
                y=np.log(df_mean[genes[i]] / library_size * 10**4 + 1),
                alpha=alpha,
                ax=axs[i],
                hue=hue,
                edgecolor=None,
            ).set(ylabel="Normalized Counts", xlabel="Inferred Phase", title=genes[i])
        if i == ncols - 1:
            axs[i].legend(loc="upper left", bbox_to_anchor=(1, 1))
        else:
            try:
                axs[i].get_legend().remove()
            except:
                pass
        axs[i].set_xlabel("Inferred θ")
        axs[i].set_xlim([-np.pi, np.pi])

    fig.suptitle(
        "Inferred mean and observed fraction of counts in function of inferred θ",
        fontsize=TITLE_FONT_SIZE,
    )
    plt.tight_layout()


def plot_fraction_counts(
    adata,
    thetas,
    genes=["Top2a", "Pcna", "Mki67", "Mcm6"],
    layer_to_use="total",
    ncols=2,
    gene_to_upper=True,
    library_size=None,
    hue=None,
    alpha=0.2,
):
    nrows = np.ceil(len(genes) / ncols).astype(int)
    fig, axs = modify_axis_labels(
        figsize=(5 * ncols + 3, 5 * nrows), ncols=ncols, nrows=nrows, axis="x"
    )
    axs = axs.flatten()
    if gene_to_upper:
        genes = [gene.upper() for gene in genes]
    if library_size is None:
        library_size = adata.layers[layer_to_use].sum(axis=1).A1

    for i in range(len(genes)):
        sns.scatterplot(
            x=thetas,
            y=np.log(
                adata[:, genes[i]].layers[layer_to_use].toarray().flatten()
                / library_size
                * 10**4
                + 1
            ),
            alpha=alpha,
            ax=axs[i],
            hue=hue,
            edgecolor=None,
        ).set(ylabel="Normalized Counts", xlabel="Inferred Phase", title=genes[i])
        if i == ncols - 1:
            axs[i].legend(loc="upper left", bbox_to_anchor=(1, 1))
        else:
            try:
                axs[i].get_legend().remove()
            except:
                pass
        axs[i].set_xlabel("Inferred θ")
        axs[i].set_xlim([-np.pi, np.pi])

    fig.suptitle(
        "Observed fraction of counts in function of inferred θ",
        fontsize=TITLE_FONT_SIZE,
    )
    plt.tight_layout()


def plot_smoothed_profiles(
    x,
    y,
    ax,
    hue=None,
    nbins=20,
    xlabel=None,
    ylabel=None,
    title=None,
    label=None,
    legend=True,
    cmap=None,
    add_end_start_points=False,
    hue_order=None,
    estimator="mean",
):
    x_raw = x.copy()
    x = [val.mid for val in pd.cut(x, bins=nbins)]
    if add_end_start_points:
        bin_width = 2 * np.pi / nbins
        # find all y where x_raw < -pi + bin_width/2 or x_raw > pi - bin_width/2
        extremities_y = y[
            (x_raw < -np.pi + bin_width / 2) | (x_raw > np.pi - bin_width / 2)
        ]
        x.extend([-np.pi, np.pi] * len(extremities_y))
        y = np.concatenate([y, extremities_y, extremities_y])
        if hue is not None:
            hue_extremities = hue[
                (x_raw < -np.pi + bin_width / 2) | (x_raw > np.pi - bin_width / 2)
            ]
            hue = np.concatenate([hue, hue_extremities, hue_extremities])

    if label is None:
        sns.lineplot(
            x=x,
            y=y,
            hue=hue,
            ax=ax,
            legend=legend,
            palette=cmap,
            hue_order=hue_order,
            estimator=estimator,
        )
    else:
        sns.lineplot(
            x=x,
            y=y,
            hue=hue,
            ax=ax,
            label=label,
            legend=legend,
            palette=cmap,
            hue_order=hue_order,
            estimator=estimator,
        )
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)


def plot_reconstruction_gene(
    df_mean,
    adata,
    gene,
    ax,
    layer_to_use="total",
    library_size=None,
    theta_col="inferred_theta",
    alpha=0.2,
    hue=None,
    legend=True,
    ylabel="log2 CP10k",
):
    if library_size is None:
        try:
            library_size = adata.layers[layer_to_use].sum(axis=1).A1
        except:
            library_size = adata.layers[layer_to_use].sum(axis=1)
    sns.scatterplot(
        x=df_mean[theta_col],
        y=np.log2(
            adata[:, gene].layers[layer_to_use].toarray().flatten()
            / library_size
            * 10**4
            + 1
        ),
        alpha=alpha,
        label="Observed",
        ax=ax,
        color="red",
        edgecolor=None,
        s=5,
    ).set(ylabel=ylabel, xlabel="Inferred Phase", title=gene)
    if hue is None:
        sns.scatterplot(
            x=df_mean[theta_col],
            y=np.log2(df_mean[gene] / library_size * 10**4 + 1),
            alpha=alpha,
            label="Inferred",
            ax=ax,
            edgecolor=None,
            s=5,
        ).set(ylabel=ylabel, xlabel="Inferred Phase", title=gene)
    else:
        sns.scatterplot(
            x=df_mean[theta_col],
            y=np.log2(df_mean[gene] / library_size * 10**4 + 1),
            alpha=alpha,
            ax=ax,
            hue=hue,
            edgecolor=None,
            s=5,
        ).set(ylabel=ylabel, xlabel="Inferred Phase", title=gene)
    if not legend:
        ax.get_legend().remove()
    ax.set_xlabel("Inferred Phase")


def plot_model_decoded_space(
    df,
    title: Literal["F space", "Z space"],
    gene_to_upper=True,
    genes=["Top2a", "Pcna", "Mki67", "Mcm6"],
    ncols=2,
    hue=None,
    alpha=1,
    size=10,
):
    """
    Plot the model's decoded space provided in df for different genes

    Parameters:
        df: df containg the decoded space for each genes and the inferred theta (ngenes+1, ncells)
    """

    nrows = np.ceil(len(genes) / ncols).astype(int)
    legend_space = 0
    if hue is not None:
        legend_space = 3
    fig, axs = modify_axis_labels(
        figsize=(6 * ncols + legend_space, 6 * nrows),
        ncols=ncols,
        nrows=nrows,
        axis="x",
    )
    axs = axs.flatten()
    if gene_to_upper:
        genes = [gene.upper() for gene in genes]

    for i in range(len(genes)):
        sns.scatterplot(
            data=df,
            x="inferred_theta",
            y=genes[i],
            ax=axs[i],
            hue=hue,
            alpha=alpha,
            s=size,
        ).set(title=genes[i], xlabel="Inferred θ")

        if i == ncols - 1:
            axs[i].legend(loc="upper left", bbox_to_anchor=(1, 1))
        elif hue is not None:
            axs[i].get_legend().remove()

    fig.suptitle(title, fontsize=TITLE_FONT_SIZE)


def plot_phase_distribution(
    weighted_mean_dict, amplitudes, phases, annotations, ax=None
):
    # Create a DataFrame for seaborn
    df = pd.DataFrame({"amplitude": amplitudes, "phase": phases, "Phase": annotations})

    if ax is None:
        plt.figure(figsize=(7, 7))
        ax = plt.subplot(111, projection="polar")
    label_order = ["G1", "G1/S", "S", "G2", "G2/M", "M"]
    color_palette = sns.color_palette("tab10", n_colors=len(label_order))
    sns.scatterplot(
        data=df,
        x="phase",
        y="amplitude",
        hue="Phase",
        style="Phase",
        ax=ax,
        s=100,
        hue_order=["G1", "G1/S", "S", "G2", "G2/M", "M"],
        palette=color_palette,
    )

    # Plot weighted mean directions with matching colors
    for cat, theta in weighted_mean_dict.items():
        color = color_palette[label_order.index(cat)]  # Match category color
        r_max = max(amplitudes) * 1.1
        ax.plot(
            [theta, theta],
            [0, r_max],
            color=color,
            linestyle="dashed",
            linewidth=2,
            label=f"{cat} Mean",
        )

        # Annotate the weighted mean at the end of the line
        ax.text(
            theta,
            r_max,
            f"{cat} Mean",
            fontsize=12,
            fontweight="bold",
            color=color,
            ha="center",
            va="bottom",
            bbox=dict(facecolor="white", edgecolor=color, boxstyle="round,pad=0.3"),
        )


def plot_posterior(posterior, adata, n=10, i=None, inferred_theta_col="inferred_theta"):
    x = np.linspace(-np.pi, np.pi, posterior.shape[1])  # X values
    # Create a compact DataFrame
    df = pd.DataFrame(posterior, index=adata.obs.index.copy(), columns=x)

    # Plot a **subset** of the observations to avoid clutter
    plt.figure(figsize=(10, 6))
    if i is None:
        i = df.sample(n, replace=False).index
    inferred_theta = adata.obs.loc[i, inferred_theta_col]
    palette = sns.color_palette(n_colors=len(i))
    ax = sns.lineplot(data=df.loc[i].T, palette=palette)

    # Add dots with matching colors
    for obs_idx, theta, color in zip(i, inferred_theta, palette):
        closest_x = df.columns[
            np.argmin(np.abs(df.columns - theta))
        ]  # Find closest x value
        y_value = df.loc[obs_idx, closest_x]  # Get corresponding y-value
        plt.scatter(
            closest_x, y_value, color=color, edgecolor="black", zorder=3
        )  # Matching color

    plt.show()


def plot_feature_importance(importance: torch.Tensor, rhythmic_gene_names):
    if isinstance(importance, torch.Tensor):
        df_feature_importance = pd.DataFrame(
            importance.detach().numpy(), columns=rhythmic_gene_names
        )
    else:
        df_feature_importance = pd.DataFrame(importance, columns=rhythmic_gene_names)

    fig, axs = plt.subplots(1, 2, figsize=(10, 6))
    df_feature_importance.loc[0].sort_values(
        ascending=False, key=lambda x: abs(x)
    ).head(10).plot.bar(ax=axs[0])
    axs[0].set_title("dimension 1")
    df_feature_importance.loc[1].sort_values(
        ascending=False, key=lambda x: abs(x)
    ).head(10).plot.bar(ax=axs[1])
    axs[1].set_title("dimension 2")
    plt.tight_layout()


def plot_fourrier_coefficients(ab_coefficients, gene_names):
    # Compute amplitude and phase
    a = ab_coefficients[:, 0]
    b = ab_coefficients[:, 1]
    amplitude = np.sqrt(a**2 + b**2)
    phase = np.arctan2(b, a)

    # Plotting in polar coordinates
    plt.figure(figsize=(8, 8))
    ax = plt.subplot(111, polar=True)

    # Scatter plot: phase is the angle, amplitude is the radius
    ax.scatter(phase, amplitude, s=50, c="blue", alpha=0.7)

    # Add gene names
    for i, name in enumerate(gene_names):
        ax.text(phase[i], amplitude[i] + 0.05, name, ha="center", va="center")

    ax.set_title("Phase and Amplitude of Rhythmic Genes", va="bottom")
    plt.show()


def plot_phase_accuracy(
    phase_gt,
    phase_pred,
    ax=None,
    cmap="rocket_r",
    offset=0,
    title="Phase Prediction Accuracy",
    title_gt="Ground Truth Phase",
    title_pred="Predicted Phase",
):
    if ax is None:
        plt.figure(figsize=(8, 8))
        ax = plt.gca()
    modify_axis_labels(ax=ax, axis="both", step=0.5)
    best_shift = utils.get_best_shift_direction(phase_pred, phase_gt)
    phase_pred = utils.normalize_angles(
        phase_pred * best_shift["direction"] - best_shift["shift"] + offset
    )

    sns.histplot(x=phase_gt, y=phase_pred, bins=50, ax=ax, cmap=cmap)
    ax.set_xlabel(title_gt)
    ax.set_ylabel(title_pred)
    ax.set_title(title)
    ax.set_xlim([-np.pi, np.pi])
    ax.set_ylim([-np.pi, np.pi])
    ax.axline((0, 0), slope=1, color="black", linestyle="--")


def plot_PMI(
    x,
    y,
    ax,
    n_bins=10,
    sigma=0.5,
    double_plot=False,
    min_value=np.log2(1.5),
    add_cbar=True,
    label_pads=-5,
):
    from matplotlib.colors import TwoSlopeNorm
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    from scipy.ndimage import gaussian_filter

    def modify_axis_labels(ax, x, y, n_bins):
        rad = ""
        if np.max(abs(x)) <= np.pi:
            ax.set_xticks([-np.pi, np.pi])
            ax.set_xticklabels(["-π", "π"])
            x_bin_edges = np.linspace(-np.pi, np.pi, n_bins, endpoint=True)
        else:
            ax.set_xticks([])
            x_bin_edges = np.linspace(np.min(x), np.max(x), n_bins)
        if np.max(abs(y)) <= np.pi:
            ax.set_yticks([-np.pi, np.pi])
            ax.set_yticklabels(["-π", "π"])
            y_bin_edges = np.linspace(-np.pi, np.pi, n_bins, endpoint=True)
        else:
            ax.set_yticks([])
            y_bin_edges = np.linspace(np.min(y), np.max(y), n_bins)
        return (x_bin_edges, y_bin_edges)

    def compute_PMI(x, y, bins, sigma):
        # Compute 2D histogram
        H, xedges, yedges = np.histogram2d(x, y, bins=bins)

        # Circular Gaussian smoothing
        H_smooth = gaussian_filter(H, sigma=(sigma, sigma), mode="wrap")

        # Normalize along x-axis (each column sums to 1)
        eps = 1e-9  # small constant to avoid log(0)

        Px = H_smooth.sum(axis=1, keepdims=True)
        Py = H_smooth.sum(axis=0, keepdims=True)

        Pxy = H_smooth / H_smooth.sum()
        Px = Px / H_smooth.sum()
        Py = Py / H_smooth.sum()

        PMI = np.log2((Pxy + eps) / (Px @ Py + eps))
        return PMI, xedges, yedges

    bins = modify_axis_labels(ax, x, y, n_bins)
    PMI, xedges, yedges = compute_PMI(x, y, bins, sigma)
    if double_plot:
        PMI = np.hstack((np.vstack((PMI, PMI)), np.vstack((PMI, PMI))))
        xedges = np.concatenate((xedges, xedges[:-1] + np.max(xedges)))
        yedges = np.concatenate((yedges, yedges[:-1] + 2 * np.pi))

    vmin = min(-min_value, np.min(PMI))
    vmax = max(min_value, np.max(PMI))

    norm = TwoSlopeNorm(
        vmin=vmin,
        vcenter=0.0,
        vmax=vmax,
    )

    im = ax.pcolormesh(
        xedges,
        yedges,
        PMI.T,
        cmap="coolwarm",
        norm=norm,
    )

    if add_cbar:
        divider = make_axes_locatable(ax)
        ax_cbar = divider.append_axes("right", size="5%", pad=0.1)
        cbar = plt.colorbar(
            im,
            cax=ax_cbar,
            orientation="vertical",
        )
        cbar.outline.set_visible(False)
        cbar.set_ticks([np.min(PMI), 0, np.max(PMI)])
        # set labels as round numbers
        cbar.set_ticklabels(
            [f"{np.round(np.min(PMI), 1)}", "0", f"{np.round(np.max(PMI), 1)}"]
        )
        # add label to cbar
        cbar.set_label("bits", labelpad=label_pads)


def label_panels_mosaic(fig, axes, mosaic, xloc=None, yloc=None, size=16, to_skip=[]):
    """
    Labels the panels in a mosaic plot.

    Parameters:
        - fig: The figure object.
        - axes: A dictionary of axes objects representing the panels.
        - xloc: The x-coordinate for the label position (default: 0).
        - yloc: The y-coordinate for the label position (default: 1.0).
        - size: The font size of the labels (default: 16).
    """
    if xloc is not None:
        assert len(mosaic) == len(xloc), "One xloc value per row required"
    else:
        xloc = [0] * len(mosaic)
    if yloc is not None:
        assert len(mosaic) == len(yloc), "One yloc value per row required"
    else:
        yloc = [1.0] * len(mosaic)

    for i, key in enumerate(mosaic):
        mosaic_row = mosaic[i]
        for key in mosaic_row:
            if key in to_skip:
                continue
            # label physical distance to the left and up:
            ax = axes[key]
            if isinstance(ax, list):
                ax = ax[0]
            trans = transforms.ScaledTranslation(-20 / 72, 7 / 72, fig.dpi_scale_trans)
            ax.text(
                xloc[i],
                yloc[i],
                key,
                transform=ax.transAxes + trans,
                fontsize=size,
                va="bottom",
            )


def plot_cell_cycle_validations(
    model,
    space_outputs,
    generative_outputs,
    adata,
    layer,
    genes=["Top2a", "Pcna", "Mki67", "Mcm6"],
    gene_to_upper=True,
    hue_key=None,
    max_n_points=10_000,
    return_values=False,
    display_labels=True,
):
    """
    Plot the cell cycle validation plots for the given model and data.

    Parameters:
        model: The trained model object.
        space_outputs: The outputs from the model's latent space.
        generative_outputs: The outputs from the model's generative process.
        adata: The AnnData object containing the data. The gene names should be the symbols of the genes for plot "E".
        layer: The layer of the AnnData object containing the counts.
        genes: List of genes to plot. Should be 4 genes, in the set of the context or rhythmic genes.
        gene_to_upper: If True, convert gene names to uppercase.
        hue_key: Key in adata.obs to use for coloring the plots.
        max_n_points: Maximum number of points to plot in the latent space visualization.
        return_values: If True, return the aligned phases and the posterior entropy distribution.
        display_labels: If True, display labels on the plots.

    Returns:
        If return_values is True, returns a tuple (fig, axs, fig_space, axs_space, thetas, entropy, context, cells_projected)
    """

    ##################################
    ### Inferred phases validation ###
    ##################################

    #### setup ####
    if hue_key is not None:
        hue = adata.obs[hue_key].values
    else:
        hue = None
    if gene_to_upper:
        genes = [gene.upper() for gene in genes]

    def replace_mosaic_cell_with_grid(fig, axs, label, nrows, ncols):
        # Extract subplotspec from the original axes
        original_ax = axs[label]
        sspec = original_ax.get_subplotspec()
        fig = original_ax.figure
        original_ax.remove()

        # Create new inner grid
        inner = GridSpecFromSubplotSpec(
            nrows, ncols, subplot_spec=sspec, hspace=0.3, wspace=0.1
        )

        # Create new axes and store them in a list
        subaxes = []
        for r in range(nrows):
            for c in range(ncols):
                ax = fig.add_subplot(inner[r, c])
                ax.spines["top"].set_visible(False)
                ax.spines["right"].set_visible(False)
                subaxes.append(ax)

        # Replace entry in axs dict
        axs[label] = subaxes

        return axs

    mosaic = [
        ["A", "B"],
        ["C", "D"],
        ["E", "F"],
    ]
    fig, axs = plt.subplot_mosaic(
        mosaic,
        figsize=(16, 24),
        per_subplot_kw={"E": {"projection": "polar"}},
        height_ratios=[
            1.5,
            1,
            1,
        ],
    )
    for l in ["A", "B"]:
        replace_mosaic_cell_with_grid(fig, axs, l, nrows=2, ncols=2)
    for l in ["C", "D", "F"]:
        axs[l].spines["top"].set_visible(False)
        axs[l].spines["right"].set_visible(False)

    #### Get Phases and plot peaking phase of annotated genes ####
    ax = axs["E"]
    modify_axis_labels(ax=ax, axis="x", step=0.5, offset=np.pi)
    thetas = model.orient_align_pseudotimes(space_outputs["theta"], ax=axs["E"])

    #### Plot the selected genes profiles ####
    for i, gene in enumerate(genes):
        ax = axs["A"][i]
        modify_axis_labels(ax=ax, axis="x", step=0.5)

        plot_smoothed_profiles(
            thetas.detach().numpy(),
            utils.get_genes_fractions(
                gene, adata, layer=layer, normalized=True
            ).flatten(),
            ax=ax,
            xlabel="Inferred Phase",
            ylabel=f"Normalized counts (log)",
            hue=hue,
            legend=display_labels and not bool(i),
        )
        ax.set_title(gene)

    #### Plot the modeled fourier contribution of the selected genes ####
    df_rhythmic = pd.DataFrame(
        generative_outputs["F"].detach().numpy(), columns=model.context_genes
    )
    df_rhythmic["inferred_theta"] = thetas.detach().numpy()
    for i, gene in enumerate(genes):
        ax = axs["B"][i]
        modify_axis_labels(ax=ax, axis="x", step=0.5)
        plot_smoothed_profiles(
            thetas.detach().numpy(),
            df_rhythmic[gene].values.flatten(),
            ax=ax,
            xlabel="Inferred Phase",
            ylabel=f"Modeled F Coeffs (log)",
            hue=hue,
            legend=display_labels and not bool(i),
        )
        ax.set_title(gene)
    #### Plot the histones profiles ####
    utils.add_histones_fraction(adata=adata, layer=layer, use_only_clustered=True)
    ax = axs["C"]
    modify_axis_labels(ax=ax, axis="x", step=0.5)
    plot_smoothed_profiles(
        thetas.detach().numpy(),
        adata.obs["histones_fraction"],
        ax=ax,
        hue=hue,
        legend=display_labels and not bool(i),
    )
    ax.set_ylabel("Histones Fraction (log normalized)")
    ax.set_xlabel("Inferred Phase")

    #### Plot the library size profiles ####
    try:
        library_size = adata.layers[layer].sum(axis=1).A1
    except:
        library_size = adata.layers[layer].sum(axis=1)
    ax = axs["D"]
    modify_axis_labels(ax=ax, axis="x", step=0.5)
    plot_smoothed_profiles(
        thetas.detach().numpy(),
        library_size,
        ax=axs["D"],
        hue=hue,
        legend=display_labels and not bool(i),
    )
    ax.set_ylabel("Library Size")
    ax.set_xlabel("Inferred Phase")

    #### Plot the phase posterior entropy distribution ####
    print("Getting posterior and calculating entropy...")
    posterior, theta_grid = model.get_posterior(adata, layer_to_use=layer, n_points=20)
    entropy = utils.normalized_shanon_entropy(posterior)
    is_cycling = space_outputs["b_z"] > 0.5
    ax = axs["F"]
    if not all(is_cycling):
        modify_axis_labels(ax=ax, axis="x", step=0.5)
        high_posterior_width = pd.Series(entropy > 0.99, name="High Posterior Width")
        sns.histplot(x=thetas, hue=high_posterior_width, ax=ax)
        ax.set_yscale("log")
        f_legend = (
            "Distribution of the inferred phases separating cells with high posterior width from others. \n"
            "(Having cells with high posterior width should be in the G1 area)"
        )

    else:
        sns.histplot(x=entropy, ax=ax)
        # add vertical line at 0.8
        ax.axvline(0.8, color="red", linestyle="--")
        ax.set_title("Posterior Entropy Distribution")
        ax.set_xlabel("Normalized Shannon Entropy")
        ax.set_xlim([0, 1])
        f_legend = (
            "Distribution of the posterior entropy of the inferred phases. \n"
            "(Having a large fraction of cells above 0.8 suggests that a large fraction of cells are in G0, using the Bernoulli VAE should help (see G0 tutorial).)"
        )

    fig.tight_layout()
    label_panels_mosaic(
        fig,
        axs,
        mosaic,
    )
    try:
        display(fig)
    except:
        fig.show()
    # print the legend of the plots
    print(f"""Validation of the inferred cell cycle phases.

A. Reconstructed gene expression profiles, obtained by binning cells according to inferred phase and averaging 
log(1 + counts per 10,000) for the indicated gene.
(You should see: Top2a 2-3 log, peak G2M; Mki67 2-3 log, peak G2M; Pcna 1-2 log, peak S; Mcm6 1-2 log, peak S.
If Top2a/Mki67 are not in phase and phase shifted with respect to Pcna/Mcm6, the inferred phases are wrong.)

B. Fourier derived predicted gene expression profiles in log
(You should see: Top2a 2-3 log, peak G2M; Mki67 2-3 log, peak G2M; Pcna 1-2 log, peak S; Mcm6 1-2 log, peak S.)

C. Histone fraction in function of inferred phase.
(You should see a peak during the S phase (see D and E to determine the S phase).)

D. Library size in function of inferred phase.
(You should ~see a doubling of the library size.)

E. Scatter plot of the inferred peaking phases and amplitudes of annotated cell cycle genes.
(You should ~see a progression from G1-G1/S-S-G2-M.)

F. {f_legend}
""")
    plt.close(fig)

    ##################################
    ### Latent space visualization ###
    ##################################

    ### setup ###
    mosaic = [
        ["A", "B", "C"],
    ]
    fig_space, axs_space = plt.subplot_mosaic(
        mosaic,
        figsize=(16, 6),
    )
    for l in mosaic[0]:
        axs_space[l].spines["top"].set_visible(False)
        axs_space[l].spines["right"].set_visible(False)
    context = space_outputs["z"].detach().numpy()

    phases = pd.Series(thetas.detach().numpy())
    cells_projected = space_outputs["x_projected"].detach().numpy()
    if context.shape[0] > max_n_points:
        idx = np.random.choice(context.shape[0], max_n_points, replace=False)
        context = context[idx]
        if hue is not None:
            hue = hue[idx]
        phases = phases[idx]
        cells_projected = cells_projected[idx]

    phases.name = "Inferred Phase"

    if context.shape[1] > 2:
        print("Reducing context space to 2D using UMAP...")
        reducer = umap.UMAP()
        context = reducer.fit_transform(context)
        labels = ["UMAP z 1", "UMAP z 2"]
    else:
        labels = ["z1", "z2"]

    ### Latent f space (defines the inferred phase) ###
    def transform_projected_space(points, phases):
        x = points[:, 0]
        y = points[:, 1]
        r = np.sqrt(x**2 + y**2)

        x_new = r * np.cos(phases)
        y_new = r * np.sin(phases)

        return np.column_stack([x_new, y_new])

    cells_projected_rotated = transform_projected_space(cells_projected, phases)
    ax = axs_space["A"]
    sns.histplot(
        x=cells_projected_rotated[:, 0], y=cells_projected_rotated[:, 1], ax=ax
    )
    ax.set_xlabel("f1")
    ax.set_ylabel("f2")
    ax.set_title("Projected f Space")

    ### Latent z space (context space) colored by hue ###
    ax = axs_space["B"]
    sns.scatterplot(
        x=context[:, 0], y=context[:, 1], hue=hue, ax=ax, s=5, legend=display_labels
    )
    ax.set_xlabel(labels[0])
    ax.set_ylabel(labels[1])
    ax.set_title("Context Space")

    ### Latent z space (context space) colored by phases ###
    ax = axs_space["C"]
    sns.scatterplot(
        x=context[:, 0], y=context[:, 1], hue=phases, ax=ax, palette="hsv", s=5
    )
    ax.set_xlabel(labels[0])
    ax.set_ylabel(labels[1])
    ax.set_title("Context Space")

    fig_space.tight_layout()
    label_panels_mosaic(
        fig_space,
        axs_space,
        mosaic,
    )
    try:
        display(fig_space)
    except:
        fig_space.show()
    plt.close(fig_space)
    print("""Latent space visualization of the inferred cell cycle phases.
    
A. Histogram of the projected f space, which defines the inferred phase. 
(Seeing an annulus is a good signal, but this depends on the weight of Trainer's closed_circle_weight)

B. Context space colored by the provided hue (if any).

C. Context space colored by the inferred phase. 
(You should see a ~ random distribution of the inferred phases in the context space, since the mutual
information between the context and the inferred phase is minimized.)
""")
    if return_values:
        return fig, axs, fig_space, axs_space, phases, entropy, context, cells_projected
