"""GUI result panels, built from `cophaser.plotting` and `cophaser.cyclic_r2`."""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch

from cophaser import plotting, utils, cyclic_r2


def gene_profiles_panel(adata, layer, thetas, genes, hue=None, ncols=2):
    """Expression profile vs. inferred phase for each gene."""
    nrows = int(np.ceil(len(genes) / ncols))
    fig, axs = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
    axs = np.atleast_1d(axs).flatten()
    for i, gene in enumerate(genes):
        y = utils.get_genes_fractions(gene, adata, layer=layer, normalized=True).flatten()
        plotting.plot_smoothed_profiles(
            thetas,
            y,
            ax=axs[i],
            hue=hue,
            xlabel="Inferred phase",
            ylabel="Normalized counts (log)",
            title=gene,
            legend=(i == 0),
        )
    for j in range(len(genes), len(axs)):
        axs[j].axis("off")
    fig.tight_layout()
    return fig


def umap_panel(z_space, hue_values=None, thetas=None, max_n_points=30_000):
    """UMAP of the context space, colored by an obs hue and by inferred phase."""
    n = z_space.shape[0]
    if n > max_n_points:
        idx = np.random.choice(n, max_n_points, replace=False)
        z_space = z_space[idx]
        if hue_values is not None:
            hue_values = np.asarray(hue_values)[idx]
        if thetas is not None:
            thetas = np.asarray(thetas)[idx]

    plotting.plot_z_space(
        torch.as_tensor(np.asarray(z_space), dtype=torch.float32),
        hue_cell_identity=hue_values,
        hue_time_component=thetas,
        cell_identity_label="selected hue",
    )
    return plt.gcf()


def context_and_cycle_space_row(z_space, cells_projected, thetas, hue_values=None,
                                max_n_points=30_000):
    """Context-space UMAP by hue and by phase, plus the projected cycle space."""
    n = z_space.shape[0]
    if n > max_n_points:
        idx = np.random.choice(n, max_n_points, replace=False)
        z_space, thetas = z_space[idx], np.asarray(thetas)[idx]
        cells_projected = np.asarray(cells_projected)[idx]
        if hue_values is not None:
            hue_values = np.asarray(hue_values)[idx]

    import umap

    embedding = umap.UMAP(n_components=2, random_state=0, n_jobs=1).fit_transform(
        np.asarray(z_space)
    )
    fig, axs = plt.subplots(1, 3, figsize=(18, 5.5))

    if hue_values is not None:
        sns.scatterplot(
            x=embedding[:, 0], y=embedding[:, 1], hue=hue_values, s=6, linewidth=0,
            ax=axs[0], legend="brief",
        )
        axs[0].legend(fontsize=7, markerscale=2, loc="best")
    else:
        axs[0].scatter(embedding[:, 0], embedding[:, 1], s=6, linewidth=0, color="0.5")
    axs[0].set_title("Context space (UMAP), by hue")

    sc = axs[1].scatter(
        embedding[:, 0], embedding[:, 1], c=np.asarray(thetas), s=6, linewidth=0,
        cmap="twilight", vmin=-np.pi, vmax=np.pi,
    )
    axs[1].set_title("Context space (UMAP), by inferred phase")
    fig.colorbar(sc, ax=axs[1], label="Inferred phase")

    rotated = plotting.project_and_rotate_f_space(
        np.asarray(cells_projected), np.asarray(thetas)
    )
    sns.histplot(x=rotated[:, 0], y=rotated[:, 1], ax=axs[2])
    axs[2].set_xlabel("f1")
    axs[2].set_ylabel("f2")
    axs[2].set_title("Projected cycle space")

    for ax in axs[:2]:
        ax.set_xticks([])
        ax.set_yticks([])
    fig.tight_layout()
    return fig


def projected_cycle_space_panel(cells_projected, thetas):
    """Histogram of the projected f-space, rotated so its angle is the inferred phase."""
    rotated = plotting.project_and_rotate_f_space(cells_projected, np.asarray(thetas))
    fig, ax = plt.subplots(figsize=(6, 6))
    sns.histplot(x=rotated[:, 0], y=rotated[:, 1], ax=ax)
    ax.set_xlabel("f1")
    ax.set_ylabel("f2")
    ax.set_title("Projected cycle space")
    return fig


def _phase_bin_edges(n_bins):
    return np.linspace(-np.pi, np.pi, n_bins + 1)


def phase_agreement_panel(x, y, xlabel, ylabel, n_bins=60, figsize=(4.2, 4.2)):
    """2D histogram of two runs' phases with marginals. Agreeing runs show one straight
    band of slope +-1 at any offset (phase is defined up to rotation/reflection)."""
    fig = plt.figure(figsize=figsize)
    # explicit margins: tight_layout clips the y label once the marginals are off
    gs = fig.add_gridspec(
        2, 2, width_ratios=(4, 1), height_ratios=(1, 4), wspace=0.05, hspace=0.05,
        left=0.19, right=0.97, top=0.97, bottom=0.15,
    )
    ax = fig.add_subplot(gs[1, 0])
    ax_top = fig.add_subplot(gs[0, 0], sharex=ax)
    ax_right = fig.add_subplot(gs[1, 1], sharey=ax)

    edges = _phase_bin_edges(n_bins)
    ax.hist2d(np.asarray(x), np.asarray(y), bins=(edges, edges))
    ax_top.hist(np.asarray(x), bins=edges, color="0.4")
    ax_right.hist(np.asarray(y), bins=edges, orientation="horizontal", color="0.4")

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xlim(-np.pi, np.pi)
    ax.set_ylim(-np.pi, np.pi)
    plotting.modify_axis_labels(ax=ax, axis="both", step=0.5)
    ax_top.axis("off")
    ax_right.axis("off")
    return fig


def phase_distribution_panel(thetas, xlabel, n_bins=60, figsize=(4.6, 3.2)):
    """Phase histogram of a single run."""
    fig, ax = plt.subplots(figsize=figsize)
    ax.hist(np.asarray(thetas), bins=_phase_bin_edges(n_bins), color="0.4")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Cells")
    ax.set_xlim(-np.pi, np.pi)
    plotting.modify_axis_labels(ax=ax, axis="x", step=0.5)
    fig.tight_layout()
    return fig


def r2_polar_panel(
    X,
    thetas,
    z,
    gene_names,
    rhythmic_genes,
    min_r2: float = 0.1,
    max_cells: int = 10_000,
):
    """Per-gene cyclic R^2 vs. peaking phase, highlighting non-rhythmic-set
    candidates. Expensive (Leiden + per-gene IRLS) - call only on demand."""
    r2_df = cyclic_r2.fit_cyclic_r2_celltypes(
        X, np.asarray(thetas), np.asarray(z), gene_names=list(gene_names), max_cells=max_cells
    )
    r2_df = r2_df.sort_values("r2", ascending=False).reset_index(drop=True)
    fig, ax = plotting.plot_r2_polar_scatter(r2_df, selected_genes=set(rhythmic_genes), min_r2=min_r2)
    return fig, r2_df
