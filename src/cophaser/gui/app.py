"""CoPhaser GUI: a Streamlit wizard from a .h5ad to inferred phases. Run with `cophaser-gui`.

Training never runs in-process: the Configure step generates a standalone script
(script_gen.build_run_script), which is both shown to the user and run as a subprocess.
"""

from __future__ import annotations

import atexit
import contextlib
import inspect
import io
import itertools
import os
import queue
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from datetime import datetime

import numpy as np
import pandas as pd
import scanpy as sc
import streamlit as st
import torch
import matplotlib.pyplot as plt

from cophaser import (
    CoPhaser,
    Trainer,
    auto_hyperparameters,
    auto_params,
    cycle_configs,
    gene_sets,
    plotting,
    species as species_mod,
    suggest_n_latent,
    utils,
)
from cophaser.gui import plots as gui_plots, script_gen

# Parse Trainer's per-epoch print, e.g. "Epoch 1/3, elbo_loss: 209.15, ...".
EPOCH_RE = re.compile(r"^Epoch (\d+)/(\d+)")
LOSS_RE = re.compile(r"(?:elbo_loss|elbo loss):\s*([\-\d.eE+]+)")
RUN_RE = re.compile(r"^=== RUN (\d+)/(\d+)")

CYCLES_BY_SPECIES = {
    "mouse": ["cell_cycle", "circadian", "somite"],
    "human": ["cell_cycle", "circadian", "menstrual"],
}

CYCLE_LABELS = {
    "cell_cycle": "Cell cycle",
    "circadian": "Circadian",
    "somite": "Somite (segmentation) clock",
    "menstrual": "Menstrual cycle",
}

N_VARIABLE_GENES = 2000
MAX_PLOT_WIDTH = 700
MIN_CELLS_AFTER_QC = 100

SUBSAMPLE_CELLS = inspect.signature(Trainer.train_model).parameters["subsample"].default

# Applied as CSS rather than theme.baseFontSize: setting any [theme] option pins Streamlit
# to light mode and removes the dark-mode switch.
BASE_FONT_SIZE_PX = 19

# Figures are notebook-sized but shown capped to MAX_PLOT_WIDTH px, so bump font sizes.
# Set here (GUI process only) so notebook users keep matplotlib's defaults.
plt.rcParams.update(
    {
        "font.size": 15,
        "axes.titlesize": 17,
        "axes.labelsize": 15,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        "legend.fontsize": 14,
        "figure.titlesize": 19,
    }
)


def _show_fig(fig, max_width=MAX_PLOT_WIDTH):
    st.pyplot(fig, width=max_width)


def _format_duration(seconds):
    seconds = max(0, int(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h {m}m"
    if m:
        return f"{m}m {s}s"
    return f"{s}s"


def init_state():
    defaults = dict(
        step=1,
        adata=None,
        h5ad_path=None,
        h5ad_display_name=None,
        layer_from_x=False,
        gene_naming="symbols",
        gene_symbol_column=None,
        path_input="",
        layer=None,
        species=None,
        cycle=None,
        min_counts=None,
        max_counts=None,
        min_pct=20.0,
        max_pct=95.0,
        rhythmic_genes=None,
        menstrual_mi_genes=[],
        extra_rhythmic_genes=[],
        n_latent=10,
        hyperparams_mode="autotune",
        preload_fourier_prior=False,
        model_kwargs={},
        trainer_kwargs={},
        train_kwargs={},
        decoder_amp_phase_prior=None,
        decoder_amp_phase_resource=None,
        n_variable_genes=N_VARIABLE_GENES,
        n_cells_after_qc=0,
        seed=0,
        n_runs=3,
        seeds=[0, 1, 2],
        run_names=["seed_0", "seed_1", "seed_2"],
        output_dir=None,
        run_script=None,
        proc=None,
        log_lines=[],
        losses=[],
        run_finished=False,
        run_done_marker=False,
        run_cancelled=False,
        log_queue=None,
        run_start_time=None,
        current_run_idx=1,
        n_runs_total=1,
    )
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def goto(step):
    """Button on_click target. Outside a callback, follow with st.rerun()."""
    st.session_state.step = step


# ---------------------------------------------------------------------------
# Step 1: load data
# ---------------------------------------------------------------------------


def _browse_dir(path_text: str, limit: int = 100):
    """Directory to browse and its first `limit` entries (subfolders first, then .h5ad files)."""
    if path_text and os.path.isdir(path_text):
        base_dir = path_text
    else:
        base_dir = os.path.dirname(path_text) if path_text else os.getcwd()
    base_dir = base_dir or os.getcwd()
    # Same folder -> same string (with or without trailing "/"), else the radio widget,
    # keyed on it, is recreated and loses its selection.
    base_dir = os.path.normpath(base_dir)
    if not os.path.isdir(base_dir):
        return base_dir, []
    try:
        entries = sorted(
            os.scandir(base_dir), key=lambda e: (not e.is_dir(), e.name.lower())
        )
    except OSError:
        return base_dir, []
    out = []
    for e in entries:
        if e.is_dir():
            out.append(e.name + os.sep)
        elif e.name.lower().endswith(".h5ad"):
            out.append(e.name)
        if len(out) >= limit:
            break
    return base_dir, out


ENSEMBL_ID_RE = re.compile(r"^ENS[A-Z]*\d{6,}(\.\d+)?$", re.IGNORECASE)
# Checked in this order, so an exact "Symbol" wins over a vaguer "Gene".
SYMBOL_VAR_COLUMNS = (
    "Symbol",
    "symbol",
    "SYMBOL",
    "gene_symbol",
    "gene_symbols",
    "GeneSymbol",
    "gene_name",
    "gene_names",
    "feature_name",
    "Gene",
    "gene",
)


def _describe_gene_naming(names, sample=500):
    """"symbols", "ensembl", "index" or "unknown". Packaged gene lists match by symbol only."""
    values = [str(v) for v in list(names)[:sample]]
    if not values:
        return "unknown"
    n = len(values)
    if sum(bool(ENSEMBL_ID_RE.match(v)) for v in values) > n / 2:
        return "ensembl"
    if sum(v.isdigit() for v in values) > n / 2:
        return "index"
    if sum(any(ch.isalpha() for ch in v) for v in values) > n / 2:
        return "symbols"
    return "unknown"


def _symbol_column_candidates(adata):
    """`.var` columns whose contents look like gene symbols, best-named first."""
    found = [
        col
        for col in adata.var.columns
        if _describe_gene_naming(adata.var[col].astype(str)) == "symbols"
    ]

    def rank(col):
        return (
            SYMBOL_VAR_COLUMNS.index(col)
            if col in SYMBOL_VAR_COLUMNS
            else len(SYMBOL_VAR_COLUMNS)
        )

    return sorted(found, key=rank)


def _use_symbol_column(adata, col):
    adata.var_names = adata.var[col].astype(str).values
    adata.var_names_make_unique()


@st.cache_resource
def _upload_dirs():
    """Upload temp folders of all sessions; whatever is left is removed when the server exits."""
    dirs = set()
    atexit.register(lambda: [shutil.rmtree(d, ignore_errors=True) for d in list(dirs)])
    return dirs


def _discard_upload_dir():
    """Delete this session's previous upload copy (kept until exit if a run may still read it)."""
    old = st.session_state.get("_upload_tmp_dir")
    proc = st.session_state.get("proc")
    if old and (proc is None or proc.poll() is not None):
        shutil.rmtree(old, ignore_errors=True)
        _upload_dirs().discard(old)
    st.session_state._upload_tmp_dir = None


def _load_h5ad(path, display_name=None):
    upload_dir = st.session_state.get("_upload_tmp_dir")
    if upload_dir and os.path.dirname(path) != upload_dir:
        _discard_upload_dir()
    with st.status(f"Loading {display_name or path} ...", expanded=True) as status:
        st.write("Reading the .h5ad file...")
        adata = sc.read_h5ad(path)
        status.update(
            label=f"Loaded {adata.n_obs} cells x {adata.n_vars} genes.",
            state="complete",
        )
    # No layers: the GUI works on a "total" copy of .X; the script gets layer=None.
    st.session_state.layer_from_x = len(adata.layers) == 0
    if st.session_state.layer_from_x:
        adata.layers["total"] = adata.X.copy()
    st.session_state.adata = adata
    st.session_state.h5ad_path = path
    st.session_state.h5ad_display_name = display_name or path
    st.session_state._last_loaded_path = path
    st.session_state.menstrual_mi_genes = []
    st.session_state.extra_rhythmic_genes = []
    # Decided here, on the names as read from disk (before any renaming below).
    st.session_state.gene_naming = _describe_gene_naming(adata.var_names)
    st.session_state.gene_symbol_column = None
    if st.session_state.gene_naming != "symbols":
        candidates = _symbol_column_candidates(adata)
        if candidates:
            _use_symbol_column(adata, candidates[0])
            st.session_state.gene_symbol_column = candidates[0]


def _stage_load(path):
    """Only stages the path: a callback inside a fragment must not render anything.
    The load itself happens in step_load()."""
    st.session_state._path_to_load = path


def _select_nav_entry(base_dir, radio_key):
    """Stage the picked entry's path; the text field picks it up on the next rerun."""
    selected = st.session_state.get(radio_key)
    if selected:
        st.session_state._pending_path = os.path.join(base_dir, selected)


@st.fragment
def _path_browser_fragment():
    """Editable path field plus a listing of the typed folder."""
    if "_pending_path" in st.session_state:
        st.session_state.path_input = st.session_state.pop("_pending_path")

    path = st.text_input(
        "Path to a .h5ad file, or a folder to browse", key="path_input"
    )

    base_dir, entries = _browse_dir(path)
    if entries:
        st.caption(f"{base_dir} ({len(entries)} shown)")
        radio_key = f"nav_radio_{base_dir}"
        with st.container(height=250, border=True):
            st.radio(
                "Entries",
                entries,
                index=None,
                key=radio_key,
                label_visibility="collapsed",
                on_change=_select_nav_entry,
                args=(base_dir, radio_key),
            )

    is_valid = bool(path) and path.endswith(".h5ad") and os.path.isfile(path)
    if is_valid and st.session_state.get("_last_loaded_path") != path:
        # Enter acts like Load. Full-app rerun so the rest of the wizard sees the dataset.
        st.session_state._path_to_load = path
        st.rerun()
    st.button("Load", disabled=not is_valid, on_click=_stage_load, args=(path,))


def step_load():
    st.header("1. Load your dataset")

    # Staged by the path browser; loaded here since fragments can't render the progress.
    pending_path = st.session_state.pop("_path_to_load", None)
    if pending_path and st.session_state.get("_last_loaded_path") != pending_path:
        _load_h5ad(pending_path)

    mode = st.radio(
        "How do you want to provide the .h5ad file?",
        ["Path on disk", "Upload"],
        horizontal=True,
    )

    if mode == "Upload":
        uploaded = st.file_uploader("Drag and drop an .h5ad file", type=["h5ad"])
        upload_id = getattr(uploaded, "file_id", None) if uploaded is not None else None
        if uploaded is not None and st.session_state.get("_upload_id") != upload_id:
            # Own folder per upload: the script reads this path, and same-named uploads
            # must not share cache keys.
            _discard_upload_dir()
            tmp_dir = tempfile.mkdtemp(prefix="cophaser_upload_")
            _upload_dirs().add(tmp_dir)
            st.session_state._upload_tmp_dir = tmp_dir
            tmp_path = os.path.join(tmp_dir, uploaded.name)
            with open(tmp_path, "wb") as f:
                f.write(uploaded.getbuffer())
            _load_h5ad(tmp_path, display_name=uploaded.name)
            st.session_state._upload_id = upload_id
    else:
        _path_browser_fragment()

    if st.session_state.adata is not None:
        adata = st.session_state.adata
        st.success(
            f"Loaded **{st.session_state.h5ad_display_name}**: {adata.n_obs} cells x {adata.n_vars} genes."
        )

        if st.session_state.layer_from_x:
            st.info(
                'No layers found in this AnnData - copied `.X` into a new layer called "total".'
            )

        naming = st.session_state.get("gene_naming", "symbols")
        if naming != "symbols":
            looks_like = {
                "ensembl": "Ensembl gene IDs",
                "index": "positional indices",
            }.get(naming, "something other than gene symbols")
            candidates = _symbol_column_candidates(adata)
            if candidates:
                st.info(
                    f"The gene names in this file look like {looks_like}. CoPhaser matches "
                    "its rhythmic gene lists by symbol, so the symbols below are used as "
                    "the gene names instead."
                )
                current = st.session_state.gene_symbol_column
                col = st.selectbox(
                    "`.var` column holding the gene symbols",
                    candidates,
                    index=candidates.index(current) if current in candidates else 0,
                )
                if col != st.session_state.gene_symbol_column:
                    _use_symbol_column(adata, col)
                    st.session_state.gene_symbol_column = col
                st.caption(
                    f"e.g. {', '.join(map(str, adata.var_names[:5]))} - the same renaming is "
                    "applied in the generated script, which reads the file from disk again."
                )
            else:
                st.warning(
                    f"The gene names in this file look like {looks_like}, and no `.var` "
                    "column holding gene symbols was found. CoPhaser matches its rhythmic "
                    "gene lists by symbol, so none of them will resolve - add a column of "
                    "symbols to `.var` (or set `.var_names` to symbols) before continuing."
                )

        layer_options = list(adata.layers.keys())
        default_layer = next(
            (l for l in ("spliced", "total") if l in layer_options), layer_options[0]
        )
        layer = st.selectbox(
            "Counts layer to use",
            layer_options,
            index=layer_options.index(default_layer),
        )
        st.session_state.layer = layer

        st.button("Next: QC ->", type="primary", on_click=goto, args=(2,))


# ---------------------------------------------------------------------------
# Step 2: QC + species + cycle
# ---------------------------------------------------------------------------


def _total_counts(adata, layer):
    total = adata.layers[layer].sum(axis=1)
    return np.asarray(getattr(total, "A1", total)).flatten()


def _qc_filtered_adata():
    """The cells kept by the QC thresholds, as the script sees them (cached per thresholds)."""
    adata = st.session_state.adata
    layer = st.session_state.layer
    min_counts, max_counts = st.session_state.min_counts, st.session_state.max_counts
    key = (id(adata), layer, st.session_state.gene_symbol_column, min_counts, max_counts)
    cached = st.session_state.get("_qc_adata")
    if cached is not None and cached[0] == key:
        return cached[1]
    counts = _total_counts(adata, layer)
    keep = np.ones(adata.n_obs, dtype=bool)
    if min_counts is not None:
        keep &= counts >= min_counts
    if max_counts is not None:
        keep &= counts <= max_counts
    filtered = adata if keep.all() else adata[keep].copy()
    st.session_state._qc_adata = (key, filtered)
    return filtered


@st.cache_data(show_spinner="Computing percentile lookup...")
def _percentile_lookup(_counts, cache_key, resolution=1001):
    """Percentile -> count table, so slider moves interpolate instead of re-sorting."""
    grid = np.linspace(0, 100, resolution)
    values = np.percentile(_counts, grid)
    return grid, values


def step_qc():
    st.header("2. Quality control")
    adata = st.session_state.adata
    layer = st.session_state.layer
    counts = _total_counts(adata, layer)
    pct_grid, pct_values = _percentile_lookup(
        counts, (st.session_state.h5ad_path, layer, len(counts))
    )

    # Keyed rather than `value=`: feeding back our own stored value would change the
    # widget's identity each rerun and drop every second slider move.
    st.session_state.setdefault(
        "qc_pct_range", (st.session_state.min_pct, st.session_state.max_pct)
    )
    min_pct, max_pct = st.slider(
        "Total-counts percentile range to keep",
        min_value=0.0,
        max_value=100.0,
        step=0.5,
        key="qc_pct_range",
    )
    min_counts = float(np.interp(min_pct, pct_grid, pct_values))
    max_counts = float(np.interp(max_pct, pct_grid, pct_values))
    st.caption(
        f"{min_pct:.1f}th percentile = {min_counts:.0f} counts, {max_pct:.1f}th percentile = {max_counts:.0f} counts"
    )

    fig, ax = plt.subplots()
    import seaborn as sns

    sns.histplot(x=counts, bins=50, log_scale=True, ax=ax)
    ax.axvline(
        min_counts,
        color="blue",
        linestyle="--",
        label=f"{min_pct:.0f}th percentile: {min_counts:.0f}",
    )
    ax.axvline(
        max_counts,
        color="red",
        linestyle="--",
        label=f"{max_pct:.0f}th percentile: {max_counts:.0f}",
    )
    ax.set_xlabel("Total counts per cell")
    ax.set_ylabel("Number of cells")
    ax.set_title("Distribution of total counts")
    ax.legend()
    _show_fig(fig)

    n_kept = int(((counts >= min_counts) & (counts <= max_counts)).sum())
    st.caption(f"{n_kept}/{adata.n_obs} cells pass these thresholds.")
    st.session_state.n_cells_after_qc = n_kept

    st.session_state.min_pct = min_pct
    st.session_state.max_pct = max_pct
    st.session_state.min_counts = min_counts
    st.session_state.max_counts = max_counts

    detected, diag = species_mod.detect_species(adata.var_names)
    st.subheader("Species")
    st.caption(
        f"Auto-detected: **{detected}** "
        f"({diag['n_human_like']} human-like / {diag['n_mouse_like']} mouse-like / {diag['n_usable']} usable symbols)"
    )
    species_options = ["mouse", "human"]
    default_idx = species_options.index(detected) if detected in species_options else 0
    if detected not in species_options:
        st.warning(
            "Could not tell the species from the gene names - check the choice below, "
            "it selects which packaged gene lists are used."
        )
    species = st.selectbox(
        "Species (override if wrong)", species_options, index=default_idx
    )
    st.session_state.species = species

    st.subheader("Cycle type")
    cycle_options = CYCLES_BY_SPECIES[species]
    # A species switch can leave the stored cycle outside the new options.
    if st.session_state.cycle not in cycle_options:
        st.session_state.cycle = cycle_options[0]
    # Own widget key, not "cycle": Streamlit drops unrendered widgets' state on step change.
    cycle = st.selectbox(
        "Which cycle do you want to infer?",
        cycle_options,
        index=cycle_options.index(st.session_state.cycle),
        format_func=lambda c: CYCLE_LABELS[c],
        key="cycle_select",
    )
    st.session_state.cycle = cycle

    if cycle == "circadian":
        # Circadian signal is weak; warn on low depth (measured on QC-kept cells).
        kept = counts[(counts >= min_counts) & (counts <= max_counts)]
        median_umi = float(np.median(kept)) if kept.size else 0.0
        if median_umi < 5_000:
            st.warning(
                f"Median UMI per cell after QC is low ({median_umi:,.0f}). The circadian "
                "signal is weak at this depth - optimal performance is above 10,000 "
                "counts per cell."
            )
        elif median_umi < 10_000:
            st.info(
                f"Median UMI per cell after QC is {median_umi:,.0f}. Optimal performance "
                "is above 10,000 counts per cell."
            )

    too_few = n_kept < MIN_CELLS_AFTER_QC
    if too_few:
        st.error(
            f"Only {n_kept} cells pass these thresholds; at least {MIN_CELLS_AFTER_QC} are "
            "needed. Widen the percentile range."
        )
    c1, c2 = st.columns(2)
    c1.button("<- Back", on_click=goto, args=(1,))
    c2.button(
        "Next: Configure ->", type="primary", on_click=goto, args=(3,), disabled=too_few
    )


# ---------------------------------------------------------------------------
# Step 3: configure model, hyperparameters, rhythmic genes, device, output
# ---------------------------------------------------------------------------


def _resolve_fixed_rhythmic_genes(adata, cycle, species):
    genes = cycle_configs.rhythmic_genes_for(cycle, species)
    present = [g for g in genes if g in adata.var_names]
    return present, genes


@st.cache_resource(show_spinner="Computing automatic hyperparameters...")
def _auto_hyperparameters_cached(
    _adata, dataset_key, layer, rhythmic_genes: tuple, n_latent, n_harm, n_variable_genes, cycle,
    cycling_status_prior=None,
):
    """`_adata` is the QC-filtered data (not hashed); `dataset_key` identifies it."""
    context_genes = utils.get_variable_genes(
        _adata, n_variable_genes=n_variable_genes, layer=layer
    )
    model = CoPhaser(
        list(rhythmic_genes), context_genes, n_latent=n_latent, n_harm=n_harm
    )
    model.load_anndata(_adata, layer_to_use=layer)
    # Same call as the generated script; adata is only used for cell-cycle scoring.
    data_kwargs = (
        dict(adata=_adata, layer=layer, cycling_status_prior=cycling_status_prior)
        if cycle == "cell_cycle"
        else {}
    )
    return auto_hyperparameters(model, cycle=cycle, verbose=False, **data_kwargs)["trainer"]


def _style_horizontal_scrollbar(container_key):
    """Make a keyed horizontal container's scrollbar visible (default is near-white)."""
    st.markdown(
        f"""
        <style>
        .st-key-{container_key}, .st-key-{container_key} div {{
            scrollbar-color: #6b6b6b #d0d0d0;   /* Firefox: thumb, track */
            scrollbar-width: auto;
        }}
        .st-key-{container_key} ::-webkit-scrollbar,
        .st-key-{container_key}::-webkit-scrollbar {{
            height: 12px;
        }}
        .st-key-{container_key} ::-webkit-scrollbar-thumb,
        .st-key-{container_key}::-webkit-scrollbar-thumb {{
            background: #6b6b6b;
            border-radius: 6px;
        }}
        .st-key-{container_key} ::-webkit-scrollbar-thumb:hover,
        .st-key-{container_key}::-webkit-scrollbar-thumb:hover {{
            background: #4a4a4a;
        }}
        .st-key-{container_key} ::-webkit-scrollbar-track,
        .st-key-{container_key}::-webkit-scrollbar-track {{
            background: #d0d0d0;
            border-radius: 6px;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


GENE_SEPARATORS_RE = re.compile(r"[,;\s]+")


def _drop_extra_gene(gene):
    st.session_state.extra_rhythmic_genes = [
        g for g in st.session_state.extra_rhythmic_genes if g != gene
    ]


def _extra_rhythmic_genes_section(adata, default_genes):
    """Add user rhythmic genes, optionally dropping the packaged ones. Returns the final list."""
    extra = st.session_state.extra_rhythmic_genes

    # Rendered above the form but filled after it, so newly added genes show immediately.
    chips_slot = st.container()

    with st.form("add_rhythmic_genes", clear_on_submit=True):
        typed = st.text_input(
            "Add rhythmic genes",
            placeholder="CCNB1, TOP2A; MKI67",
            help="Separate with commas, semicolons or spaces. Genes that are not in this "
            "dataset are reported and skipped.",
        )
        submitted = st.form_submit_button("Add")

    if submitted and typed.strip():
        by_upper = {}
        for name in adata.var_names:
            by_upper.setdefault(str(name).upper(), str(name))
        added, unknown = [], []
        for token in GENE_SEPARATORS_RE.split(typed.strip()):
            if not token:
                continue
            # Case-insensitive match, stored with the dataset's spelling.
            match = by_upper.get(token.upper())
            if match is None:
                unknown.append(token)
            elif match not in extra:
                added.append(match)
        st.session_state.extra_rhythmic_genes = extra + added
        extra = st.session_state.extra_rhythmic_genes
        if added:
            st.success(f"Added {len(added)}: {', '.join(added)}")
        if unknown:
            st.warning(
                f"Not in this dataset, skipped: {', '.join(unknown)}. Gene names here are "
                "matched against `.var_names`, which must be symbols."
            )

    if extra:
        with chips_slot:
            _style_horizontal_scrollbar("extra_gene_chips")
            with st.container(horizontal=True, wrap=False, key="extra_gene_chips"):
                for gene in extra:
                    st.button(
                        gene,
                        key=f"drop_gene_{gene}",
                        icon=":material/delete:",
                        on_click=_drop_extra_gene,
                        args=(gene,),
                        help=f"Remove {gene}",
                    )

    # Forced on while there are no extra genes, so the list can't end up empty.
    st.session_state.setdefault("use_default_rhythmic", True)
    if not extra:
        st.session_state.use_default_rhythmic = True
    use_default = st.toggle(
        f"Also use the {len(default_genes)} packaged rhythmic genes",
        key="use_default_rhythmic",
        disabled=not extra,
        help="Turn off to train on only the genes you added above. Available once you "
        "have added at least one.",
    )

    genes = (list(default_genes) if use_default else []) + [
        g for g in extra if not use_default or g not in set(default_genes)
    ]
    if extra:
        st.caption(f"Training on {len(genes)} rhythmic genes.")
    return genes


def step_configure():
    st.header("3. Configure")
    adata = st.session_state.adata
    cycle = st.session_state.cycle
    species = st.session_state.species
    layer = st.session_state.layer
    # Somite uses 1000, as in the paper.
    n_variable_genes = cycle_configs.CYCLE_TRAINER_DEFAULTS.get(cycle, {}).get(
        "n_variable_genes", N_VARIABLE_GENES
    )
    st.session_state.n_variable_genes = n_variable_genes

    # ---- rhythmic genes ----
    if cycle == "menstrual":
        st.subheader("Rhythmic genes")
        st.warning(
            "The packaged menstrual gene list targets the **fibroblasts (stromal cells) "
            "of the endometrium** - it is the mutual-information ranking the paper "
            "computed on that cell type. On any other tissue or cell type it is not the "
            "right list: compute your own below instead."
        )
        gene_source = st.radio(
            "Gene source",
            ["Packaged endometrial fibroblast list", "Compute from an annotation"],
            help="The paper selects rhythmic genes by mutual information between "
            "expression and a categorical annotation. The packaged list is that "
            "selection, already computed on endometrial stromal cells.",
        )
        if gene_source == "Packaged endometrial fibroblast list":
            packaged, full_list = _resolve_fixed_rhythmic_genes(adata, cycle, species)
            st.caption(
                f"{len(packaged)}/{len(full_list)} genes scoring above a mutual "
                f"information of {cycle_configs.MENSTRUAL_MI_THRESHOLD} on endometrial "
                "stromal cells are present in this dataset."
            )
            if not packaged:
                st.warning(
                    "None of those genes are in this dataset - are these human gene "
                    "symbols? Compute your own list instead."
                )
            base_genes = packaged
        else:
            candidate_cols = [
                c
                for c in adata.obs.columns
                if adata.obs[c].dtype.name in ("category", "object")
                or adata.obs[c].nunique() < 20
            ]
            if not candidate_cols:
                st.warning(
                    "No categorical `.obs` column to compute mutual information against."
                )
            annotation_col = st.selectbox("Annotation column", candidate_cols)
            mi_threshold = st.number_input(
                "Mutual information threshold",
                value=cycle_configs.MENSTRUAL_MI_THRESHOLD,
                step=0.01,
            )
            if st.button("Compute rhythmic genes", disabled=not candidate_cols):
                with st.spinner("Computing mutual information..."):
                    from sklearn.feature_selection import mutual_info_classif

                    hvgs = utils.get_variable_genes(
                        adata, n_variable_genes=n_variable_genes, layer=layer
                    )
                    X = adata[:, hvgs].layers[layer]
                    X = np.asarray(getattr(X, "toarray", lambda: X)())
                    labels = (
                        adata.obs[annotation_col].astype("category").cat.codes.values
                    )
                    mi = mutual_info_classif(
                        X, labels, discrete_features=False, random_state=0
                    )
                    mi_series = pd.Series(mi, index=hvgs).sort_values(ascending=False)
                    st.session_state.menstrual_mi_genes = mi_series[
                        mi_series > mi_threshold
                    ].index.tolist()
            base_genes = st.session_state.menstrual_mi_genes
            if base_genes:
                st.success(f"{len(base_genes)} rhythmic genes selected.")
    else:
        base_genes, full_list = _resolve_fixed_rhythmic_genes(adata, cycle, species)
        st.subheader("Rhythmic genes")
        st.caption(
            f"{len(base_genes)}/{len(full_list)} packaged {CYCLE_LABELS[cycle]} marker genes "
            f"found in this dataset."
        )

    # Rebuilt from the base list every rerun, so the extra genes never leak into it.
    rhythmic_genes = _extra_rhythmic_genes_section(adata, base_genes)
    st.session_state.rhythmic_genes = rhythmic_genes

    if not st.session_state.rhythmic_genes:
        st.warning("No rhythmic genes resolved yet - required before continuing.")
        st.button("<- Back", on_click=lambda: goto(2))
        return

    # ---- model: sample type -> n_latent ----
    st.subheader("Model")
    sample_type = st.radio(
        "Sample type",
        ["Complex tissue (in vivo)", "Cell line (in vitro)", "Custom"],
        horizontal=True,
        help="n_latent follows the biology rather than a search: 10 for tissues with "
        "substantial non-cycle variability, 2 for cell lines. Rarely worth changing.",
    )
    is_custom = sample_type == "Custom"
    if is_custom:
        locked_value = st.session_state.n_latent
    else:
        locked_value = suggest_n_latent(
            "tissue" if sample_type.startswith("Complex") else "cell_line"
        )
    n_latent = st.number_input(
        "n_latent", value=locked_value, min_value=1, step=1, disabled=not is_custom
    )
    st.session_state.n_latent = n_latent

    base_model_kwargs = (
        dict(cycle_configs.CYCLE_TRAINER_DEFAULTS[cycle]["model"])
        if cycle != "cell_cycle"
        else {}
    )
    n_harm = base_model_kwargs.get("n_harm", 3)
    model_kwargs = dict(base_model_kwargs)
    model_kwargs["n_latent"] = n_latent
    model_kwargs["n_harm"] = n_harm

    # ---- hyperparameters ----
    st.subheader("Hyperparameters")
    qc_adata = _qc_filtered_adata()
    dataset_key = (
        st.session_state.h5ad_path,
        qc_adata.n_obs,
        st.session_state.gene_symbol_column,
        st.session_state.min_counts,
        st.session_state.max_counts,
    )
    if cycle == "cell_cycle":
        auto = _auto_hyperparameters_cached(
            qc_adata,
            dataset_key,
            layer,
            tuple(rhythmic_genes),
            n_latent,
            n_harm,
            n_variable_genes,
            "cell_cycle",
        )
        auto_prior = float(auto.get("cycling_status_prior", 1.0))

        st.caption(
            "Seurat-like G1/S/G2M scoring on the packaged cell-cycle genes "
            f"{'detected' if auto_prior < 1 else 'did not detect'} a high enough G0/G1 fraction "
            "to need the non-cycling (Bernoulli) branch."
        )
        high_g0 = st.toggle(
            "High fraction of G0/G1 (non-cycling) cells", value=auto_prior < 1.0
        )
        if high_g0:
            default_prior = (
                auto_prior if auto_prior < 1.0 else auto_params.NON_CYCLING_PRIOR
            )
            cycling_status_prior = st.number_input(
                "cycling_status_prior (probability a cell is cycling, ≤ 1)",
                value=default_prior,
                min_value=0.01,
                max_value=1.0,
                step=0.05,
            )
            st.caption(
                "The RPE Fourier-coefficient prior will be preloaded and frozen for the genes it "
                "covers, to warm-start the rhythmic decoder when few cells carry direct cycle signal."
            )
        else:
            cycling_status_prior = 1.0
        if cycling_status_prior != auto_prior:
            # the likelihood weights depend on the prior; re-derive them for the chosen one
            auto = _auto_hyperparameters_cached(
                qc_adata,
                dataset_key,
                layer,
                tuple(rhythmic_genes),
                n_latent,
                n_harm,
                n_variable_genes,
                "cell_cycle",
                cycling_status_prior=cycling_status_prior,
            )

        hyperparams_mode = st.radio("Mode", ["autotune", "manual"], horizontal=True)
        if hyperparams_mode == "autotune":
            trainer_kwargs = dict(auto)
            st.caption(
                "Automatically derived from the data (reconstruction-loss weights). The "
                "script recomputes them on the QC-filtered cells; the values below are "
                "that same computation."
            )
            st.json(trainer_kwargs)
        else:
            st.caption(
                "Prefilled from CoPhaser's automatic reconstruction-loss-weight "
                "detection - override anything below."
            )
            trainer_kwargs = {}
            trainer_kwargs["entropy_weight_factor"] = st.number_input(
                "entropy_weight_factor",
                value=float(auto.get("entropy_weight_factor", 100.0)),
            )
            trainer_kwargs["closed_circle_weight"] = st.number_input(
                "closed_circle_weight",
                value=float(auto.get("closed_circle_weight", 10.0)),
            )
            trainer_kwargs["MI_weight"] = st.number_input(
                "MI_weight", value=float(auto.get("MI_weight", 100.0))
            )
            trainer_kwargs["non_rhythmic_likelihood_weight"] = st.number_input(
                "non_rhythmic_likelihood_weight",
                value=float(auto.get("non_rhythmic_likelihood_weight", 1.0)),
            )
            trainer_kwargs["rhythmic_likelihood_weight"] = st.number_input(
                "rhythmic_likelihood_weight",
                value=float(auto.get("rhythmic_likelihood_weight", 1.0)),
            )

        # The toggle above is the single source for cycling_status_prior.
        trainer_kwargs["cycling_status_prior"] = cycling_status_prior
        if cycling_status_prior < 1:
            # As in auto_hyperparameters: keep the warm-started decoder frozen for 20 epochs.
            trainer_kwargs.setdefault("unfreeze_epoch_layer", [(20, "rhythmic_decoder")])
        train_kwargs = dict(n_epochs=200, lr=1e-2)
    else:
        fixed = cycle_configs.CYCLE_TRAINER_DEFAULTS[cycle]
        st.caption(
            f"Based on the paper's configuration (source: {fixed['source']})."
        )
        # autotune = paper config with the likelihood weights re-derived from this data;
        # fixed = the paper's exact numbers.
        mode = st.radio(
            "Mode", ["autotune", "fixed", "manual"], horizontal=True, key="fixed_mode"
        )
        hyperparams_mode = mode
        if mode == "fixed":
            trainer_kwargs = dict(fixed["trainer"])
            st.json(trainer_kwargs)
        else:
            # manual = autotune with these weights overridden.
            auto = _auto_hyperparameters_cached(
                qc_adata,
                dataset_key,
                layer,
                tuple(rhythmic_genes),
                n_latent,
                n_harm,
                n_variable_genes,
                cycle,
            )
            trainer_kwargs = dict(auto)
            if mode == "autotune":
                st.caption(
                    "The likelihood weights are derived from this dataset (every cell is "
                    "taken to carry a phase); the paper's other weights are kept."
                )
                st.json(trainer_kwargs)
            else:
                st.caption(
                    "Prefilled from the derived configuration - override anything below."
                )
                for name, default in (
                    ("entropy_weight_factor", 100.0),
                    ("closed_circle_weight", 10.0),
                    ("MI_weight", 100.0),
                    ("non_rhythmic_likelihood_weight", 1.0),
                    ("rhythmic_likelihood_weight", 1.0),
                ):
                    trainer_kwargs[name] = st.number_input(
                        name, value=float(trainer_kwargs.get(name, default))
                    )
        train_kwargs = dict(fixed["train"])

    st.subheader("Advanced parameters")
    default_mi_detach = "none" if cycle == "somite" else "f"
    mi_detach_options = ["f", "z", "none"]
    show_advanced = st.toggle("Show advanced parameters", value=False)
    if show_advanced:
        n_epochs = st.number_input(
            "n_epochs", value=int(train_kwargs.get("n_epochs", 200)), step=1
        )
        lr = st.number_input(
            "lr", value=float(train_kwargs.get("lr", 1e-2)), format="%.4f"
        )
        mi_detach = st.selectbox(
            "MI_detach",
            mi_detach_options,
            index=mi_detach_options.index(default_mi_detach),
            help="Whether to detach f or z (or neither) when computing the mutual-information "
            "loss between them.",
        )
        beta_kl_f = st.number_input(
            "beta_kl_f (KL weight for the phase posterior)",
            value=float(trainer_kwargs.get("beta_kl_f", 0.1)),
            format="%.4f",
        )
        beta_kl_cycling_status = st.number_input(
            "beta_kl_cycling_status (KL weight between cycling-status posterior and prior; "
            "only used if cycling_status_prior < 1)",
            value=float(trainer_kwargs.get("beta_kl_cycling_status", 10.0)),
        )
    else:
        n_epochs, lr = train_kwargs.get("n_epochs", 200), train_kwargs.get("lr", 1e-2)
        mi_detach, beta_kl_f, beta_kl_cycling_status = default_mi_detach, 0.1, 10.0

    trainer_kwargs["MI_detach"] = mi_detach
    trainer_kwargs["beta_kl_f"] = beta_kl_f
    trainer_kwargs["beta_kl_cycling_status"] = beta_kl_cycling_status
    train_kwargs["n_epochs"] = n_epochs
    train_kwargs["lr"] = lr

    st.session_state.hyperparams_mode = hyperparams_mode
    st.session_state.model_kwargs = model_kwargs
    if hyperparams_mode == "autotune":
        # Only the user's choices: the script derives the rest itself at runtime.
        override_keys = ["MI_detach", "beta_kl_f", "beta_kl_cycling_status"]
        if cycle == "cell_cycle":
            override_keys += ["cycling_status_prior", "unfreeze_epoch_layer"]
        st.session_state.trainer_kwargs = {
            k: trainer_kwargs[k] for k in override_keys if k in trainer_kwargs
        }
    else:
        st.session_state.trainer_kwargs = trainer_kwargs
    # Same rule as fit_with_restarts: warm-start when cycling_status_prior < 1.
    st.session_state.preload_fourier_prior = (
        cycle == "cell_cycle"
        and float(trainer_kwargs.get("cycling_status_prior", 1) or 1) < 1
    )
    # Circadian/somite warm-start the decoder from known amp/phase (tied to their unfreeze
    # schedule, so not optional).
    st.session_state.decoder_amp_phase_prior = (
        dict(gene_sets.amp_phase_circadian) if cycle == "circadian" else None
    )
    st.session_state.decoder_amp_phase_resource = (
        cycle_configs.CYCLE_TRAINER_DEFAULTS.get(cycle, {}).get(
            "decoder_prior_resource"
        )
    )
    n_prior_genes = None
    if cycle == "circadian":
        # only genes in the model get a prior (e.g. just one of Arntl/Bmal1)
        in_model = {g.upper() for g in rhythmic_genes}
        n_prior_genes = sum(
            g.upper() in in_model for g in gene_sets.amp_phase_circadian
        )
    elif st.session_state.decoder_amp_phase_resource:
        n_prior_genes = len(cycle_configs.somite_amp_phase_prior(adata.var_names))
    if n_prior_genes:
        st.caption(
            f"The rhythmic decoder is warm-started from known phases/amplitudes of "
            f"{n_prior_genes} genes, then released at the epoch given by "
            "`unfreeze_epoch_layer`."
        )

    st.subheader("Device")
    cuda_available = torch.cuda.is_available()
    device_options = ["cuda", "cpu"] if cuda_available else ["cpu"]
    if not cuda_available:
        st.caption(
            "No CUDA GPU detected - training will run on CPU (slower). "
            "See the README's installation section for how to set up a CUDA-enabled PyTorch install."
        )
    device = st.selectbox("Device", device_options)
    train_kwargs["device"] = device

    st.subheader("Training speed")
    n_cells = int(st.session_state.n_cells_after_qc or 0)
    subsample_on = st.toggle(
        f"Train on {SUBSAMPLE_CELLS:,} cells resampled each epoch",
        value=True,
        help="Every epoch draws a fresh random subset of this many cells, so the model "
        "still sees the whole dataset over the run while doing less work per epoch. This "
        "has been found to approximate training on all cells well while being "
        "substantially faster. Every cell is assigned a phase either way.",
    )
    # Always explicit: train_model subsamples by default.
    train_kwargs["subsample"] = SUBSAMPLE_CELLS if subsample_on else None
    if not subsample_on:
        st.caption(f"Training on all {n_cells:,} cells - slower on a large dataset.")
    elif n_cells > SUBSAMPLE_CELLS:
        st.caption(
            f"Each epoch visits {SUBSAMPLE_CELLS:,} of {n_cells:,} cells, redrawn every "
            "epoch. Speeds up training and approximates training on the whole dataset "
            f"well; all {n_cells:,} cells still get a phase."
        )
    else:
        st.caption(
            f"This dataset has {n_cells:,} cells, fewer than {SUBSAMPLE_CELLS:,}, so "
            "every cell is used either way."
        )

    st.session_state.train_kwargs = train_kwargs

    st.subheader("Reproducibility")
    # Own widget keys seeded from seed/n_runs, which survive leaving the step.
    seed = int(
        st.number_input(
            "Random seed",
            min_value=0,
            value=max(0, int(st.session_state.seed)),
            step=1,
            key="seed_input",
        )
    )
    n_runs = int(
        st.number_input(
            "Number of runs (different seeds)",
            min_value=1,
            value=max(1, int(st.session_state.n_runs)),
            step=1,
            key="n_runs_input",
            help="Train this exact configuration multiple times with incrementing seeds (to check how "
            "sensitive results are to initialization) - each run is saved in its own subfolder.",
        )
    )
    st.session_state.seed = seed
    st.session_state.n_runs = n_runs
    seeds = list(range(seed, seed + n_runs))
    run_names = [f"seed_{s}" for s in seeds]
    st.session_state.seeds = seeds
    st.session_state.run_names = run_names
    if n_runs > 1:
        st.caption(
            f"Will train {n_runs} times, sequentially, with seeds {seeds[0]}..{seeds[-1]}."
        )

    st.subheader("Output folder")
    # The default name follows the cycle until the user types their own folder.
    auto_dir = st.session_state.get("_auto_output_dir")
    user_edited = st.session_state.output_dir and st.session_state.output_dir != auto_dir
    if not user_edited and not (auto_dir and auto_dir.endswith(f"_{cycle}")):
        auto_dir = os.path.join(
            os.path.expanduser("~"), "cophaser_runs", f"{datetime.now():%Y%m%d_%H%M%S}_{cycle}"
        )
        st.session_state._auto_output_dir = auto_dir
    default_dir = st.session_state.output_dir if user_edited else auto_dir
    output_dir = st.text_input("Output directory", value=default_dir)
    output_dir = os.path.abspath(os.path.expanduser(output_dir))
    st.session_state.output_dir = output_dir
    if n_runs > 1:
        st.caption("Each run is saved under `" + output_dir + "/<seed_name>/`.")
    # Never write a run on top of another one.
    dir_not_empty = os.path.isdir(output_dir) and bool(os.listdir(output_dir))
    if dir_not_empty:
        st.error("This folder is not empty - choose a new output directory.")

    config = build_config()
    st.session_state.run_script = script_gen.build_run_script(config)

    c1, c2 = st.columns(2)
    c1.button("<- Back", on_click=goto, args=(2,))
    c2.button(
        "Next: Run ->", type="primary", on_click=start_run, disabled=dir_not_empty
    )

    st.subheader("Script")
    st.caption(
        "This is exactly what will run when you click Next - copy it to run it yourself from a .py file."
    )
    if st.toggle("Show generated script", value=False):
        # Rendered on request only: the long code block makes the page slow.
        st.code(st.session_state.run_script, language="python")


def build_config():
    layer = st.session_state.layer
    if st.session_state.layer_from_x and layer == "total":
        layer = None  # the script reads the file again, where the counts are in .X
    return dict(
        h5ad_path=st.session_state.h5ad_path,
        layer=layer,
        gene_symbol_column=st.session_state.gene_symbol_column,
        cycle=st.session_state.cycle,
        species=st.session_state.species,
        rhythmic_genes=st.session_state.rhythmic_genes,
        n_variable_genes=st.session_state.n_variable_genes,
        min_counts=st.session_state.min_counts,
        max_counts=st.session_state.max_counts,
        model_kwargs=st.session_state.model_kwargs,
        hyperparams_mode=st.session_state.hyperparams_mode,
        preload_fourier_prior=st.session_state.preload_fourier_prior,
        decoder_amp_phase_prior=st.session_state.decoder_amp_phase_prior,
        decoder_amp_phase_resource=st.session_state.decoder_amp_phase_resource,
        trainer_kwargs=st.session_state.trainer_kwargs,
        train_kwargs=st.session_state.train_kwargs,
        seeds=st.session_state.seeds,
        run_names=st.session_state.run_names,
        output_dir=st.session_state.output_dir,
    )


# ---------------------------------------------------------------------------
# Step 4: run training
# ---------------------------------------------------------------------------


def _reader_thread(proc, q):
    for line in iter(proc.stdout.readline, ""):
        q.put(line.rstrip("\n"))
    proc.stdout.close()
    q.put(None)


def start_run():
    output_dir = st.session_state.output_dir
    os.makedirs(output_dir, exist_ok=True)
    script_path = os.path.join(output_dir, "run.py")
    with open(script_path, "w") as f:
        f.write(st.session_state.run_script)

    q = queue.Queue()
    env = dict(os.environ, PYTHONUNBUFFERED="1")
    proc = subprocess.Popen(
        [sys.executable, "-u", script_path],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env=env,
    )
    threading.Thread(target=_reader_thread, args=(proc, q), daemon=True).start()
    # Lets a run be found (and killed) after its browser session is gone.
    with open(os.path.join(output_dir, "run.pid"), "w") as f:
        f.write(str(proc.pid))
    st.session_state.proc = proc
    st.session_state.log_queue = q
    st.session_state.log_lines = []
    st.session_state.losses = []
    st.session_state.run_finished = False
    st.session_state.run_done_marker = False
    st.session_state.run_cancelled = False
    # Set on the first epoch line, so loading time doesn't skew the ETA.
    st.session_state.run_start_time = None
    st.session_state.current_run_idx = 1
    st.session_state.n_runs_total = len(st.session_state.seeds)
    st.session_state.pop("results_hue_key", None)
    st.session_state.pop("results_run_name", None)
    goto(4)


def step_run():
    st.header("4. Run")

    if st.session_state.proc is None:
        st.info("No run in progress.")
        st.button("<- Back to configure", on_click=goto, args=(3,))
        return

    _run_progress_fragment()


@st.fragment(run_every=1)
def _run_progress_fragment():
    """Live training progress; only this fragment reruns every second."""
    _drain_log_queue()

    if st.session_state.losses:
        run_idx, epoch, n_epochs, _ = st.session_state.losses[-1]
        n_runs_total = st.session_state.n_runs_total
        overall_done = (run_idx - 1) * n_epochs + epoch
        overall_target = n_runs_total * n_epochs
        progress_text = f"Epoch {epoch}/{n_epochs}"
        if n_runs_total > 1:
            progress_text = f"Run {run_idx}/{n_runs_total} - {progress_text}"
        st.progress(min(overall_done / overall_target, 1.0), text=progress_text)

        # Clock starts at the first epoch line, so rate = elapsed / (epochs seen - 1).
        epochs_since_first = overall_done - 1
        if epochs_since_first > 0 and st.session_state.run_start_time is not None:
            elapsed = time.time() - st.session_state.run_start_time
            remaining = max(overall_target - overall_done, 0)
            if remaining > 0:
                eta_seconds = (elapsed / epochs_since_first) * remaining
                st.caption(f"Estimated time left: {_format_duration(eta_seconds)}")

        # Current run only; concatenating runs would show a sawtooth.
        current_run_losses = [
            (e, l) for (r, e, _, l) in st.session_state.losses if r == run_idx
        ]
        loss_df = pd.DataFrame(current_run_losses, columns=["epoch", "loss"])
        st.line_chart(loss_df.set_index("epoch")["loss"])

    failed = st.session_state.run_finished and not st.session_state.get("run_done_marker")
    with st.expander(
        "Full log", expanded=failed or not st.session_state.run_finished
    ):
        with st.container(height=300, autoscroll=True):
            st.code(
                "\n".join(st.session_state.log_lines[-500:]) or "Waiting for output...",
                language=None,
            )

    proc = st.session_state.proc
    if not st.session_state.run_finished:
        if st.button("Cancel run"):
            proc.terminate()
            st.session_state.run_cancelled = True
        return

    try:
        returncode = proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        returncode = None
    with contextlib.suppress(OSError):
        os.remove(os.path.join(st.session_state.output_dir, "run.pid"))

    if returncode == 0 and st.session_state.run_done_marker:
        # App-scoped rerun leaves the fragment timer behind.
        st.success("Training finished - moving to results...")
        goto(5)
        st.rerun()

    if st.session_state.run_cancelled:
        st.warning("Run cancelled.")
    else:
        st.error(
            f"Training failed (exit code {returncode}) - see the end of the log above."
        )
    if st.button("<- Back to configure"):
        _reset_run_state()
        goto(3)
        st.rerun()


def _reset_run_state():
    st.session_state.proc = None
    st.session_state.log_queue = None
    st.session_state.run_finished = False
    st.session_state.log_lines = []
    st.session_state.losses = []
    # Next run gets a fresh default folder instead of overwriting this one.
    st.session_state.output_dir = None
    st.session_state._auto_output_dir = None


def _drain_log_queue():
    q = st.session_state.log_queue
    if q is None:
        return
    while True:
        try:
            line = q.get_nowait()
        except queue.Empty:
            break
        if line is None:
            st.session_state.run_finished = True
            break
        st.session_state.log_lines.append(line)
        rm = RUN_RE.match(line)
        if rm:
            st.session_state.current_run_idx = int(rm.group(1))
            st.session_state.n_runs_total = int(rm.group(2))
        m = EPOCH_RE.match(line)
        if m:
            lm = LOSS_RE.search(line)
            if lm:
                if st.session_state.run_start_time is None:
                    st.session_state.run_start_time = time.time()
                st.session_state.losses.append(
                    (
                        st.session_state.current_run_idx,
                        int(m.group(1)),
                        int(m.group(2)),
                        float(lm.group(1)),
                    )
                )
        if line.strip() == "DONE":
            st.session_state.run_done_marker = True


# ---------------------------------------------------------------------------
# Step 5: results
# ---------------------------------------------------------------------------


def _fig_download_button(fig, label, filename):
    buf = io.BytesIO()
    fig.savefig(buf, format="svg", bbox_inches="tight")
    st.download_button(
        label, data=buf.getvalue(), file_name=filename, mime="image/svg+xml"
    )


def _clipboard_fallback():
    """Make st.code's copy icon work over plain http (the Network URL, remote hosts).

    That icon only calls navigator.clipboard.writeText, which browsers remove outside
    secure contexts (https/localhost), so it silently copies nothing; st.json's icon
    works because it falls back to execCommand("copy"). Give the page the same fallback.
    The component iframe is same-origin, so it can patch the parent window.
    """
    # st.iframe, not st.components.v1.html (deprecated, removal announced for 2026-06-01)
    # and not st.html, which sanitises the markup away: this needs a real iframe so the
    # script runs, and a same-origin one so it can reach window.parent.
    st.iframe(
        """
        <script>
        const w = window.parent;
        if (!w.__cophaserClipboardFallback) {
            w.__cophaserClipboardFallback = true;
            const fallback = (text) => {
                const doc = w.document;
                const active = doc.activeElement;
                const t = doc.createElement("textarea");
                t.value = text;
                t.style.position = "fixed";
                t.style.opacity = "0";
                doc.body.appendChild(t);
                t.select();
                const ok = doc.execCommand("copy");
                doc.body.removeChild(t);
                if (active && active.focus) active.focus();
                return ok ? Promise.resolve() : Promise.reject(new Error("copy failed"));
            };
            if (!w.navigator.clipboard) {
                Object.defineProperty(w.navigator, "clipboard", {
                    value: { writeText: fallback }, configurable: true,
                });
            }
        }
        </script>
        """,
        # This iframe only runs a script and shows nothing; st.iframe requires a positive
        # height (unlike the old components.html, which took 0), so ask for the smallest.
        height=1,
    )


def _results_loading_indicator():
    """While the results page reruns, keep it at full opacity (instead of Streamlit's
    dimming) and show a centred spinner.

    Pure CSS on the app's running state, so it appears as soon as the rerun starts - not
    only once the script reaches an st.spinner - and keeps animating while Python is busy.
    Delayed slightly so quick reruns don't flash it.
    """
    st.markdown(
        """
        <style>
        [data-testid="stElementContainer"][data-stale="true"] {
            opacity: 1 !important;
        }
        [data-testid="stApp"]::before,
        [data-testid="stApp"]::after {
            position: fixed;
            left: 50%;
            z-index: 1000000;
            pointer-events: none;
            opacity: 0;
            visibility: hidden;
        }
        [data-testid="stApp"]::after {
            content: "";
            top: 50%;
            width: 3rem;
            height: 3rem;
            margin: -1.5rem 0 0 -1.5rem;
            border: 0.3rem solid rgba(128, 128, 128, 0.25);
            border-top-color: #ff4b4b;
            border-radius: 50%;
        }
        [data-testid="stApp"]::before {
            content: "Generating plots...";
            top: calc(50% + 2.5rem);
            transform: translateX(-50%);
            padding: 0.2rem 0.8rem;
            border-radius: 1rem;
            background: rgba(255, 255, 255, 0.9);
            color: #31333f;
            font-size: 0.9rem;
            white-space: nowrap;
        }
        [data-testid="stApp"][data-test-script-state="running"]::after {
            visibility: visible;
            animation: cophaser-spin 0.8s linear infinite,
                cophaser-show 0.2s ease 0.4s forwards;
        }
        [data-testid="stApp"][data-test-script-state="running"]::before {
            visibility: visible;
            animation: cophaser-show 0.2s ease 0.4s forwards;
        }
        @keyframes cophaser-spin { to { transform: rotate(360deg); } }
        @keyframes cophaser-show { to { opacity: 1; } }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_resource(show_spinner="Loading trained model and computing outputs...")
def _load_model_and_outputs(model_path, model_mtime, _adata, layer):
    """Cached forward pass. `model_mtime` invalidates it on retraining into the same folder."""
    model = CoPhaser.load(model_path)
    model.load_anndata(_adata, layer_to_use=layer)
    generative_outputs, space_outputs = model.get_outputs()
    return model, generative_outputs, space_outputs


@st.cache_data(show_spinner="Reading each seed's inferred phases...")
def _load_run_phases(output_dir, run_names, mtimes):
    """Each run's inferred phase from its CSV, aligned on obs_names. `mtimes` is a cache key."""
    series = {}
    for name in run_names:
        csv_path = os.path.join(output_dir, name, "inferred_phases.csv")
        if os.path.exists(csv_path):
            series[name] = pd.read_csv(csv_path, index_col=0)["inferred_phase"]
    if not series:
        return {}
    aligned = pd.DataFrame(series).dropna()
    return {name: aligned[name].to_numpy() for name in aligned.columns}


def _seed_agreement_section(output_dir, run_names):
    phases = _load_run_phases(
        output_dir,
        tuple(run_names),
        tuple(
            os.path.getmtime(p)
            for p in (
                os.path.join(output_dir, n, "inferred_phases.csv") for n in run_names
            )
            if os.path.exists(p)
        ),
    )
    if not phases:
        return

    names = list(phases)
    if len(names) == 1:
        st.subheader("Phase distribution")
        fig = gui_plots.phase_distribution_panel(phases[names[0]], f"{names[0]} phase")
        _show_fig(fig, max_width=460)
        return

    st.subheader("Phase agreement between seeds")
    st.caption(
        "One 2D histogram per pair of seeds, with each seed's own phase distribution on the "
        "top and right. Phase is only defined up to a rotation and a reflection, so two "
        "seeds that found the same cycle show a single straight band - of any offset, and "
        "of slope +1 or -1, not necessarily the diagonal. A diffuse cloud means they "
        "disagree, which is the case worth looking into."
    )
    _style_horizontal_scrollbar("seed_agreement_row")
    with st.container(key="seed_agreement_row", horizontal=True, wrap=False):
        for name_a, name_b in itertools.combinations(names, 2):
            fig = gui_plots.phase_agreement_panel(
                phases[name_a], phases[name_b], f"{name_a} phase", f"{name_b} phase"
            )
            _show_fig(fig, max_width=380)


MAX_HUE_CATEGORIES = 10
CAPPED_HUE_COLUMN = "_cophaser_hue"


def _capped_hue_key(adata, hue_key):
    """obs column to colour by; beyond MAX_HUE_CATEGORIES values, the rarest are folded into
    "other" in a derived column (plots treat hue as categorical)."""
    if hue_key is None:
        return None

    values = adata.obs[hue_key]
    if values.nunique(dropna=True) <= MAX_HUE_CATEGORIES:
        return hue_key

    kept = list(values.value_counts().index[:MAX_HUE_CATEGORIES])
    collapsed = values.astype(str).where(values.isin(kept), "other")
    adata.obs[CAPPED_HUE_COLUMN] = pd.Categorical(collapsed)
    n_other = int((collapsed == "other").sum())
    st.warning(
        f"`{hue_key}` has {values.nunique(dropna=True):,} distinct values, too many to "
        f"colour by. Showing the {MAX_HUE_CATEGORIES} most frequent and grouping the "
        f'remaining {n_other:,} cells as "other".'
    )
    return CAPPED_HUE_COLUMN


# A fragment: its button reruns only this section, not every plot above it.
@st.fragment
def _r2_diagnostic_section(adata, layer, model, thetas, space_outputs):
    st.subheader("Cyclic R² vs. peaking phase")
    st.caption("Expensive to compute (clustering + per-gene fit) - only run on demand.")
    min_r2 = 0.1
    if st.button("Compute R² diagnostic"):
        X = adata[:, model.context_genes].layers[layer]
        X = np.asarray(getattr(X, "toarray", lambda: X)())
        selected = set(st.session_state.rhythmic_genes or [])
        with st.spinner("Fitting cyclic R² per gene..."):
            fig, r2_df = gui_plots.r2_polar_panel(
                X,
                thetas,
                space_outputs["z"].detach().numpy(),
                model.context_genes,
                st.session_state.rhythmic_genes,
                min_r2=min_r2,
            )
        _show_fig(fig)
        st.caption(
            "Per-gene cyclic R² (independent of the training-time rhythmic gene set) vs. "
            "peaking phase. Grey labels are the model's own rhythmic genes, for "
            "orientation; black labels are genes outside that set that still clear the R² "
            "threshold - candidates worth adding."
        )
        _fig_download_button(fig, "Download (SVG)", "r2_polar.svg")

        # The black-labelled genes, best first, pasteable into "Add rhythmic genes".
        candidates = r2_df[
            (r2_df["r2"] > min_r2) & (~r2_df["gene"].isin(selected))
        ].sort_values("r2", ascending=False)
        st.caption(
            f"{len(candidates)} genes above R² {min_r2} that are not already rhythmic, "
            "best first - copy into *Add rhythmic genes* on the Configure step to train "
            "with them:"
        )
        st.code(", ".join(candidates["gene"].astype(str)), language=None, wrap_lines=True)


def _request_shutdown():
    """Only flags it; run() exits after rendering the goodbye message."""
    st.session_state._shutdown_requested = True


def _shutdown_control():
    """Sidebar button to stop the server (closing the tab doesn't)."""
    with st.sidebar:
        st.caption("This interface runs a local server on your machine.")
        run_in_progress = (
            st.session_state.proc is not None and st.session_state.proc.poll() is None
        )
        with st.popover("Stop the server", width="stretch"):
            if run_in_progress:
                st.warning(
                    "A training run is still going - stopping the server stops it too."
                )
            st.caption(
                "The page stops responding and you can close the tab. Anything already "
                "written to the output folder is kept."
            )
            st.button("Confirm - stop it", type="primary", on_click=_request_shutdown)


def _perform_shutdown():
    proc = st.session_state.proc
    if proc is not None and proc.poll() is None:
        proc.terminate()
    # Let the goodbye message reach the browser.
    time.sleep(1.0)
    os._exit(0)


def step_results():
    st.header("5. Results")
    _results_loading_indicator()
    # Same cells as the training script and the CSV.
    adata = _qc_filtered_adata()
    layer = st.session_state.layer
    cycle = st.session_state.cycle
    output_dir = st.session_state.output_dir

    run_names = st.session_state.run_names or ["seed_0"]
    _seed_agreement_section(output_dir, run_names)

    if len(run_names) > 1:
        selected_run = st.selectbox(
            "Result (seed) to display", run_names, key="results_run_name"
        )
    else:
        selected_run = run_names[0]
    run_dir = os.path.join(output_dir, selected_run)

    model_path = os.path.join(run_dir, "model.pt")
    model_mtime = os.path.getmtime(model_path)
    model, generative_outputs, space_outputs = _load_model_and_outputs(
        model_path, model_mtime, adata, layer
    )

    hue_key = st.selectbox(
        "Hue (from obs)",
        ["(none)"] + [c for c in adata.obs.columns if c != CAPPED_HUE_COLUMN],
        key="results_hue_key",
    )
    hue_key = None if hue_key == "(none)" else hue_key
    hue_key = _capped_hue_key(adata, hue_key)

    if cycle == "cell_cycle":
        captured = io.StringIO()
        with st.spinner("Rendering validation plots..."), contextlib.redirect_stdout(
            captured
        ):
            fig, _, fig_space, *_ = plotting.plot_cell_cycle_validations(
                model,
                space_outputs,
                generative_outputs,
                adata,
                layer,
                gene_to_upper=(st.session_state.species == "human"),
                hue_key=hue_key,
                return_values=True,
            )
        legend_text = captured.getvalue()
        marker = "Latent space visualization"
        split_at = legend_text.find(marker)
        main_legend, space_legend = (
            (legend_text[:split_at], legend_text[split_at:])
            if split_at != -1
            else (legend_text, "")
        )

        _show_fig(fig)
        st.caption(main_legend)
        _fig_download_button(fig, "Download validation panel (SVG)", "validation.svg")
        _show_fig(fig_space)
        st.caption(space_legend)
        _fig_download_button(
            fig_space, "Download latent space panel (SVG)", "latent_space.svg"
        )
    elif cycle == "circadian":
        hue_values = adata.obs[hue_key].values if hue_key else None

        captured = io.StringIO()
        with st.spinner("Rendering validation plots..."), contextlib.redirect_stdout(
            captured
        ):
            fig, axs, thetas, shown_genes = plotting.plot_circadian_validations(
                model,
                space_outputs,
                generative_outputs,
                adata,
                layer,
                hue_key=hue_key,
                return_values=True,
            )
        _show_fig(fig)
        st.caption(captured.getvalue())
        if any(g is None for g in shown_genes):
            st.caption(
                "Blank panels mean this dataset has fewer than four usable rhythmic genes."
            )
        _fig_download_button(fig, "Download validation panel (SVG)", "validation.svg")

        fig_space = gui_plots.context_and_cycle_space_row(
            space_outputs["z"].detach().numpy(),
            space_outputs["x_projected"].detach().numpy(),
            thetas,
            hue_values=hue_values,
        )
        _show_fig(fig_space, max_width=1100)
        st.caption(
            "Context space coloured by the selected hue and by inferred phase, and the "
            "projected rhythmic (f) space rotated so its angle is the inferred phase. The "
            "phase should look roughly independent of cell identity, since context and "
            "phase are trained to be uninformative of each other; an annulus in the third "
            "panel is a good signal, but depends on the closed_circle_weight used."
        )
        _fig_download_button(fig_space, "Download (SVG)", "context_and_cycle_space.svg")

        _r2_diagnostic_section(adata, layer, model, thetas, space_outputs)
    else:
        thetas = space_outputs["theta"].detach().numpy()
        if cycle == "menstrual":
            genes = st.session_state.rhythmic_genes[:4]
        else:
            genes = list(
                cycle_configs.rhythmic_genes_for(cycle, st.session_state.species)
            )[:4]
        genes = [g for g in genes if g in adata.var_names]
        hue_values = adata.obs[hue_key].values if hue_key else None

        st.subheader("Gene profiles")
        fig1 = gui_plots.gene_profiles_panel(
            adata, layer, thetas, genes, hue=hue_values
        )
        _show_fig(fig1)
        st.caption(
            "Inferred expression of the 4 selected genes, binned by inferred phase (mean "
            "normalized counts). Genes tied to the same process should peak in the same phase window."
        )
        _fig_download_button(fig1, "Download (SVG)", "gene_profiles.svg")

        st.subheader("UMAP")
        fig2 = gui_plots.umap_panel(
            space_outputs["z"].detach().numpy(), hue_values=hue_values, thetas=thetas
        )
        _show_fig(fig2)
        st.caption(
            "UMAP of the non-cyclic context space (top), colored by the selected hue, and by the "
            "inferred phase (bottom). The inferred phase should look roughly independent of cell "
            "identity/context, since context and phase are trained to be uninformative of each other."
        )
        _fig_download_button(fig2, "Download (SVG)", "umap.svg")

        st.subheader("Projected cycle space")
        fig3 = gui_plots.projected_cycle_space_panel(
            space_outputs["x_projected"].detach().numpy(), thetas
        )
        _show_fig(fig3)
        st.caption(
            "Histogram of the projected rhythmic (f) space, rotated so its angle is the inferred "
            "phase. Seeing an annulus is a good signal, but depends on the closed_circle_weight used."
        )
        _fig_download_button(fig3, "Download (SVG)", "projected_cycle_space.svg")

        _r2_diagnostic_section(adata, layer, model, thetas, space_outputs)

    st.subheader("Downloads")
    csv_path = os.path.join(run_dir, "inferred_phases.csv")
    with open(csv_path, "rb") as f:
        st.download_button(
            "Download inferred phases + latent dims (CSV)",
            f,
            file_name=f"inferred_phases_{selected_run}.csv",
        )
    st.caption(f"Model and outputs are also saved on disk under `{run_dir}`.")

    def _start_again():
        _reset_run_state()
        goto(3)

    st.button("<- Start again (same selections)", on_click=_start_again)


# ---------------------------------------------------------------------------


def run():
    st.set_page_config(page_title="CoPhaser", layout="wide")
    st.markdown(
        f"<style>html {{ font-size: {BASE_FONT_SIZE_PX}px; }}</style>",
        unsafe_allow_html=True,
    )
    init_state()
    _clipboard_fallback()
    st.title("CoPhaser")
    st.caption("Context-dependent single-cell Phase inference.")

    steps = {1: step_load, 2: step_qc, 3: step_configure, 4: step_run, 5: step_results}
    # Fixed key so each rerun replaces the step's content instead of stacking copies.
    page = st.container(key="wizard_page")

    _shutdown_control()
    if st.session_state.get("_shutdown_requested"):
        with page:
            st.success("Server stopped - you can close this tab.")
            st.caption("Anything already written to the output folder is kept.")
        _perform_shutdown()
        return

    # On a step change, render one placeholder-only frame first so no stale widgets linger.
    is_first_load = st.session_state.get("_rendered_step") is None
    if st.session_state.get("_rendered_step") != st.session_state.step:
        st.session_state._rendered_step = st.session_state.step
        with page:
            if is_first_load:
                st.markdown("### CoPhaser is loading, please wait...")
            else:
                with st.spinner("Loading next step..."):
                    pass
        st.rerun()
        return

    with page:
        steps[st.session_state.step]()


if __name__ == "__main__":
    run()
