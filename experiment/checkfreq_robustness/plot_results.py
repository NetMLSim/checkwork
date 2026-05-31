#!/usr/bin/env python3
"""
Plot results of the CheckFreq robustness study.

Reads ``results/runs.csv`` and produces PNGs under ``results/plots/``:

  * heatmap_overhead_<model>_<mode>.png
        2-D heatmap of overhead % over (checkpoint_every_n, size_mb)
        for each (model, mode in {checkfreq, sync}). Shared color scale per
        model so checkfreq vs sync are visually comparable.

  * heatmap_overhead_ratio.png
        Paper-ready figure: per (model, checkpoint_every_n, size_mb) cell,
        the ``Sync overhead / CheckFreq overhead`` ratio. One panel per
        model with a shared colour bar; each cell is annotated as
        ``X.XX×``. Use this to show the speedup is workload-invariant.

  * overhead_vs_freq_<model>.png
        Line plot: x = checkpoint_every_n, y = overhead %, one line per
        (mode, size). Confirms how overhead scales with checkpoint frequency.

  * overhead_vs_size_<model>.png
        Line plot: x = size_mb, y = overhead %, one line per (mode, freq).
        Confirms how overhead scales with checkpoint volume.

  * network_sensitivity.png
        Grouped bars: networks on x, modes as bar groups (default cell).

  * storage_sensitivity.png
        Grouped bars: storage tiers on x, modes as bar groups (default cell).

  * qualitative_invariant.png
        Scatter of checkfreq_overhead_pct vs sync_overhead_pct across every
        sampled cell with y = x diagonal. All points should be below the
        diagonal -- this is the qualitative claim the paper makes.

Usage:
  python3 plot_results.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LogNorm, Normalize

SWEEP_DIR = Path(__file__).resolve().parent
RESULTS_DIR = SWEEP_DIR / "results"
PLOTS_DIR = RESULTS_DIR / "plots"
RUNS_CSV = RESULTS_DIR / "runs.csv"

MODE_COLOR = {"checkfreq_like": "#1f77b4", "sync": "#d62728", "baseline": "#7f7f7f"}
MODE_LABEL = {"checkfreq_like": "CheckFreq", "sync": "Sync", "baseline": "Baseline"}


def _ensure_plots_dir() -> None:
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)


def _load() -> pd.DataFrame:
    if not RUNS_CSV.exists():
        sys.exit(f"No runs.csv found at {RUNS_CSV}. Run run_robustness.py first.")
    df = pd.read_csv(RUNS_CSV)
    return df


# ----------------------------------------------------------------------------
# Heatmaps: overhead % vs (freq, size_mb) per (model, mode)
# ----------------------------------------------------------------------------
def plot_heatmaps(df: pd.DataFrame) -> None:
    main = df[(df["tag"] == "main") & (df["mode"].isin(["checkfreq_like", "sync"]))].copy()
    if main.empty:
        print("  [skip] heatmaps: no main-grid rows")
        return
    for model, dfm in main.groupby("model"):
        vmax = dfm["overhead_pct"].max()
        vmin = max(0.0, dfm["overhead_pct"].min())
        # One PNG per mode so the panels can be placed independently in a
        # paper. Both share a single colour scale (vmin/vmax computed across
        # the cf+sync pair for this model) so they remain visually
        # comparable side-by-side.
        for mode in ("checkfreq_like", "sync"):
            dm = dfm[dfm["mode"] == mode]
            pivot = dm.pivot_table(
                index="freq", columns="size_mb", values="overhead_pct", aggfunc="mean"
            ).sort_index(ascending=True)
            if pivot.empty:
                continue
            fig, ax = plt.subplots(figsize=(6.4, 4.6))
            im = ax.imshow(
                pivot.values,
                aspect="auto",
                origin="lower",
                cmap="viridis",
                vmin=vmin, vmax=vmax,
                interpolation="nearest",
            )
            ax.set_xticks(range(len(pivot.columns)))
            ax.set_xticklabels([f"{c:,}" for c in pivot.columns])
            ax.set_yticks(range(len(pivot.index)))
            ax.set_yticklabels([str(i) for i in pivot.index])
            ax.set_xlabel("checkpoint size (MB/rank)")
            ax.set_ylabel("checkpoint_every_n")
            for i, row_v in enumerate(pivot.index):
                for j, col_v in enumerate(pivot.columns):
                    val = pivot.loc[row_v, col_v]
                    if pd.isna(val):
                        continue
                    ax.text(j, i, f"{val:.1f}%", ha="center", va="center",
                            color="white", fontsize=10, weight="bold")
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="overhead (%)")
            fig.tight_layout()
            mode_slug = "checkfreq" if mode == "checkfreq_like" else mode
            out = PLOTS_DIR / f"heatmap_overhead_{model}_{mode_slug}.png"
            fig.savefig(out, dpi=200, bbox_inches="tight")
            plt.close(fig)
            print(f"  wrote {out.relative_to(RESULTS_DIR)}")


# ----------------------------------------------------------------------------
# Paper figure: Sync / CheckFreq overhead ratio per cell (one panel per model)
# ----------------------------------------------------------------------------
_MODEL_DISPLAY = {
    "bert":       "BERT  (24L, 1024d, 512S)",
    "gpt_small":  "GPT-Small  (12L, 1024d, 1024S)",
    "gpt_medium": "GPT-Medium  (24L, 2048d, 2048S)",
}
_MODEL_ORDER = ["bert", "gpt_small", "gpt_medium"]


def plot_overhead_ratio(df: pd.DataFrame) -> None:
    """Per-cell ratio of Sync overhead / CheckFreq overhead, one panel per model.

    The ratio is annotated to two decimal places (e.g. "5.00×") on every
    cell. A shared colour bar across panels makes cross-model comparison
    trivially visual: if every cell sits at the same colour, the speedup
    is invariant across (model, cadence, size).
    """
    main = df[(df["tag"] == "main") & (df["mode"].isin(["checkfreq_like", "sync"]))].copy()
    if main.empty:
        print("  [skip] heatmap_overhead_ratio: no main-grid rows")
        return

    available = [m for m in _MODEL_ORDER if m in main["model"].unique()]
    available += sorted(set(main["model"].unique()) - set(_MODEL_ORDER))
    n = len(available)
    if n == 0:
        return

    # Pre-compute ratio grids per model.
    ratios: dict[str, pd.DataFrame] = {}
    for model in available:
        dfm = main[main["model"] == model]
        cf = dfm[dfm["mode"] == "checkfreq_like"].pivot_table(
            index="freq", columns="size_mb", values="overhead_pct", aggfunc="mean"
        )
        sy = dfm[dfm["mode"] == "sync"].pivot_table(
            index="freq", columns="size_mb", values="overhead_pct", aggfunc="mean"
        )
        # Sort indices so axes are monotone (small -> large).
        cf = cf.sort_index().sort_index(axis=1)
        sy = sy.sort_index().sort_index(axis=1)
        ratios[model] = (sy / cf).replace([np.inf, -np.inf], np.nan)

    # Shared colour scale across panels.
    all_vals = np.concatenate(
        [r.values.flatten() for r in ratios.values() if r.size > 0]
    )
    all_vals = all_vals[~np.isnan(all_vals)]
    rmin, rmax = float(np.min(all_vals)), float(np.max(all_vals))
    # If the data is effectively constant (e.g. uniform 5.00× as in the
    # paper sweep), widen the colour range so the constant value lands in
    # the middle of the colormap rather than at one extreme.
    if rmax - rmin < 0.25:
        mid = (rmin + rmax) / 2.0
        vmin, vmax = mid - 0.5, mid + 0.5
    else:
        pad = max(0.05, (rmax - rmin) * 0.05)
        vmin, vmax = max(0.0, rmin - pad), rmax + pad

    fig, axes = plt.subplots(
        1, n, figsize=(4.7 * n + 0.6, 4.6), sharey=True, squeeze=False
    )
    axes = list(axes[0])

    im = None
    for ax, model in zip(axes, available):
        ratio = ratios[model]
        if ratio.empty:
            ax.set_title(f"{_MODEL_DISPLAY.get(model, model)}\n(no data)")
            continue
        im = ax.imshow(
            ratio.values,
            aspect="auto",
            origin="lower",
            cmap="viridis",
            vmin=vmin, vmax=vmax,
            interpolation="nearest",
        )
        ax.set_xticks(range(len(ratio.columns)))
        ax.set_xticklabels([f"{c:,}" for c in ratio.columns])
        ax.set_yticks(range(len(ratio.index)))
        ax.set_yticklabels([str(i) for i in ratio.index])
        ax.set_xlabel("Checkpoint size per rank (MB)")
        if ax is axes[0]:
            ax.set_ylabel("Checkpoint cadence  (checkpoint_every_n iters)")
        ax.set_title(_MODEL_DISPLAY.get(model, model), fontsize=11)
        for i, row_v in enumerate(ratio.index):
            for j, col_v in enumerate(ratio.columns):
                val = ratio.loc[row_v, col_v]
                if pd.isna(val):
                    continue
                ax.text(
                    j, i, f"{val:.2f}\u00d7",
                    ha="center", va="center",
                    color="white", fontsize=11, weight="bold",
                )

    if im is not None:
        cbar = fig.colorbar(
            im, ax=axes, shrink=0.92, pad=0.02, aspect=22,
        )
        cbar.set_label("Sync overhead / CheckFreq overhead  (\u00d7)", fontsize=10)

    # Headline: spell out min/median/max ratio across the whole sweep so the
    # invariance claim is auditable from the figure alone.
    rmed = float(np.median(all_vals))
    fig.suptitle(
        "Wall-time overhead ratio of synchronous vs. CheckFreq checkpointing across the robustness sweep.\n"
        f"Across {len(all_vals)} cells (3 models \u00d7 4 cadences \u00d7 3 checkpoint sizes), "
        f"the ratio is {rmin:.2f}\u2013{rmax:.2f}\u00d7 (median {rmed:.2f}\u00d7) "
        f"\u2014 Sync consistently pays \u2248{rmed:.2f}\u00d7 the overhead of CheckFreq.",
        fontsize=10.5, y=1.02,
    )
    out = PLOTS_DIR / "heatmap_overhead_ratio.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out.relative_to(RESULTS_DIR)}")


# ----------------------------------------------------------------------------
# 1-D sweeps along freq and size per model
# ----------------------------------------------------------------------------
def _line_plot_overhead(df: pd.DataFrame, x_col: str, line_col: str, out_path: Path, title: str, xlabel: str) -> None:
    if df.empty:
        return
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    line_vals = sorted(df[line_col].unique())
    modes = ["checkfreq_like", "sync"]
    linestyles = {"checkfreq_like": "-", "sync": "--"}
    markers = ["o", "s", "^", "D", "v"]
    for j, lv in enumerate(line_vals):
        for mode in modes:
            d = df[(df[line_col] == lv) & (df["mode"] == mode)].sort_values(x_col)
            if d.empty:
                continue
            ax.plot(
                d[x_col], d["overhead_pct"],
                label=f"{MODE_LABEL[mode]} ({line_col}={lv}{'MB' if line_col=='size_mb' else ''})",
                color=MODE_COLOR[mode],
                linestyle=linestyles[mode],
                marker=markers[j % len(markers)],
                alpha=0.6 + 0.4 * (j / max(1, len(line_vals) - 1)),
            )
    ax.set_xlabel(xlabel)
    ax.set_ylabel("overhead vs baseline wall (%)")
    ax.set_title(title)
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.legend(fontsize=8, ncol=2, loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  wrote {out_path.relative_to(RESULTS_DIR)}")


def plot_lines_freq_size(df: pd.DataFrame) -> None:
    main = df[(df["tag"] == "main") & (df["mode"].isin(["checkfreq_like", "sync"]))].copy()
    for model, dfm in main.groupby("model"):
        _line_plot_overhead(
            dfm, x_col="freq", line_col="size_mb",
            out_path=PLOTS_DIR / f"overhead_vs_freq_{model}.png",
            title=f"Overhead vs checkpoint frequency — model={model}",
            xlabel="checkpoint_every_n (iters between checkpoints)",
        )
        _line_plot_overhead(
            dfm, x_col="size_mb", line_col="freq",
            out_path=PLOTS_DIR / f"overhead_vs_size_{model}.png",
            title=f"Overhead vs checkpoint size — model={model}",
            xlabel="checkpoint size per rank (MB)",
        )


# ----------------------------------------------------------------------------
# Network + storage sensitivity (grouped bars)
# ----------------------------------------------------------------------------
# Preferred ordering for axis values where alphabetical is meaningless.
_ORDER = {
    "storage": ["sata_slow", "nvme_std", "nvme_fast"],
    "network": ["network_8_bw100", "network_8_ring", "network_8_fast", "network_8_balanced"],
}


def _ordered_unique(df: pd.DataFrame, col: str) -> list:
    seen = list(df[col].unique())
    if col in _ORDER:
        return [v for v in _ORDER[col] if v in seen] + [v for v in seen if v not in _ORDER[col]]
    return sorted(seen)


def _grouped_bar(df: pd.DataFrame, x_col: str, out_path: Path, title: str, xlabel: str) -> None:
    if df.empty:
        return
    x_vals = _ordered_unique(df, x_col)
    modes = ["checkfreq_like", "sync"]
    width = 0.35
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    x_pos = np.arange(len(x_vals))
    for i, mode in enumerate(modes):
        ys = []
        for v in x_vals:
            d = df[(df[x_col] == v) & (df["mode"] == mode)]
            ys.append(d["overhead_pct"].mean() if not d.empty else 0.0)
        bars = ax.bar(x_pos + (i - 0.5) * width, ys, width=width,
                      color=MODE_COLOR[mode], label=MODE_LABEL[mode], edgecolor="black", linewidth=0.5)
        for b, y in zip(bars, ys):
            ax.text(b.get_x() + b.get_width() / 2, b.get_height(),
                    f"{y:.1f}%", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x_pos)
    ax.set_xticklabels([str(v) for v in x_vals], rotation=0)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("overhead vs baseline wall (%)")
    ax.set_title(title)
    ax.grid(True, axis="y", linestyle=":", alpha=0.5)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  wrote {out_path.relative_to(RESULTS_DIR)}")


def plot_sensitivity(df: pd.DataFrame) -> None:
    # OFAT slicing: pull EVERY row matching the default cell, regardless of
    # which sweep tag it was originally added under. This guarantees the
    # default cell (which dedup tagged as "main") is included alongside the
    # non-default points.
    default_model = "bert"
    default_freq = 25
    default_size = 2000
    default_storage = "nvme_std"
    default_network = "network_8_balanced"

    base_cell = (
        (df["model"] == default_model)
        & (df["freq"] == default_freq)
        & (df["size_mb"] == default_size)
        & (df["mode"].isin(["checkfreq_like", "sync"]))
    )
    net = df[base_cell & (df["storage"] == default_storage)].copy()
    _grouped_bar(net, "network", PLOTS_DIR / "network_sensitivity.png",
                 f"Network sensitivity — overhead by topology / BW\n"
                 f"(model={default_model}, freq={default_freq}, "
                 f"size={default_size}MB, storage={default_storage})",
                 "network config (topology + bandwidth)")
    st = df[base_cell & (df["network"] == default_network)].copy()
    _grouped_bar(st, "storage", PLOTS_DIR / "storage_sensitivity.png",
                 f"Storage tier sensitivity — overhead by snapshot/persist BW\n"
                 f"(model={default_model}, freq={default_freq}, "
                 f"size={default_size}MB, network={default_network})",
                 "storage tier (snapshot, persist GB/s)")


# ----------------------------------------------------------------------------
# Qualitative invariant: cf overhead vs sync overhead per cell
# ----------------------------------------------------------------------------
def plot_qualitative_invariant(df: pd.DataFrame) -> None:
    pair = (
        df[df["mode"].isin(["checkfreq_like", "sync"])]
        .pivot_table(index=["model", "freq", "size_mb", "storage", "network"],
                     columns="mode", values="overhead_pct", aggfunc="mean")
        .reset_index()
    )
    pair = pair.dropna(subset=["checkfreq_like", "sync"])
    if pair.empty:
        print("  [skip] qualitative_invariant: no paired rows")
        return
    fig, ax = plt.subplots(figsize=(6.2, 5.6))
    models = sorted(pair["model"].unique())
    palette = plt.get_cmap("tab10")
    for i, m in enumerate(models):
        d = pair[pair["model"] == m]
        ax.scatter(d["checkfreq_like"], d["sync"], s=42, alpha=0.75,
                   color=palette(i), label=m, edgecolor="black", linewidth=0.4)
    hi = max(pair["sync"].max(), pair["checkfreq_like"].max()) * 1.05
    lo = 0.0
    ax.plot([lo, hi], [lo, hi], color="black", linestyle="--", linewidth=1.0, label="y = x")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel("CheckFreq overhead (%)")
    ax.set_ylabel("Sync overhead (%)")
    n_below = int((pair["sync"] > pair["checkfreq_like"]).sum())
    n_tot = len(pair)
    ax.set_title(f"Qualitative invariant: CheckFreq < Sync\n({n_below}/{n_tot} cells satisfy sync > checkfreq)")
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.legend(loc="lower right")
    fig.tight_layout()
    out = PLOTS_DIR / "qualitative_invariant.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  wrote {out.relative_to(RESULTS_DIR)}")


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main() -> int:
    _ensure_plots_dir()
    df = _load()
    df = df.dropna(subset=["overhead_pct"])
    print(f"Loaded {len(df)} rows from {RUNS_CSV.relative_to(SWEEP_DIR)}")
    print("Generating plots:")
    plot_heatmaps(df)
    plot_overhead_ratio(df)
    plot_lines_freq_size(df)
    plot_sensitivity(df)
    plot_qualitative_invariant(df)
    print(f"Done. Plots are under {PLOTS_DIR.relative_to(SWEEP_DIR)}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
