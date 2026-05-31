#!/usr/bin/env python3
"""
Plot results of the dense BERT CheckFreq sweep.

Reads ``results/runs.csv`` and writes PNGs under ``results/plots/``:

  * heatmap_overhead_bert.png
        Side-by-side heatmaps of overhead % over
        (checkpoint_every_n, size_mb) for CheckFreq and Sync. Shared
        colour scale per model so the two panels are directly comparable.

  * overhead_vs_size_bert.png
        Line plot, x = size_mb, y = overhead %, one line per (mode, freq).
        Confirms how overhead scales with checkpoint volume across every
        frequency in the dense grid.

  * overhead_vs_freq_bert.png
        Line plot, x = checkpoint_every_n, y = overhead %, one line per
        (mode, size). Confirms how overhead scales with cadence.

  * qualitative_invariant.png
        Scatter of CheckFreq overhead vs Sync overhead per cell with the
        ``y = x`` diagonal. Every point should sit below ``y = x``; this
        is the qualitative claim of the paper.

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
    return pd.read_csv(RUNS_CSV)


# ----------------------------------------------------------------------------
# Heatmaps
# ----------------------------------------------------------------------------
def plot_heatmaps(df: pd.DataFrame) -> None:
    main = df[df["mode"].isin(["checkfreq_like", "sync"])].copy()
    if main.empty:
        print("  [skip] heatmap: no main-grid rows")
        return
    for model, dfm in main.groupby("model"):
        vmax = dfm["overhead_pct"].max()
        vmin = max(0.0, dfm["overhead_pct"].min())
        fig, axes = plt.subplots(1, 2, figsize=(13, 6.2), sharey=True)
        for ax, mode in zip(axes, ["checkfreq_like", "sync"]):
            dm = dfm[dfm["mode"] == mode]
            pivot = dm.pivot_table(
                index="freq", columns="size_mb", values="overhead_pct", aggfunc="mean"
            ).sort_index(ascending=True)
            if pivot.empty:
                ax.set_title(f"{MODE_LABEL[mode]} (no data)")
                continue
            im = ax.imshow(
                pivot.values,
                aspect="auto",
                origin="lower",
                cmap="viridis",
                vmin=vmin, vmax=vmax,
                interpolation="nearest",
            )
            ax.set_xticks(range(len(pivot.columns)))
            ax.set_xticklabels([f"{c:,}" for c in pivot.columns], rotation=30, ha="right")
            ax.set_yticks(range(len(pivot.index)))
            ax.set_yticklabels([str(i) for i in pivot.index])
            ax.set_xlabel("checkpoint size (MB/rank)")
            if ax is axes[0]:
                ax.set_ylabel("checkpoint_every_n")
            ax.set_title(MODE_LABEL[mode])
            # Annotate every cell with its overhead %. With a 10x10 grid we
            # keep the font small so the labels do not overlap.
            for i, row_v in enumerate(pivot.index):
                for j, col_v in enumerate(pivot.columns):
                    val = pivot.loc[row_v, col_v]
                    if pd.isna(val):
                        continue
                    ax.text(j, i, f"{val:.1f}", ha="center", va="center",
                            color="white", fontsize=7, weight="bold")
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="overhead (%)")
        fig.suptitle(
            f"Checkpoint overhead heatmap (dense sweep) -- model={model}\n"
            f"(network={dfm['network'].iloc[0]}, storage={dfm['storage'].iloc[0]}, "
            f"dp=8, num_iterations={int(dfm['num_iterations'].iloc[0])})"
        )
        fig.tight_layout()
        out = PLOTS_DIR / f"heatmap_overhead_{model}.png"
        fig.savefig(out, dpi=150)
        plt.close(fig)
        print(f"  wrote {out.relative_to(RESULTS_DIR)}")


# ----------------------------------------------------------------------------
# 1D line plots
# ----------------------------------------------------------------------------
def _line_plot_overhead(df: pd.DataFrame, x_col: str, line_col: str,
                        out_path: Path, title: str, xlabel: str) -> None:
    if df.empty:
        return
    fig, ax = plt.subplots(figsize=(9.0, 5.2))
    line_vals = sorted(df[line_col].unique())
    modes = ["checkfreq_like", "sync"]
    linestyles = {"checkfreq_like": "-", "sync": "--"}
    markers = ["o", "s", "^", "D", "v", "P", "X", "*", "<", ">"]
    cmap_cf = plt.get_cmap("Blues")
    cmap_sy = plt.get_cmap("Reds")
    n = max(1, len(line_vals))
    for j, lv in enumerate(line_vals):
        # Use a colour gradient inside each mode so the reader can tell
        # which line is which line_col value at a glance.
        shade = 0.35 + 0.6 * (j / max(1, n - 1))
        colors = {"checkfreq_like": cmap_cf(shade), "sync": cmap_sy(shade)}
        unit = "MB" if line_col == "size_mb" else ""
        for mode in modes:
            d = df[(df[line_col] == lv) & (df["mode"] == mode)].sort_values(x_col)
            if d.empty:
                continue
            ax.plot(
                d[x_col], d["overhead_pct"],
                label=f"{MODE_LABEL[mode]} ({line_col}={lv}{unit})",
                color=colors[mode],
                linestyle=linestyles[mode],
                marker=markers[j % len(markers)],
                markersize=5,
                linewidth=1.4,
            )
    ax.set_xlabel(xlabel)
    ax.set_ylabel("overhead vs baseline wall (%)")
    ax.set_title(title)
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.legend(fontsize=7, ncol=2, loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  wrote {out_path.relative_to(RESULTS_DIR)}")


def plot_lines_freq_size(df: pd.DataFrame) -> None:
    main = df[df["mode"].isin(["checkfreq_like", "sync"])].copy()
    for model, dfm in main.groupby("model"):
        _line_plot_overhead(
            dfm, x_col="freq", line_col="size_mb",
            out_path=PLOTS_DIR / f"overhead_vs_freq_{model}.png",
            title=f"Overhead vs checkpoint frequency (dense sweep) -- model={model}",
            xlabel="checkpoint_every_n (iterations between checkpoints)",
        )
        _line_plot_overhead(
            dfm, x_col="size_mb", line_col="freq",
            out_path=PLOTS_DIR / f"overhead_vs_size_{model}.png",
            title=f"Overhead vs checkpoint size (dense sweep) -- model={model}",
            xlabel="checkpoint size per rank (MB)",
        )


# ----------------------------------------------------------------------------
# Qualitative invariant (cf < sync per cell)
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
    fig, ax = plt.subplots(figsize=(6.6, 6.0))
    # Colour each point by checkpoint size so the band structure is visible.
    sc = ax.scatter(pair["checkfreq_like"], pair["sync"],
                    c=pair["size_mb"], cmap="viridis",
                    s=44, alpha=0.85, edgecolor="black", linewidth=0.4)
    hi = max(pair["sync"].max(), pair["checkfreq_like"].max()) * 1.05
    ax.plot([0, hi], [0, hi], color="black", linestyle="--", linewidth=1.0, label="y = x")
    ax.set_xlim(0, hi)
    ax.set_ylim(0, hi)
    ax.set_xlabel("CheckFreq overhead (%)")
    ax.set_ylabel("Sync overhead (%)")
    n_below = int((pair["sync"] > pair["checkfreq_like"]).sum())
    n_tot = len(pair)
    ax.set_title(f"Qualitative invariant -- CheckFreq < Sync\n"
                 f"({n_below}/{n_tot} cells satisfy sync > checkfreq)")
    cb = fig.colorbar(sc, ax=ax, label="checkpoint size (MB/rank)")
    cb.ax.tick_params(labelsize=8)
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
    plot_lines_freq_size(df)
    plot_qualitative_invariant(df)
    print(f"Done. Plots are under {PLOTS_DIR.relative_to(SWEEP_DIR)}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
