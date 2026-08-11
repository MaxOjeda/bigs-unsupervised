#!/usr/bin/env python3
"""Plot the eight BIGS distance distributions from fixed-bin summaries."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from common import COLORS, FIGURES, GRID, INK, INPUTS, LABELS, PROGRAMS


def _gaussian_smooth(values: np.ndarray, sigma_bins: float = 1.5) -> np.ndarray:
    radius = max(1, int(np.ceil(4 * sigma_bins)))
    offsets = np.arange(-radius, radius + 1, dtype=float)
    kernel = np.exp(-0.5 * (offsets / sigma_bins) ** 2)
    kernel /= kernel.sum()
    return np.convolve(values, kernel, mode="same")


def _read_distribution(path: Path) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    bins = payload["bins"]
    centers = np.asarray([(row["low"] + row["high"]) / 2 for row in bins])
    counts = np.asarray([row["count"] for row in bins], dtype=float)
    density = _gaussian_smooth(counts / counts.sum())
    return centers, density, payload["summary"]


def _draw_violin(
    ax: plt.Axes,
    *,
    x: float,
    centers: np.ndarray,
    density: np.ndarray,
    summary: dict[str, float],
    color: str,
) -> None:
    visible = (centers >= 0.0) & (centers <= 0.75)
    y = centers[visible]
    width = density[visible]
    width = 0.39 * width / width.max()
    outline = width >= 0.01 * width.max()
    y = y[outline]
    width = width[outline]

    ax.fill_betweenx(
        y,
        x - width,
        x + width,
        facecolor=color,
        edgecolor="none",
        alpha=0.82,
        zorder=2,
    )
    ax.plot(x - width, y, color=color, linewidth=1.3, zorder=3)
    ax.plot(x + width, y, color=color, linewidth=1.3, zorder=3)
    ax.vlines(x, summary["p25"], summary["p75"], color=INK, linewidth=3.0, zorder=4)
    ax.scatter(
        [x],
        [summary["p50"]],
        s=28,
        facecolor="white",
        edgecolor=INK,
        linewidth=1.0,
        zorder=5,
    )


def plot(input_dir: Path, output_dir: Path) -> tuple[Path, Path]:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "axes.labelsize": 12,
            "axes.titlesize": 12,
            "xtick.labelsize": 10.5,
            "ytick.labelsize": 10.5,
            "axes.edgecolor": "black",
            "axes.linewidth": 1.2,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    fig, axes = plt.subplots(1, 2, figsize=(8.0, 4.35), sharey=True, constrained_layout=True)
    panels = (
        (axes[0], "right", "BIGS→: documents to graph", "(a)"),
        (axes[1], "left", "BIGS←: graph to documents", "(b)"),
    )
    positions = np.arange(1, len(PROGRAMS) + 1)

    for ax, suffix, title, panel_label in panels:
        for x, program in zip(positions, PROGRAMS, strict=True):
            centers, density, summary = _read_distribution(
                input_dir / f"{program}-{suffix}.json"
            )
            _draw_violin(
                ax,
                x=float(x),
                centers=centers,
                density=density,
                summary=summary,
                color=COLORS[program],
            )

        ax.set_title(title, pad=9)
        ax.set_xlim(0.45, len(PROGRAMS) + 0.55)
        ax.set_ylim(0.0, 0.75)
        ax.set_xticks(positions, [LABELS[p] for p in PROGRAMS])
        ax.tick_params(axis="x", rotation=20)
        ax.grid(axis="y", color=GRID, linewidth=0.9, alpha=0.72)
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.text(
            0.02,
            0.97,
            panel_label,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=14,
            fontweight="bold",
        )

    axes[0].set_ylabel("Cosine distance")
    output_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = output_dir / "use_case_bigs_distance_violins.pdf"
    png_path = output_dir / "use_case_bigs_distance_violins.png"
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return pdf_path, png_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=INPUTS / "distance_distributions",
    )
    parser.add_argument("--output-dir", type=Path, default=FIGURES)
    args = parser.parse_args()
    for path in plot(args.input_dir, args.output_dir):
        print(path)


if __name__ == "__main__":
    main()
