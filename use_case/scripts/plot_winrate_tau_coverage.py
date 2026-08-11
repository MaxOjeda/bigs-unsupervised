#!/usr/bin/env python3
"""Plot paired win rate against threshold-based document coverage."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt

from common import COLORS, FIGURES, GRID, INK, INPUTS, LABELS


def _read_rows(path: Path) -> list[dict[str, float | str]]:
    rows: list[dict[str, float | str]] = []
    with path.open(encoding="utf-8", newline="") as handle:
        for raw in csv.DictReader(handle):
            rows.append(
                {
                    "program": raw["program"],
                    "win_rate": float(raw["pairwise_win_rate_percent"]),
                    "coverage": float(raw["tau_coverage_percent"]),
                    "graph_units": float(raw["graph_units"]),
                }
            )
    return rows


def plot(input_path: Path, output_dir: Path) -> tuple[Path, Path]:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "axes.labelsize": 12,
            "xtick.labelsize": 10.5,
            "ytick.labelsize": 10.5,
            "axes.edgecolor": "black",
            "axes.linewidth": 1.2,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    rows = _read_rows(input_path)
    max_units = max(float(row["graph_units"]) for row in rows)
    fig, ax = plt.subplots(figsize=(7.2, 4.65), constrained_layout=True)
    offsets = {
        "graphify": (-12, -20),
        "graphrag": (42, 2),
        "lightrag": (-14, -20),
        "neo4j": (-12, 14),
    }

    for row in rows:
        program = str(row["program"])
        size = 720.0 * float(row["graph_units"]) / max_units
        x = float(row["win_rate"])
        y = float(row["coverage"])
        ax.scatter(
            x,
            y,
            s=size,
            facecolor=COLORS[program],
            edgecolor="none",
            linewidth=0.0,
            alpha=0.82,
            zorder=3,
        )
        ax.annotate(
            LABELS[program],
            (x, y),
            xytext=offsets[program],
            textcoords="offset points",
            ha="center",
            va="center",
            fontsize=10.5,
            fontweight="bold",
            color=INK,
            zorder=4,
        )

    ax.set_xlim(20, 90)
    ax.set_ylim(95, 100.15)
    ax.set_xticks(range(20, 91, 10))
    ax.set_yticks((95, 96, 97, 98, 99, 100))
    ax.set_xlabel("Pairwise win rate (%)")
    ax.set_ylabel(r"Threshold-based coverage at $\tau=0.43$ (%)")
    ax.grid(color=GRID, linewidth=0.9, alpha=0.72)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    output_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = output_dir / "use_case_winrate_tau_coverage_bubbles.pdf"
    png_path = output_dir / "use_case_winrate_tau_coverage_bubbles.png"
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return pdf_path, png_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default=INPUTS / "winrate_tau_coverage.csv",
    )
    parser.add_argument("--output-dir", type=Path, default=FIGURES)
    args = parser.parse_args()
    for path in plot(args.input, args.output_dir):
        print(path)


if __name__ == "__main__":
    main()
