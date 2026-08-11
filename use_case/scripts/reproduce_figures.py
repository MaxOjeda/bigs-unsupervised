#!/usr/bin/env python3
"""Regenerate every included figure for the news case study."""

from __future__ import annotations

from common import FIGURES, INPUTS
from plot_bigs_distance_violins import plot as plot_distances
from plot_winrate_tau_coverage import plot as plot_winrate
from reproduce_tex_figures import reproduce as reproduce_tex_figures


def main() -> None:
    outputs = [
        *plot_distances(INPUTS / "distance_distributions", FIGURES),
        *plot_winrate(INPUTS / "winrate_tau_coverage.csv", FIGURES),
        *reproduce_tex_figures(),
    ]
    for path in outputs:
        print(path)


if __name__ == "__main__":
    main()
