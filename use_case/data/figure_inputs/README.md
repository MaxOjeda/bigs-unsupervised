# Figure Inputs

These compact files regenerate the news case-study figures without the
row-level BIGS results.

- `distance_distributions/`: fixed-width histograms and distribution
  summaries for all four programs in both BIGS directions.
- `concentration_lorenz/`: Lorenz-curve points for nearest-match selection
  frequencies.
- `winrate_tau_coverage.csv`: aggregate paired win rates, document coverage,
  and graph-unit counts.
- `tail_selected_examples.csv`: the text shown in the selected tail-example
  figure.

Run `scripts/reproduce_figures.py` from the repository root to
recreate the corresponding PDF and PNG files under `figures/`.
