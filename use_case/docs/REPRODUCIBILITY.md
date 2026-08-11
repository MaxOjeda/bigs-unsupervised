# Reproducibility Guide

## Scope

This package covers only the Chilean news use case and the appendix material
that documents that use case. It starts from compact result tables rather than
the original articles, graphs, embeddings, or row-level nearest-neighbor data.

## Result mapping

| Reported result | Repository source |
| --- | --- |
| Corpus statistics | `data/results/corpus_summary.csv` |
| Graph sizes and BIGS scores | `data/results/graph_bigs_summary.csv` |
| Paired win/loss table | `data/results/paired_win_loss.csv` |
| Threshold calibration and coverage | `data/results/threshold_calibration.csv`, `data/results/threshold_coverage.csv` |
| Distance-distribution figure | `data/figure_inputs/distance_distributions/` |
| Win-rate and threshold-coverage figure | `data/figure_inputs/winrate_tau_coverage.csv` |
| Nearest-match concentration | `data/results/nearest_match_concentration.csv`, `data/figure_inputs/concentration_lorenz/` |
| Tail noise review | `data/results/noise_by_distance_band.csv` |
| Tail findings and selected examples | `data/results/tail_findings.csv`, `data/figure_inputs/tail_selected_examples.csv` |
| Graph structure and degree tables | `data/results/graph_structure.csv`, `data/results/graph_degree_summary.csv` |
| Graph-unit composition | `data/results/graph_unit_composition.csv` |
| Frequent graph categories and PageRank | `data/results/graph_categories.csv`, `data/results/pagerank_top3.csv` |
| Feature and query comparison | `data/results/graph_feature_support.csv`, `data/results/query_support.csv` |
| Construction measurements | `data/results/construction_cost.csv` |
| Largest observed distances | `data/results/largest_bigs_distances.csv` |

The CSV files are the full machine-readable versions of the main-text and
appendix tables. Their column names preserve the quantities reported in the
paper.

## Figure reproduction

Install the locked Python environment and regenerate every included figure:

```bash
uv sync
```

```bash
uv run python scripts/reproduce_figures.py
```

Each script can also be run separately. Its command-line help lists the input
and output arguments. The Lorenz and selected-tail-example figures use the
canonical LaTeX sources in `figure_sources/`; the full command therefore also
requires a TeX installation with `latexmk` and `pdftoppm`.

## Verification

The tests check agreement among the graph summary, threshold table, paired
comparisons, plotted values, distribution counts, Lorenz curves, and noise
review counts.

```bash
uv run pytest -q
```

The data files are independently covered by `data/checksums.sha256`:

```bash
sha256sum -c data/checksums.sha256
```

## Reproducibility boundary

The compact package supports regeneration of all included plots and direct
verification of the values reported in the use-case tables. Recomputing graph
construction, embeddings, nearest-neighbor searches, or individual BIGS rows
requires the excluded source artifacts and computing environment.
