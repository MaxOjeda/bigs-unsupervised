# BIGS Chilean News Use-Case Artifacts

This repository contains the compact data and plotting code for the Chilean
news use case in the BIGS paper. Its scope is limited to that case study and
the appendix material that supports it.

The study compares Graphify, GraphRAG, LightRAG, and the Neo4j LLM Knowledge
Graph Builder on 68,569 Spanish-language news articles published by six
Chilean news media between June and December 2025.

## Included

- aggregate corpus, graph, BIGS, paired-comparison, threshold, tail, and
  structural results under `data/results/`;
- compact histogram and curve data under `data/figure_inputs/`;
- the figure-generation programs under `scripts/` and the two canonical
  LaTeX figure sources under `figure_sources/`;
- the PDF and PNG figures under `figures/`;
- consistency tests and SHA-256 checksums.

These files cover the results shown in the use-case section and its supporting
appendix tables. See [docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md) for the
mapping between reported results and repository files.

## Not included

The article corpus, normalized graphs, verbalized graph units, embeddings,
retrieval indexes, and row-level nearest-neighbor results are too large or
cannot be redistributed. They are not required to regenerate the included
figures or verify the aggregate values reported in the paper.

This repository does not contain artifacts from the paper's earlier controlled
experiments.

## Reproduce the figures

Python 3.11 or newer and [uv](https://docs.astral.sh/uv/) are recommended.

```bash
uv sync
```

```bash
uv run python scripts/reproduce_figures.py
```

Run the consistency checks with:

```bash
uv run pytest -q
```

Verify the packaged data directly with:

```bash
sha256sum -c data/checksums.sha256
```

## Interpretation

BIGS-right compares document units with graph units, while BIGS-left compares
graph units with document units. Lower distances and lower BIGS scores indicate
closer semantic correspondence. The paired comparison uses a distance
tolerance of 0.01. The threshold analysis uses one pooled threshold,
`tau = 0.43`, for all four programs.

The tail-review counts are discoveries within fixed review sets. They are not
estimates of full-population error rates.

## License

Code is released under the MIT License. The compact result tables and figures
are provided for research reproducibility; third-party program names remain the
property of their respective owners.
