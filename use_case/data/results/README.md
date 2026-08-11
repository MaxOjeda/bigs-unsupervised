# Chilean News Case Study

This directory contains compact, redistributable results for the comparison of
Graphify, GraphRAG, LightRAG, and the Neo4j LLM Knowledge Graph Builder. The
tables correspond to the values reported in the paper and can be checked
without the original multi-gigabyte graph, embedding, and nearest-neighbor
artifacts.

## Contents

- `corpus_summary.csv`: document collection and sentence statistics.
- `graph_bigs_summary.csv`: normalized graph sizes, prepared graph-unit counts,
  and the three BIGS scores.
- `paired_win_loss.csv`: all six document-level pairwise comparisons.
- `threshold_calibration.csv` and `threshold_coverage.csv`: the shared
  document-coverage threshold and the resulting counts.
- `nearest_match_concentration.csv`: concentration measurements used with the
  Lorenz curves.
- `graph_structure.csv`, `graph_degree_summary.csv`,
  `graph_unit_composition.csv`, `graph_categories.csv`, and
  `pagerank_top3.csv`: complementary graph measurements.
- `graph_feature_support.csv` and `query_support.csv`: the qualitative support
  ratings reported in the comparison.
- `construction_cost.csv`: recorded graph-construction resource measurements.
- `tail_findings.csv`, `largest_bigs_distances.csv`, and
  `noise_by_distance_band.csv`: compact tail-analysis results.

The complete article corpus, normalized graphs, graph-text shards, embeddings,
FAISS indexes, and row-level BIGS results are not included. They are not needed
to regenerate the included figures or verify the reported aggregate tables.
The tail counts describe discoveries in fixed review sets and are not estimates
of full-population error rates.
