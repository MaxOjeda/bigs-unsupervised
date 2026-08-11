# Data Dictionary

## Shared terms

- `program`: `graphify`, `graphrag`, `lightrag`, or `neo4j`.
- `graph_units`: texts embedded on the graph side of BIGS. Depending on the
  program, these are verbalized graph statements, directly embedded
  descriptions, or both.
- `bigs_right`: mean nearest-neighbor distance from document units to graph
  units.
- `bigs_left`: mean nearest-neighbor distance from graph units to document
  units.
- `f1_bigs`: harmonic mean of `bigs_right` and `bigs_left`.
- `distance`: cosine distance between normalized BGE-M3 embeddings. Lower is
  closer.

## Paired comparison

For each document unit, the four program distances are compared pairwise. A
program wins when its distance is more than 0.01 lower than its competitor's;
differences within 0.01 are ties. The six program pairs cover every document
unit exactly once.

## Threshold analysis

The threshold is pooled across the four programs. A pair with distance above
`tau = 0.43` is classified as unrelated for this downstream analysis. The
calibration table records the sampled tail size, binary-label counts, and the
bootstrap interval for the observed related-pair proportion.

The threshold is not part of BIGS and is not a universal semantic constant.

## Tail analysis

The candidate lists contain the 1,000 largest distances for each program and
direction. The reported findings are reviewed cases discovered in these fixed
lists. No population prevalence should be inferred from those counts.

## Qualitative support tables

The values `direct`, `limited`, and `none` summarize whether each program's
delivered graph directly supports, supports only with restrictions or extra
processing, or does not support the listed feature or query type.
