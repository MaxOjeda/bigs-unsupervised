"""Shared paths and colors for the news case-study figures."""

from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
INPUTS = ROOT / "data" / "figure_inputs"
FIGURES = ROOT / "figures"

PROGRAMS = ("graphify", "graphrag", "lightrag", "neo4j")
LABELS = {
    "graphify": "Graphify",
    "graphrag": "GraphRAG",
    "lightrag": "LightRAG",
    "neo4j": "Neo4j",
}
COLORS = {
    "graphify": "#1e66f5",
    "graphrag": "#8839ef",
    "lightrag": "#179299",
    "neo4j": "#d20f39",
}
INK = "#1f1f28"
GRID = "#c8ccd4"
