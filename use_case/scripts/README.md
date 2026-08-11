# Figure Scripts

Run all included plots with:

```bash
uv run python scripts/reproduce_figures.py
```

The two Matplotlib scripts can also be run separately and accept explicit
input and output paths. `reproduce_tex_figures.py` compiles the two canonical
LaTeX figures. The default paths point to `data/figure_inputs/`,
`figure_sources/`, and `figures/`.

The scripts use only the compact data shipped in this repository. They do not
require the article corpus, graph stores, embeddings, FAISS indexes, or the
row-level BIGS outputs.
