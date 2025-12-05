
# BIGS — Text ⇄ Graph Experiments
---

## Quick start

### Linux / macOS

```bash
# 1️⃣ Clone the repo and enter it
git clone <repo-url> bigs-unsupervised
cd bigs-unsupervised

# 2️⃣ Create and activate a conda environment
conda create -n bigs python=3.9
conda activate bigs

# 3️⃣ Install Python dependencies
pip install --upgrade pip
pip install -r requirements.txt

# 4️⃣ Export your OpenAI API key
export OPENAI_API_KEY="sk-..."
```

## Project Structure
```bash
.
├── data/
│   ├── docs_texts/                 # raw documents (.txt)
│   ├── text2kgbench/               # WebNLG / WebWiki benchmarks
│   ├── textualization/             # triple-level sentences (.pkl)
│   ├── textualization_neighbors/   # neighbor-level paragraphs (.pkl)
│   └── triplet_lists/              # KG triplets as plain text
│
├── graph_to_texts/
│   ├── graph_textualization.py           # one sentence per triple
│   ├── graph_neighbor_textualization.py  # one paragraph per node
│   └── benchmarks_textualization.py      # WebNLG / WebWiki triples
│
├── bigs_score.py                 # BIGS ← / → for two corpora
├── bigs_bench.py                 # BIGS on WebNLG / WebWiki splits
├── run_bigs.py                   # batch-runner for our datasets
│
├── results/                      # auto-generated CSV files
│   ├── scalability/
│   └── CSV files
│
└── README.md
```

### Dataset notation

| Collection | Folder name | Manuscript symbols |
|------------|-------------|-------------------|
| Japan      | `japan`     | $Text_A$, $G_A$   |
| Croatia    | `croatia`   | $Text_B$, $G_B$   |

## Verbalization

Textualize graphs (triple level):
```bash
python graph_to_texts/graph_textualization.py --case japan --folders original langchain
python graph_to_texts/graph_textualization.py --case croatia --folders original langchain
```

Textualize neighbor level:
```bash
python graph_to_texts/graph_neighbor_textualization.py --case japan --folders neighbors/langchain neighbors/original
python graph_to_texts/graph_neighbor_textualization.py --case croatia --folders neighbors/langchain neighbors/original
```

Textualize benchmarks:
```bash
python graph_to_texts/benchmarks_textualization.py --bench webwiki --split test
```

## Re-create experiments

Experiments on graph refinement:
```bash
python bigs_refinement.py
```

Experiments on verbalization methods:
```bash
python bigs_verbalization_comparison.py
```

Experiments on benchmarks:
```bash
python bigs_bench.py --bench webwiki --split test
```

Experiments on Japan and Croatia datasets:
```bash
python run_bigs.py
```
