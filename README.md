# BIGS — Text ⇄ Graph Experiments  
---

## 1 · Quick start

### Linux / macOS

```Bash
# 1️⃣ Clone the repo and enter it
git clone <repo-url> bigs
cd bigs

# 2️⃣ Create a venv and activate it
python3 -m venv .venv
source .venv/bin/activate

# 3️⃣ Install Python dependencies
pip install --upgrade pip
pip install -r requirements.txt

# 4️⃣ Export your OpenAI API key
export OPENAI_API_KEY="sk-..."

# 5️⃣ Run the full metric grid (≈ 15 min on GPU, longer on CPU)
python run_bigs_experiments.py
```

### Windows (Powershell)
```Bash
# 1️⃣ Clone the repo and enter it
git clone <repo-url> bigs
cd bigs

# 2️⃣ Create a venv and activate it
python -m venv .venv
.venv\Scripts\Activate.ps1

# 3️⃣ Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# 4️⃣ Set your OpenAI API key
setx OPENAI_API_KEY "sk-..."

# 5️⃣ Run the experiments
python run_bigs_experiments.py
```

## 2 · Project Structure (1-Depth)
```Bash
.
├── data/
│   ├── docs_graphs/                # JSON KGs
│   ├── docs_texts/                 # raw documents (.txt)
│   ├── text2kgbench/               # WebNLG / WebWiki benchmarks
│   ├── textualization/             # triple-level sentences (.pkl)
│   ├── textualization_neighbors/   # neighbor-level paragraphs (.pkl)
│   └── triplet_lists/              # KG triplets as plain text
│
├── graph_to_text/
│   ├── graph_textualization.py           # one sentence per triple
│   ├── graph_neighbor_textualization.py  # one paragraph per node
│   └── benchmarks_textualization.py      # WebNLG / WebWiki triples
│
├── bigs_score.py                 # BIGS ← / → for two corpora
├── bigs_bench.py                 # BIGS on WebNLG / WebWiki splits
├── run_bigs_experiments.py       # batch-runner for our datasets
│
├── results/                      # auto-generated CSV files
│   ├── documents/
│   └── text2kgBench/
│
└── README.md
```


### Dataset notation in the manuscript

| Collection    |  Folder name | Manuscript symbols       |
| ------------- | ------------------ | ------------------- |
| Lonquen       | `lonquen`          | \$Text\_A\$, \$G\_A\$ |
| San Gregorio  | `san_gregorio`     | \$Text\_B\$, \$G\_B\$ |
| 20\_docs      | `20_docs`          | \$Text\_C\$, \$G\_C\$ |


## 3 · Re-create experiments
```Bash
# 1) textualize graphs (triple level) — example for lonquen
python graph_textualization.py --case lonquen \
       --folders original langchain original_res

# 2) textualize neighbour level
python graph_neighbor_textualization.py --case lonquen \
       --folders neighbors/langchain neighbors/original neighbors/original_res

# 3) textualize benchmarks
python benchmarks_textualization.py --bench webwiki --split test
python benchmarks_textualization.py --bench webnlg  --split train

# 4) compute BIGS on our corpora
python run_bigs_experiments.py

# 5) compute BIGS on benchmarks
python bigs_bench.py --bench webwiki --split test
python bigs_bench.py --bench webnlg  --split train
```
