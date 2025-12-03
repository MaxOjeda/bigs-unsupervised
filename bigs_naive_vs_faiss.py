from sentence_transformers import SentenceTransformer
import numpy as np
import sys
import resource
import faiss
import time
import json
import re
from tqdm.auto import tqdm
from scipy.spatial.distance import cdist
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from textwrap import dedent


serialized_triples_list = []
sentences_list = []

def parse_tsv_json_lines(data_path: Path):
    """Cada línea del archivo es un JSON con 'serialized_triples' y 'sentence'."""
    serialized_triples_list = []
    sentences_list = []

    with open(data_path, "r", encoding="utf-8") as f:
        for line in tqdm(f, desc="Reading lines"):
            obj = json.loads(line)
            serialized_triples_list.append(obj["serialized_triples"])
            sentences_list.append(obj["sentence"])

    print(f"Collected {len(serialized_triples_list)} items.")
    for i in range(min(3, len(serialized_triples_list))):
        print(f"- serialized_triples: {serialized_triples_list[i]}")
        print(f"  sentence          : {sentences_list[i]}")

    return serialized_triples_list, sentences_list


def _validate_finite(name: str, x: np.ndarray) -> None:
    if not np.isfinite(x).all():
        bad_rows = np.unique(np.argwhere(~np.isfinite(x))[:, 0])[:10]
        raise ValueError(f"{name}: found non-finite values; first bad rows: {bad_rows}")

def generate_embeddings(
    originals: list[str],
    generated: list[str],
    model: SentenceTransformer,
    batch_size: int = 64,
) -> tuple[np.ndarray, np.ndarray]:
    device = "cuda" if model.device.type == "cuda" else "cpu"
    print("Using device:", device)
    print("Generating embeddings...")
    orig_emb = model.encode(
        originals, batch_size=batch_size, convert_to_numpy=True, show_progress_bar=True, normalize_embeddings=True, device=device
    )
    gen_emb = model.encode(
        generated, batch_size=batch_size, convert_to_numpy=True, show_progress_bar=True, normalize_embeddings=True, device=device
    )
    print("Embeddings generated.")
    return orig_emb, gen_emb

def bigs_scores_normal(orig_emb, gen_emb, batch_size=8192):
    """Recibe embeddings (torch.Tensor o numpy.ndarray), calcula distances (cosine) en batches y devuelve:
       (score_r, score_r_std, score_r_med, score_l, score_l_std, score_l_med)"""
    import numpy as np
    from scipy.spatial.distance import cdist

    n_orig = orig_emb.shape[0]
    n_gen = gen_emb.shape[0]

    # Right (document → graph)
    right_min = np.empty(n_orig)
    for i in tqdm(range(0, n_orig, batch_size), desc="Right BIGS"):
        batch = orig_emb[i:i+batch_size]
        dists = cdist(batch, gen_emb, metric="cosine")
        right_min[i:i+batch_size] = dists.min(axis=1)

    score_r = right_min.mean()
    score_r_std = right_min.std()
    score_r_med = float(np.median(right_min))

    # Left (graph → document)
    left_min = np.empty(n_gen)
    for i in tqdm(range(0, n_gen, batch_size), desc="Left BIGS"):
        batch = gen_emb[i:i+batch_size]
        dists = cdist(batch, orig_emb, metric="cosine")
        left_min[i:i+batch_size] = dists.min(axis=1)

    score_l = left_min.mean()
    score_l_std = left_min.std()
    score_l_med = float(np.median(left_min))

    return (score_r, score_r_std, score_r_med,
            score_l, score_l_std, score_l_med)


def bigs_scores_hnsw(
    original_embeddings: np.ndarray,
    generated_embeddings: np.ndarray,
    hnsw_M: int = 16,
    ef_construction: int = 200,
    ef_search: int = 128,
) -> tuple[float, float, float, float, float, float, float]:
    """
    Cosine distance via L2-normalized vectors + L2 metric (d2 = 2*(1-cos); cos_dist = d2/2).
    returns: (score_r, score_r_std, score_r_med, score_l, score_l_std, score_l_med, encode_elapsed)
    """

    X = np.ascontiguousarray(original_embeddings, dtype="float32")
    Y = np.ascontiguousarray(generated_embeddings, dtype="float32")

    if X.ndim != 2 or Y.ndim != 2 or X.shape[1] != Y.shape[1]:
        raise ValueError("Embeddings must be 2D and have matching dimensions.")

    _validate_finite("X (norm)", X)
    _validate_finite("Y (norm)", Y)

    print("X min/max:", X.min(), X.max(), "Y min/max:", Y.min(), Y.max())

    d = X.shape[1]


    print("Calculating BIGS -> ...")

    # --- Right: document -> graph ---
    idx_r = faiss.IndexHNSWFlat(d, hnsw_M)  # L2, devuelve distancias L2^2
    idx_r.hnsw.efConstruction = ef_construction
    idx_r.hnsw.efSearch = ef_search

    idx_r.add(Y)  # index sobre Y
    Dr, _ = idx_r.search(X, 1)  # (n_orig, 1) L2^2
    right_min = Dr[:, 0] / 2.0  # cos_dist

    print("Calculating BIGS <- ...")
    # --- Left: graph -> document ---
    idx_l = faiss.IndexHNSWFlat(d, hnsw_M)
    idx_l.hnsw.efConstruction = ef_construction
    idx_l.hnsw.efSearch = ef_search

    idx_l.add(X)  # index sobre X
    Dl, _ = idx_l.search(Y, 1)  # (n_gen, 1) L2^2
    left_min = Dl[:, 0] / 2.0  # cos_dist

    # Stats
    score_r = float(right_min.mean())
    score_r_std = float(right_min.std())
    score_r_med = float(np.median(right_min))

    score_l = float(left_min.mean())
    score_l_std = float(left_min.std())
    score_l_med = float(np.median(left_min))

    return (score_r, score_r_std, score_r_med, score_l, score_l_std, score_l_med)



# ---- tiny memory helpers (simple & cross-platform-ish) ----
def _ru_maxrss_mb() -> float:
    ru = resource.getrusage(resource.RUSAGE_SELF)
    # Linux: KB, macOS/BSD: bytes
    if sys.platform.startswith("linux"):
        return ru.ru_maxrss / 1024.0
    return ru.ru_maxrss / (1024.0 * 1024.0)

def _rss_now_mb() -> float:
    try:
        import psutil
        return psutil.Process().memory_info().rss / (1024.0 * 1024.0)
    except Exception:
        return 0.0

def _measure(callable_fn, *args, **kwargs):
    """Run callable_fn(*args, **kwargs) and return (result, elapsed_s, mem_delta_mb)."""
    t0 = time.perf_counter()
    ru0 = _ru_maxrss_mb()
    rss0 = _rss_now_mb()
    result = callable_fn(*args, **kwargs)
    elapsed = time.perf_counter() - t0
    mem_delta = max(0.0, _ru_maxrss_mb() - ru0, _rss_now_mb() - rss0)
    return result, elapsed, mem_delta


def run_experiment_for_tsv(
    tsv_path: Path,
    model_name: str,
    *,
    batch_size: int = 64,
    hnsw_M: int = 16,
    ef_construction: int = 200,
    ef_search: int = 128,
    use_faiss: bool = True
) -> dict[str, object]:
    """Parse -> embed (timed/mem) -> BIGS (timed/mem). Returns a dict with all stats."""
    # 1) Parse
    serialized_triples, sentences = parse_tsv_json_lines(tsv_path)
    n = len(serialized_triples)

    # 2) Build model and generate embeddings (timed/mem)
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(model_name)

    (embeds, enc_time, enc_mem) = _measure(
        generate_embeddings,
        serialized_triples, sentences, model,
        batch_size=batch_size
    )
    print("Embeddings generated.")
    print(f"Encoding time: {enc_time:.4f}s, memory delta: {enc_mem:.2f} MB")
    
    orig_emb, gen_emb = embeds
    d = int(orig_emb.shape[1])

    # # 3) BIGS with FAISS HNSW (timed/mem)
    if use_faiss == True:
        (bigs, search_time, search_mem) = _measure(
            bigs_scores_hnsw,
            orig_emb, gen_emb,
            hnsw_M=hnsw_M,
            ef_construction=ef_construction,
            ef_search=ef_search,
        )
        (score_r, score_r_std, score_r_med,
        score_l, score_l_std, score_l_med) = bigs


    # 3) BIGS Normal
    else:
        (bigs, search_time, search_mem) = _measure(
            bigs_scores_normal,
            orig_emb, gen_emb
        )
        (score_r, score_r_std, score_r_med,
        score_l, score_l_std, score_l_med) = bigs

    # 4) Package results
    return {
        # dataset / model
        "file": tsv_path.name,
        "faiss": use_faiss,
        "samples": n,
        "model_name": model_name,
        "embedding_dim": d,
        "batch_size": batch_size,
        # hnsw params
        "hnsw_M": hnsw_M,
        "ef_construction": ef_construction,
        "ef_search": ef_search,
        # timings
        "encode_time_s": round(enc_time, 4),
        "search_time_s": round(search_time, 4),
        "total_time_s": round(enc_time + search_time, 4),
        # memory (best-effort process delta)
        "encode_mem_delta_mb": round(enc_mem, 2),
        "search_mem_delta_mb": round(search_mem, 2),
        # BIGS stats
        "bigs_r_mean": round(score_r, 6),
        "bigs_r_std": round(score_r_std, 6),
        "bigs_r_med": round(score_r_med, 6),
        "bigs_l_mean": round(score_l, 6),
        "bigs_l_std": round(score_l_std, 6),
        "bigs_l_med": round(score_l_med, 6),
    }


import csv

CSV_COLS = [
    "file", "faiss", "samples", "model_name", "embedding_dim", "batch_size",
    "hnsw_M", "ef_construction", "ef_search",
    "encode_time_s", "search_time_s", "total_time_s",
    "encode_mem_delta_mb", "search_mem_delta_mb",
    "bigs_r_mean", "bigs_r_std", "bigs_r_med",
    "bigs_l_mean", "bigs_l_std", "bigs_l_med",
]

def _append_rows_to_csv(rows: list[dict[str, object]], out_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    write_header = not out_csv.exists()
    with out_csv.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CSV_COLS)
        if write_header:
            w.writeheader()
        for r in rows:
            w.writerow(r)

def run_experiments(
    tsv_paths: list[Path],
    model_name: str,
    out_csv: Path,
    *,
    batch_size: int = 64,
    hnsw_M: int = 16,
    ef_construction: int = 200,
    ef_search: int = 128,
    verbose_each: bool = True,
    use_faiss: bool = True
) -> list[dict[str, object]]:
    """
    For each TSV path:
      - parse -> embed -> BIGS (using your run_experiment_for_tsv)
      - collect results
    Finally append all rows to out_csv (creates if missing).
    Returns the list of result dicts.
    """
    rows: list[dict[str, object]] = []
    if use_faiss:
        end = len(tsv_paths)
    else:
        end = 3  # Limit to first 3 for normal BIGS (slow)
    for p in tsv_paths[:end]:
        print(f"[run] {p.name} ...")
        print(f"{p.name}: use_faiss={use_faiss}")
        row = run_experiment_for_tsv(
            p,
            model_name,
            batch_size=batch_size,
            hnsw_M=hnsw_M,
            ef_construction=ef_construction,
            ef_search=ef_search,
            use_faiss=use_faiss
        )
        if verbose_each:
            print(f"[done] {p.name}  →  encode={row['encode_time_s']}s  BIGS={row['search_time_s']}s")
        rows.append(row)
        _append_rows_to_csv([row], out_csv)
    #_append_rows_to_csv(rows, out_csv)
    print(f"[csv] wrote {len(rows)} rows → {out_csv}")
    return rows


data_dir = Path("quadruples/")

def sort_key(path: Path) -> int:
    match = re.search(r"(\d+)[kK]", path.name)  
    return int(match.group(1)) if match else 0

tsv_paths = sorted(data_dir.glob("*.tsv"), key=sort_key)
print([p.name for p in tsv_paths])

out_csv = Path("results/scalability/scalability_experiments.csv")
out_csv.parent.mkdir(parents=True, exist_ok=True)

for use_faiss in [False, True]:
    print(f"\n=== Running experiments with use_faiss={use_faiss} ===\n")
    rows = run_experiments(
        tsv_paths=tsv_paths,
        model_name="sentence-transformers/all-mpnet-base-v2",
        out_csv=out_csv,
        batch_size=256,
        hnsw_M=16, ef_construction=200, ef_search=128,
        use_faiss=use_faiss,
    )
    
################### PLOTS #######################

csv_path = Path("results/scalability/scalability_experiments.csv")
df = pd.read_csv(csv_path)
df = df.sort_values("samples")
df["method"] = "hnsw"

N = df["samples"].to_numpy()

# ---------- helper: fit slope (scaling exponent) ----------
def lin_slope(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float)
    m = (x > 0) & (y > 0)
    if m.sum() < 2:
        return np.nan
    return np.polyfit(x[m], y[m], 1)[0]

# ---------- 1) TIME PLOT (linear, by method) ----------
plt.figure(figsize=(9,5))
for method in df['method'].unique():
    d = df[df['method'] == method]
    N_m = d['samples'].to_numpy()
    plt.plot(N_m, d['encode_time_s'], 'o-', label=f"Encode time ({method})")
    plt.plot(N_m, d['search_time_s'], 'o-', label=f"Index+Search time ({method})")
    plt.plot(N_m, d['total_time_s'],  'o-', label=f"Total time ({method})")
    # Add text annotations for N
    for i, txt in enumerate(N_m):
        if i > 2:
            plt.annotate(f"{d['encode_time_s'].iloc[i]:.0f}", (N_m[i], d["encode_time_s"].iloc[i]), textcoords="offset points", xytext=(0,10), ha='center', fontsize=8)
            plt.annotate(f"{d['search_time_s'].iloc[i]:.0f}", (N_m[i], d["search_time_s"].iloc[i]), textcoords="offset points", xytext=(0,-15), ha='center', fontsize=8)
            plt.annotate(f"{d['total_time_s'].iloc[i]:.0f}", (N_m[i], d["total_time_s"].iloc[i]), textcoords="offset points", xytext=(0,25), ha='center', fontsize=8)

plt.xlabel("Dataset size N")
plt.ylabel("Time (seconds)")
plt.title("Relationship Between Dataset Size and Runtime Components.")
plt.ylim(0, 5500)
plt.grid(True, which="both", ls="--", lw=0.5)
plt.legend()
plt.tight_layout()

out_png_time = csv_path.with_name("results/scalability/bigs_hnsw_components.png")
plt.savefig(out_png_time, dpi=160)
print(f"Saved {out_png_time}")

# ---------- 2) TOTAL TIME ONLY (log-log, by method) ----------
csv_path = Path("results/scalability/scalability_experiments.csv")
df = pd.read_csv(csv_path)
naive = df.iloc[:3].copy()
hnsw = df.iloc[3:].copy()

naive["method"] = "naive"
hnsw["method"] = "hnsw"

df = pd.concat([naive, hnsw], ignore_index=True)

N = df["samples"].to_numpy()

plt.figure(figsize=(7,5))
for method in df['method'].unique():
    d = df[df['method'] == method]
    N_m = d['samples'].to_numpy()
    plt.loglog(N_m, d['total_time_s'], 'o-', label=f"Total time ({method})")
    for i, txt in enumerate(N_m):
        plt.annotate(txt, (N_m[i], d["total_time_s"].iloc[i]), textcoords="offset points", xytext=(0,10), ha='center', fontsize=8)

plt.xlabel("Dataset size N")
plt.ylabel("Total Time (seconds)")
plt.title("Total Runtime vs Dataset Size (Log–Log Scale)")
plt.ylim((0, 9000))
plt.grid(True, which="both", ls="--", lw=0.5)
plt.legend()
plt.tight_layout()

out_png_total_time_log = csv_path.with_name("results/scalability/bigs_naive_vs_hnsw.png")
plt.savefig(out_png_total_time_log, dpi=160)
print(f"Saved {out_png_total_time_log}")