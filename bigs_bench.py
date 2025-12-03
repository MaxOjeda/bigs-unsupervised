import argparse
import csv
import pickle
import time
from pathlib import Path
from typing import List, Tuple
import resource
import faiss
import sys

import nltk
import numpy as np
import torch
from nltk.tokenize import sent_tokenize
from scipy.spatial.distance import cdist
from sentence_transformers import SentenceTransformer

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

def _validate_finite(name: str, x: np.ndarray) -> None:
    if not np.isfinite(x).all():
        bad_rows = np.unique(np.argwhere(~np.isfinite(x))[:, 0])[:10]
        raise ValueError(f"{name}: found non-finite values; first bad rows: {bad_rows}")


# --------------------------------------------------------------------------- #
# Text utilities
# --------------------------------------------------------------------------- #
def chunk_text(text: str, size: int, overlap: int):
    """
    Split *text* into word-level chunks of length *size*
    with *overlap* words shared between consecutive chunks.
    """
    tokens = text.split()
    step = size - overlap
    return [" ".join(tokens[i : i + size]) for i in range(0, len(tokens), step)]


# --------------------------------------------------------------------------- #
# Data loading
# --------------------------------------------------------------------------- #
def load_category_names(bench: str, split_type: str):
    """Return all category IDs available for the given benchmark and split."""
    base = Path("data/text2kgbench")
    folder = base / bench / split_type / "sentences"
    return [p.stem.split("_")[-1] for p in folder.iterdir() if p.is_file()]


def read_files(bench: str, split_type: str, category: str, split: str, *, chunk_size: int = 15, overlap: int = 2):
    """Load and optionally split the originals; return originals and generated texts."""
    base_sent = Path("data/text2kgbench") / bench / split_type / "sentences"
    base_kg = Path("data/textualization") / bench / split_type

    originals_path = base_sent / f"sentences_{bench}_{split_type}_{category}.txt"
    kg_path = base_kg / f"triples_{bench}_{split_type}_{category}.pkl"

    with originals_path.open(encoding="utf-8") as fh:
        originals = [line.strip() for line in fh]

    with kg_path.open("rb") as fh:
        generated = pickle.load(fh)

    # Optional chunking
    if split == "chunks":
        originals = chunk_text(" ".join(originals), chunk_size, overlap)

    return originals, generated


# --------------------------------------------------------------------------- #
# Scoring
# --------------------------------------------------------------------------- #

def generate_embeddings(originals: list[str], generated: list[str], model: SentenceTransformer, batch_size: int = 64) -> tuple[np.ndarray, np.ndarray]:
    print("Generating embeddings...")
    orig_emb = model.encode(originals, batch_size=batch_size, convert_to_numpy=True, show_progress_bar=True, normalize_embeddings=True)
    gen_emb = model.encode(generated, batch_size=batch_size, convert_to_numpy=True, show_progress_bar=True, normalize_embeddings=True)
    print("Embeddings generated.")
    return orig_emb, gen_emb

def bigs_scores(originals: List[str], generated: List[str], model: SentenceTransformer, batch_size: int):
    """Compute BIGS left/right scores and basic statistics."""
    start = time.time()
    orig_emb = model.encode(originals, batch_size=batch_size, convert_to_tensor=True, show_progress_bar=True)
    gen_emb = model.encode(generated, batch_size=batch_size, convert_to_tensor=True, show_progress_bar=True)
    elapsed = time.time() - start

    distances = cdist(orig_emb.cpu(), gen_emb.cpu(), metric="cosine")

    # Right (document → graph)
    right_min = distances.min(axis=1)
    score_r = right_min.mean()
    score_r_std = right_min.std()
    score_r_med = float(np.median(right_min))

    # Left (graph → document)
    left_min = distances.min(axis=0)
    score_l = left_min.mean()
    score_l_std = left_min.std()
    score_l_med = float(np.median(left_min))

    print(f"BIGS→  mean={score_r:.4f}  std={score_r_std:.4f}  median={score_r_med:.4f}")
    print(f"BIGS←  mean={score_l:.4f}  std={score_l_std:.4f}  median={score_l_med:.4f}")
    return (score_r, score_r_std, score_r_med,
        score_l, score_l_std, score_l_med, elapsed)

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

    # en macbook m1 necesitaba agregar esto o daba error:
    #try:
    #    faiss.omp_set_num_threads(1)
    #except Exception:
    #    pass

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


# ---------------------------------------------------------------------------- #
if __name__ == "__main__":
    nltk.download("punkt", quiet=True)

    parser = argparse.ArgumentParser("BIGS scorer for Text2KGBench")
    parser.add_argument("--model_name", default="all-mpnet-base-v2", help="Sentence-BERT model.")
    parser.add_argument("--bench_name", default="webwiki", choices=["webwiki", "webnlg"])
    parser.add_argument("--split_type", default="test", help="Dataset split (e.g. train/valid/test).")
    parser.add_argument("--split", default="sentences", choices=["sentences", "chunks"])
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--output_name", default="results_text2kg", help="CSV prefix for results.")
    parser.add_argument("--hnsw_M", type=int, default=16, help="HNSW M parameter.")
    parser.add_argument("--ef_construction", type=int, default=200, help="HNSW efConstruction parameter.")
    parser.add_argument("--ef_search", type=int, default=128, help="HNSW efSearch parameter.")
    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer(args.model_name, device=device)

    categories = load_category_names(args.bench_name, args.split_type)
    output_path = Path(f"{args.output_name}_{args.bench_name}_{args.split_type}_{args.split}.csv")

    header = ["category", "n_original", "n_generated", "split_type", "model_name", "embedding_dim", "batch_size", "hnsw_M", "ef_construction", "ef_search", "encoding_time_s", "search_time_s", "total_time_s", "encode_mem_delta_mb", "search_mem_delta_mb",
        "bigs_right", "bigs_right_std", "bigs_right_median", "bigs_left", "bigs_left_std", "bigs_left_median"]
    write_header = not output_path.exists()

    with output_path.open("a", newline="") as fh:
        writer = csv.writer(fh)
        if write_header:
            writer.writerow(header)

        for cat in categories:
            print(f"\nCategory: {cat}")
            originals, generated = read_files(
                args.bench_name, args.split_type, cat, args.split
            )
            print(f"# original units:  {len(originals):,}")
            print(f"# generated texts: {len(generated):,}")

            model = SentenceTransformer(args.model_name, device=device)

            (embeds, enc_time, enc_mem) = _measure(
                generate_embeddings,
                originals, generated, model,
                batch_size=args.batch_size
            )
            orig_emb, gen_emb = embeds

            (bigs, search_time, search_mem) = _measure(
                    bigs_scores_hnsw,
                    orig_emb, gen_emb,
                    hnsw_M=args.hnsw_M,
                    ef_construction=args.ef_construction,
                    ef_search=args.ef_search,
            )
            (score_r, score_r_std, score_r_med,
            score_l, score_l_std, score_l_med) = bigs

            writer.writerow(
                [
                cat, len(originals), len(generated), args.split,
                args.model_name, model.get_sentence_embedding_dimension(), args.batch_size, args.hnsw_M, args.ef_construction, args.ef_search,
                round(enc_time, 3), round(search_time, 3), round(enc_time + search_time, 3),
                round(enc_mem, 1), round(search_mem, 1),
                round(score_l, 4), round(score_l_std, 4), round(score_l_med, 4), round(score_r, 4),
                round(score_r_std, 4), round(score_r_med, 4),
                ]
            )
            print(f"Saved results for '{cat}'")

    print(f"\nResults appended to  {output_path.resolve()}")


