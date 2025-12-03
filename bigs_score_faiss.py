import argparse
import csv
import pickle
import re
import time
from pathlib import Path
from typing import Iterable, List, Tuple

import nltk
import numpy as np
import torch
from nltk.tokenize import sent_tokenize
from scipy.spatial.distance import cdist
from sentence_transformers import SentenceTransformer
import resource
import faiss
import sys
nltk.download('punkt_tab')
# -----------------------------------------------------------------------------
# Regex patterns
# -----------------------------------------------------------------------------
PUNCT_REGEX = re.compile(r'[^A-Za-z0-9áéíóúñÁÉÍÓÚÑüÜ()[]{}\s]')
URL_REGEX = re.compile(r'http\S+|www\S+')


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


# -----------------------------------------------------------------------------
# Text utilities
# -----------------------------------------------------------------------------
def clean_text(text: str):
    """Remove URLs and most punctuation; normalise whitespace."""
    text = URL_REGEX.sub("", text)
    text = PUNCT_REGEX.sub("", text)
    return text.strip()


def sentences_from_docs(docs: Iterable[str]):
    """Split every document into sentences (NLTK)."""
    return [sent for doc in docs for sent in sent_tokenize(doc)]


def chunk_text(text: str, size: int, overlap: int):
    """
    Split *text* into word-level chunks of length *size*
    with *overlap* words shared between consecutive chunks.
    """
    tokens = text.split()
    step = size - overlap
    return [" ".join(tokens[i : i + size]) for i in range(0, len(tokens), step)]

"carabinernos fuerzas policia estado chile nacional seguridad ciudadana orden publico paco pacos delito delincuencia crimen arresto manifestacion protesta toque queda"
# -----------------------------------------------------------------------------
# Data loading
# -----------------------------------------------------------------------------
def load_generated_texts(filename: str):
    """Unpickle a list of generated texts (triples or paragraphs)."""
    with open(filename, "rb") as fh:
        data = pickle.load(fh)

    # If the file contains (entity, [triplet_texts]) pairs, flatten them
    if isinstance(data, list) and data and isinstance(data[0], tuple):
        flattened = [clean_text(triplet) for _, triplets in data for triplet in triplets]
        return flattened
    return [clean_text(doc) for doc in data]


def load_original_docs(case: str):
    """Read raw documents for a given *case*."""
    DOC_PATHS = {
        "japan": "data/docs_texts/japan_wiki.txt",
        "croatia": "data/docs_texts/croatia_wiki.txt",
    }
    try:
        path = DOC_PATHS[case]
    except KeyError as exc:
        raise ValueError(f"Unknown case '{case}'.") from exc

    with open(path, "r", encoding="utf-8") as fh:
        original = fh.readlines()
        print(f"Loaded {len(original):,} original documents from {path}.")
        return [clean_text(line) for line in original]


def read_files(kg_text_path: str, case: str, split: str = "sentences", *, chunk_size: int = 10, overlap: int = 2) -> Tuple[List[str], List[str]]:
    """Return cleaned and optionally split corpora."""
    originals = load_original_docs(case)
    generated = load_generated_texts(f"data/{kg_text_path}.pkl")

    if split == "sentences":
        originals = sentences_from_docs(originals)
    elif split == "chunks":
        originals = chunk_text(" ".join(originals), chunk_size, overlap)
    else:
        raise ValueError("--split must be 'sentences' or 'chunks'.")
    print(f"# original sentences:  {len(originals):,}")
    print(f"# generated texts: {len(generated):,}")
    return originals, generated


# -----------------------------------------------------------------------------
# Scoring
# -----------------------------------------------------------------------------
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

def generate_embeddings(originals: list[str], generated: list[str], model: SentenceTransformer, batch_size: int = 64) -> tuple[np.ndarray, np.ndarray]:
    print("Generating embeddings...")
    orig_emb = model.encode(originals, batch_size=batch_size, convert_to_numpy=True, show_progress_bar=True, normalize_embeddings=True)
    gen_emb = model.encode(generated, batch_size=batch_size, convert_to_numpy=True, show_progress_bar=True, normalize_embeddings=True)
    print("Embeddings generated.")
    return orig_emb, gen_emb


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

# -----------------------------------------------------------------------------
if __name__ == "__main__":
    nltk.download("punkt", quiet=True)

    parser = argparse.ArgumentParser(description="Compute BIGS scores between two corpora.")
    parser.add_argument("--model_name", default="all-mpnet-base-v2", help="Sentence-BERT model.")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for encoding.")
    parser.add_argument("--kg_text_path", default="texts_gpt3_no_resolution", help="Pickle file with KG texts (no ext).")
    parser.add_argument("--case", default="japan", choices=["japan", "croatia"])
    parser.add_argument("--split", default="sentences", choices=["sentences", "chunks"])
    parser.add_argument("--filename", default="results.csv", help="CSV file to append the scores.")
    parser.add_argument("--hnsw_M", type=int, default=16, help="HNSW M parameter.")
    parser.add_argument("--ef_construction", type=int, default=200, help="HNSW efConstruction parameter.")
    parser.add_argument("--ef_search", type=int, default=128, help="HNSW efSearch parameter.")

    args = parser.parse_args()
    originals, generated = read_files(args.kg_text_path, args.case, args.split)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

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

    if "neighbor" in args.kg_text_path:
        #base_name = args.kg_text_path
        args.filename = args.filename.replace("results/", "results/neighbor_")
        # args.filename = f"{args.filename}.csv"

    out_path = Path(args.filename)
    header = ["file", "n_original", "n_generated", "split_type", "model_name", "embedding_dim", "batch_size", "hnsw_M", "ef_construction", "ef_search", "encoding_time_s", "search_time_s", "total_time_s", "encode_mem_delta_mb", "search_mem_delta_mb",
        "bigs_right", "bigs_right_std", "bigs_right_median", "bigs_left", "bigs_left_std", "bigs_left_median"]

    write_header = not out_path.exists()
    with out_path.open("a", newline="") as fh:
        writer = csv.writer(fh)
        if write_header:
            writer.writerow(header)

        writer.writerow(
            [
                args.kg_text_path, len(originals), len(generated), args.split,
                args.model_name, model.get_sentence_embedding_dimension(), args.batch_size, args.hnsw_M, args.ef_construction, args.ef_search,
                round(enc_time, 3), round(search_time, 3), round(enc_time + search_time, 3),
                round(enc_mem, 1), round(search_mem, 1),
                round(score_l, 4), round(score_l_std, 4), round(score_l_med, 4), round(score_r, 4),
                round(score_r_std, 4), round(score_r_med, 4),
            ]
        )
    print(f"\nResults appended to {out_path.resolve()}")
