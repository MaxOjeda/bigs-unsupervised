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
nltk.download('punkt_tab')
# -----------------------------------------------------------------------------
# Regex patterns
# -----------------------------------------------------------------------------
PUNCT_REGEX = re.compile(r'[^A-Za-z0-9áéíóúñÁÉÍÓÚÑüÜ()[]{}\s]')
URL_REGEX = re.compile(r'http\S+|www\S+')


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
        "lonquen": "data/docs_texts/lonquen.txt",
        "20_docs": "data/docs_texts/20_docs.txt",
        "san_gregorio": "data/docs_texts/san_gregorio.txt",
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
    generated = load_generated_texts(f"data/textualization/{kg_text_path}.pkl")

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

# -----------------------------------------------------------------------------
if __name__ == "__main__":
    nltk.download("punkt", quiet=True)

    parser = argparse.ArgumentParser(description="Compute BIGS scores between two corpora.")
    parser.add_argument("--model_name", default="all-mpnet-base-v2", help="Sentence-BERT model.")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for encoding.")
    parser.add_argument("--kg_text_path", default="texts_gpt3_no_resolution", help="Pickle file with KG texts (no ext).")
    parser.add_argument("--case", default="lonquen", choices=["lonquen", "20_docs", "san_gregorio", "japan", "croatia"])
    parser.add_argument("--split", default="sentences", choices=["sentences", "chunks"])
    parser.add_argument("--filename", default="results.csv", help="CSV file to append the scores.")
    args = parser.parse_args()
    originals, generated = read_files(args.kg_text_path, args.case, args.split)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    model = SentenceTransformer(args.model_name, device=device)

    (score_r, score_r_std, score_r_med, score_l, score_l_std, score_l_med, elapsed) = bigs_scores(originals, generated, model, args.batch_size)

    out_path = Path(args.filename)
    header = ["kg_text_path", "n_original", "n_generated", "split_type", "model_name", "embedding_dim", "batch_size",
        "bigs_left", "bigs_left_std", "bigs_left_median", "bigs_right", "bigs_right_std", "bigs_right_median", "execution_time_s"]

    write_header = not out_path.exists()
    with out_path.open("a", newline="") as fh:
        writer = csv.writer(fh)
        if write_header:
            writer.writerow(header)

        writer.writerow(
            [
                args.kg_text_path, len(originals), len(generated), args.split,
                args.model_name, model.get_sentence_embedding_dimension(), args.batch_size,
                round(score_l, 4), round(score_l_std, 4), round(score_l_med, 4), round(score_r, 4),
                round(score_r_std, 4), round(score_r_med, 4), round(elapsed, 2),
            ]
        )
    print(f"\nResults appended to {out_path.resolve()}")
