from pathlib import Path
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from tqdm.auto import tqdm
import pandas as pd


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
    print("Generating embeddings...")
    orig_emb = model.encode(
        originals,
        batch_size=batch_size,
        convert_to_numpy=True,
        show_progress_bar=True,
        normalize_embeddings=True,
    )
    gen_emb = model.encode(
        generated,
        batch_size=batch_size,
        convert_to_numpy=True,
        show_progress_bar=True,
        normalize_embeddings=True,
    )
    print("Embeddings generated.")
    return orig_emb, gen_emb


def bigs_scores_hnsw(
    original_embeddings: np.ndarray,
    generated_embeddings: np.ndarray,
    hnsw_M: int = 16,
    ef_construction: int = 200,
    ef_search: int = 128,
    top_k: int = 1,
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

    # troubleshoot for m1 macbook:
    # try:
    #     faiss.omp_set_num_threads(1)
    # except Exception:
    #     pass

    print("Calculating BIGS -> ...")

    # --- Right: document -> graph ---
    idx_r = faiss.IndexHNSWFlat(d, hnsw_M)  # L2, devuelve distancias L2^2
    idx_r.hnsw.efConstruction = ef_construction
    idx_r.hnsw.efSearch = ef_search

    idx_r.add(Y)  # index sobre Y
    Dr, _ = idx_r.search(X, top_k)  # (n_orig, 1) L2^2
    right_min = Dr[:, 0] / 2.0  # cos_dist

    print("Calculating BIGS <- ...")
    # --- Left: graph -> document ---
    idx_l = faiss.IndexHNSWFlat(d, hnsw_M)
    idx_l.hnsw.efConstruction = ef_construction
    idx_l.hnsw.efSearch = ef_search

    idx_l.add(X)  # index sobre X
    Dl, _ = idx_l.search(Y, top_k)  # (n_gen, 1) L2^2
    left_min = Dl[:, 0] / 2.0  # cos_dist

    # Stats
    score_r = float(right_min.mean())
    score_r_std = float(right_min.std())
    score_r_med = float(np.median(right_min))

    score_l = float(left_min.mean())
    score_l_std = float(left_min.std())
    score_l_med = float(np.median(left_min))

    return (
        score_r,
        score_r_std,
        score_r_med,
        score_l,
        score_l_std,
        score_l_med,
    )


original_sentences_jp: list[str] = (
    Path("./data/docs_texts/sentence_split/japan_wiki.txt")
    .read_text(encoding="utf-8")
    .splitlines()
)
original_sentences_cr: list[str] = (
    Path("./data/docs_texts/sentence_split/croatia_wiki.txt")
    .read_text(encoding="utf-8")
    .splitlines()
)
generated_sentences_path = Path("./data/refinement_sentences/")


MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"
model = SentenceTransformer(MODEL_NAME)


def _load_sentences(txt_path: Path) -> list[str]:
    return [
        line.strip()
        for line in txt_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


original_lookup = {
    "croatia": original_sentences_cr,
    "japan": original_sentences_jp,
}

records: list[dict] = []

if not generated_sentences_path.exists():
    print(f"Skipping refinement BIGS: directory not found ({generated_sentences_path})")
else:
    files = sorted(generated_sentences_path.rglob("*.txt"))
    print(f"Found {len(files)} files under {generated_sentences_path}")

    for file_path in tqdm(files, desc="Processing task"):
        lower_name = file_path.stem.lower()
        if "croatia" in lower_name:
            dataset_key = "croatia"
        elif "japan" in lower_name:
            dataset_key = "japan"
        else:
            print(f"Warning: could not infer dataset from {file_path.name}, skipping.")
            continue

        if "clean" in lower_name:
            dataset_type = "clean"
        elif "full" in lower_name:
            dataset_type = "full"
        else:
            dataset_type = "unknown"

        generated_sentences = _load_sentences(file_path)
        if not generated_sentences:
            print(f"Warning: {file_path.name} is empty, skipping.")
            continue

        orig_emb, gen_emb = generate_embeddings(
            original_lookup[dataset_key],
            generated_sentences,
            model=model,
        )

        (
            score_r,
            score_r_std,
            score_r_med,
            score_l,
            score_l_std,
            score_l_med,
        ) = bigs_scores_hnsw(orig_emb, gen_emb)

        records.append(
            {
                "filename": file_path.name,
                "dataset": dataset_key,
                "type": dataset_type,
                "score_r_mean": score_r,
                "score_r_std": score_r_std,
                "score_r_median": score_r_med,
                "score_l_mean": score_l,
                "score_l_std": score_l_std,
                "score_l_median": score_l_med,
            }
        )

bigs_df = pd.DataFrame(records)
print(f"refinemnt files processed: {len(bigs_df)}")

results_path = Path("./results")
results_path.mkdir(parents=True, exist_ok=True)

if not bigs_df.empty:
    results_csv = results_path / "refinement_bigs_results.csv"
    bigs_df.to_csv(results_csv, index=False)
    print(f"Saved refinement BIGS results to {results_csv}")
else:
    print("No BIGS results to save.")
