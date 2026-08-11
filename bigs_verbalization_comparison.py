from pathlib import Path
import math
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from tqdm.auto import tqdm
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from scipy import stats


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
generated_sentences_path = Path("./data/verbalization_methods_sentences")

MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"
verbalization_methods = (
    "llm",
    "llm_local",
    "prefix",
    "random",
    "concat_spo",
    "concat_sop",
    "concat_pos",
    "concat_pso",
    "concat_ops",
    "concat_osp",
)
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

for method in verbalization_methods:
    method_dir = generated_sentences_path / method
    if not method_dir.exists():
        print(f"Skipping {method}: directory not found ({method_dir})")
        continue

    txt_files = sorted(method_dir.glob("*.txt"))
    for txt_path in tqdm(txt_files, desc=f"Processing {method}"):
        lower_name = txt_path.stem.lower()
        if "croatia" in lower_name:
            dataset_key = "croatia"
        elif "japan" in lower_name:
            dataset_key = "japan"
        else:
            print(f"Warning: could not infer dataset from {txt_path.name}, skipping.")
            continue

        generated_sentences = _load_sentences(txt_path)
        if not generated_sentences:
            print(f"Warning: {txt_path.name} is empty, skipping.")
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
                "filename": txt_path.name,
                "verbalization_method": method,
                "dataset": dataset_key,
                "score_r_mean": score_r,
                "score_r_std": score_r_std,
                "score_r_median": score_r_med,
                "score_l_mean": score_l,
                "score_l_std": score_l_std,
                "score_l_median": score_l_med,
            }
        )

bigs_df = pd.DataFrame(records)
print(f"files processed: {len(bigs_df)}")

results_path = Path("./results")
results_path.mkdir(parents=True, exist_ok=True)

if not bigs_df.empty:
    results_csv = results_path / "verbalization_bigs_results.csv"
    bigs_df.to_csv(results_csv, index=False)
    print(f"Saved refinement BIGS results to {results_csv}")
else:
    print("No BIGS results to save.")

#########
# Plots #
#########

japan_df = bigs_df[bigs_df["dataset"] == "japan"]
croatia_df = bigs_df[bigs_df["dataset"] == "croatia"]


PDF_METADATA = {"Creator": "bigs_verbalization_comparison.py"}

PALETTE = {
    "teal": "#179299",
    "sky": "#04a5e5",
    "sapphire": "#209fb5",
    "blue": "#1e66f5",
    "text": "#4c4f69",
    "overlay1": "#8c8fa1",
}

TEXT = PALETTE["text"]
GRID = PALETTE["overlay1"]

METHOD_ORDER = [
    "llm",
    "llm_local",
    "prefix",
    "random",
    "concat_spo",
    "concat_sop",
    "concat_pos",
    "concat_pso",
    "concat_ops",
    "concat_osp",
]

METHOD_LABELS = {
    "llm": "LLM API",
    "llm_local": "Llama",
    "prefix": "S:P:O",
    "random": "Random",
    "concat_spo": "SPO",
    "concat_sop": "SOP",
    "concat_pos": "POS",
    "concat_pso": "PSO",
    "concat_ops": "OPS",
    "concat_osp": "OSP",
}


def setup_plot_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "mathtext.fontset": "dejavuserif",
            "font.size": 11,
            "axes.labelsize": 11,
            "axes.titlesize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.color": GRID,
            "grid.linewidth": 0.55,
            "grid.alpha": 0.22,
            "axes.axisbelow": True,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.08,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def save_pdf_png(fig: plt.Figure, out_dir: Path, filename: str) -> None:
    stem = Path(filename).with_suffix("").name
    defaults = {"bbox_inches": "tight", "pad_inches": 0.08, "facecolor": "white"}
    fig.savefig(out_dir / f"{stem}.pdf", metadata=PDF_METADATA, **defaults)
    fig.savefig(out_dir / f"{stem}.png", dpi=300, **defaults)


def verbalization_correlation_matrix(df: pd.DataFrame, target: str) -> np.ndarray:
    if target not in df.columns:
        raise KeyError(f"Missing column: {target}")

    pivot = df[df["verbalization_method"].isin(METHOD_ORDER)].pivot_table(
        index="filename",
        columns="verbalization_method",
        values=target,
        aggfunc="mean",
    )
    pivot = pivot[[method for method in METHOD_ORDER if method in pivot.columns]]

    matrix = np.full((len(METHOD_ORDER), len(METHOD_ORDER)), np.nan)
    index = {method: i for i, method in enumerate(METHOD_ORDER)}
    for method_a in pivot.columns:
        for method_b in pivot.columns:
            paired = pivot[[method_a, method_b]].dropna()
            if len(paired) < 3:
                continue
            if method_a == method_b:
                rho = 1.0
            else:
                result = stats.spearmanr(paired[method_a], paired[method_b])
                rho = float(getattr(result, "correlation", result[0]))
            matrix[index[method_a], index[method_b]] = rho

    return matrix


def plot_verbalization_heatmap(df: pd.DataFrame, target: str, out_dir: Path, filename: str) -> None:
    matrix = verbalization_correlation_matrix(df, target)
    fig, ax = plt.subplots(figsize=(7.2, 6.1), constrained_layout=True)
    cmap = LinearSegmentedColormap.from_list(
        "correlation_strength",
        [
            PALETTE["teal"],
            PALETTE["sapphire"],
            PALETTE["sky"],
            PALETTE["blue"],
        ],
    )
    im = ax.imshow(matrix, vmin=0.70, vmax=1.00, cmap=cmap)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = matrix[i, j]
            if not math.isnan(value):
                label = "1.00" if value >= 0.995 else f"{value:.2f}"
                red, green, blue, _alpha = cmap(im.norm(value))
                luminance = 0.299 * red + 0.587 * green + 0.114 * blue
                text_color = "white" if luminance < 0.58 else TEXT
                ax.text(j, i, label, ha="center", va="center", color=text_color, fontsize=8.2)

    ax.set_xticks(range(len(METHOD_ORDER)))
    ax.set_xticklabels(
        [METHOD_LABELS[method] for method in METHOD_ORDER],
        rotation=45,
        ha="right",
        rotation_mode="anchor",
    )
    ax.set_yticks(range(len(METHOD_ORDER)))
    ax.set_yticklabels([METHOD_LABELS[method] for method in METHOD_ORDER])
    ax.set_xticks(np.arange(-0.5, len(METHOD_ORDER), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(METHOD_ORDER), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=0.8)
    ax.tick_params(which="minor", bottom=False, left=False)
    ax.tick_params(axis="both", labelsize=9, length=0)
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    cbar = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.035)
    cbar.ax.tick_params(labelsize=8)
    cbar.set_label("Spearman $\\rho$", fontsize=9)
    save_pdf_png(fig, out_dir, filename)
    plt.close(fig)

    print(f"Saved plot to {out_dir / Path(filename).with_suffix('.pdf').name}")
    print(f"Saved plot to {out_dir / Path(filename).with_suffix('.png').name}")


if __name__ == "__main__":
    combined_df = pd.concat([japan_df, croatia_df], ignore_index=True)
    OUTPUT_DIR = Path("./results/verbalization")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    setup_plot_style()
    for dataset_name, dataset_df in [
        ("japan", japan_df),
        ("croatia", croatia_df),
    ]:
        for target in ["score_l_mean", "score_r_mean"]:
            plot_verbalization_heatmap(
                dataset_df,
                target,
                OUTPUT_DIR,
                f"corr_{dataset_name}_{target}.pdf",
            )

    plot_verbalization_heatmap(
        combined_df,
        "score_l_mean",
        OUTPUT_DIR,
        "verbalization_bigs_l_full.pdf",
    )
    plot_verbalization_heatmap(
        combined_df,
        "score_r_mean",
        OUTPUT_DIR,
        "verbalization_bigs_r_full.pdf",
    )
