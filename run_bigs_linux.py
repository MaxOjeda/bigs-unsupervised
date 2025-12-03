#!/usr/bin/env python3
import sys
import os
import subprocess
from pathlib import Path
from typing import List, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
import argparse

# MODELS = [
#     "hiiamsid/sentence_similarity_spanish_es",
#     "paraphrase-multilingual-mpnet-base-v2",
#     "distiluse-base-multilingual-cased-v1",
# ]

MODELS = ["sentence-transformers/all-mpnet-base-v2"]#, "paraphrase-multilingual-mpnet-base-v2", "sentence-transformers/all-MiniLM-L12-v2"]  # For testing in English

CONFIG = {
    # "japan": ["japan/langchain/gpt-4o-mini","japan/langchain/gpt-5", "japan/langchain/gpt-4-1","japan/langchain/gpt-5-nano","japan/original/gpt-4o-mini_full","japan/original/gpt-5_full","japan/original/gpt-4-1_full","japan/original/gpt-5-nano_full"],

    # "croatia": ["croatia/langchain/gpt-4o-mini","croatia/langchain/gpt-5", "croatia/langchain/gpt-4-1","croatia/langchain/gpt-5-nano","croatia/original/gpt-4o-mini_full","croatia/original/gpt-5_full","croatia/original/gpt-4-1_full","croatia/original/gpt-5-nano_full"],

    #"japan": ["../textualization_neighbors/japan/langchain/gpt-4o-mini","../textualization_neighbors/japan/langchain/gpt-5", "../textualization_neighbors/japan/langchain/gpt-4-1","../textualization_neighbors/japan/langchain/gpt-5-nano","../textualization_neighbors/japan/original/gpt-4o-mini_full","../textualization_neighbors/japan/original/gpt-5_full","../textualization_neighbors/japan/original/gpt-4-1_full","../textualization_neighbors/japan/original/gpt-5-nano_full"],

    "croatia": ["../textualization_neighbors/croatia/langchain/gpt-4o-mini","../textualization_neighbors/croatia/langchain/gpt-5", "../textualization_neighbors/croatia/langchain/gpt-4-1","../textualization_neighbors/croatia/langchain/gpt-5-nano","../textualization_neighbors/croatia/original/gpt-4o-mini_full","../textualization_neighbors/croatia/original/gpt-5_full","../textualization_neighbors/croatia/original/gpt-4-1_full","../textualization_neighbors/croatia/original/gpt-5-nano_full"],
}

def results_file(case_name: str, kg_paths: List[str]) -> str:
    neighbor_prefix = "neighbors_" if all("neighbors" in p.split("/") for p in kg_paths) else ""
    graph_root = kg_paths[0].split("/")[0]
    outdir = Path("test_results/results_emb_models")
    outdir.mkdir(parents=True, exist_ok=True)
    return str(outdir / f"{neighbor_prefix}{case_name}_vs_{graph_root}.csv")

def launch_run(kg_path: str, model: str, batch_size: int, split: str, case: str, out_csv: str):
    cmd = [
        sys.executable, "bigs_score_faiss.py",
        "--model_name", model,
        "--batch_size", str(batch_size),
        "--kg_text_path", kg_path,
        "--split", split,
        "--case", case,
        "--filename", out_csv,
        "--hnsw_M", "16",
        "--ef_construction", "200",
        "--ef_search", "128",
    ]
    print("»", " ".join(cmd))
    # Puedes setear env aquí (ej. desactivar paralelismo de tokenizers)
    env = os.environ.copy()
    env.setdefault("TOKENIZERS_PARALLELISM", "false")
    subprocess.run(cmd, check=True, env=env)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--split", default="sentences", choices=["sentences","chunks"])
    parser.add_argument("--workers", type=int, default=1, help="Paralelismo (jobs en paralelo)")
    parser.add_argument("--only_cases", nargs="*", default=None, help="Filtra casos por nombre")
    parser.add_argument("--only_models", nargs="*", default=None, help="Filtra modelos por nombre")
    args = parser.parse_args()

    cases = {k: v for k, v in CONFIG.items() if (not args.only_cases or k in args.only_cases)}

    tasks = []
    for case_key, kg_paths in cases.items():
        base_case = case_key.replace("_neighbors", "")
        out_csv = results_file(base_case, kg_paths)
        out_csv = "results_neighbors.csv"
        for kg_path in kg_paths:
            for model in (args.only_models or MODELS):
                tasks.append((kg_path, model, args.batch_size, args.split, base_case, out_csv))

    if args.workers <= 1:
        for t in tasks:
            launch_run(*t)
    else:
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futs = [ex.submit(launch_run, *t) for t in tasks]
            for f in as_completed(futs):
                f.result()

if __name__ == "__main__":
    main()
