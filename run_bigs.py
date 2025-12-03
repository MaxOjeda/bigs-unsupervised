#!/usr/bin/env python3
import sys
import os
import subprocess
from pathlib import Path
from typing import List, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
import argparse


MODELS = ["sentence-transformers/all-mpnet-base-v2", "paraphrase-multilingual-mpnet-base-v2", "sentence-transformers/all-MiniLM-L12-v2"]

SPLITS = ["sentences", "chunks"]

CONFIG = {
    # "japan": ["textualization/japan/langchain/gpt-4o-mini","textualization/japan/langchain/gpt-5", "textualization/japan/langchain/gpt-4-1","textualization/japan/langchain/gpt-5-nano","textualization/japan/original/gpt-4o-mini_full","textualization/japan/original/gpt-5_full","textualization/japan/original/gpt-4-1_full","textualization/japan/original/gpt-5-nano_full"],

    # "croatia": ["textualization/croatia/langchain/gpt-4o-mini","textualization/croatia/langchain/gpt-5", "textualization/croatia/langchain/gpt-4-1","textualization/croatia/langchain/gpt-5-nano","textualization/croatia/original/gpt-4o-mini_full","textualization/croatia/original/gpt-5_full","textualization/croatia/original/gpt-4-1_full","textualization/croatia/original/gpt-5-nano_full"],

    "neighbor_japan": ["textualization_neighbors/japan/langchain/gpt-4o-mini","textualization_neighbors/japan/langchain/gpt-5", "textualization_neighbors/japan/langchain/gpt-4-1","textualization_neighbors/japan/langchain/gpt-5-nano","textualization_neighbors/japan/original/gpt-4o-mini_full","textualization_neighbors/japan/original/gpt-5_full","textualization_neighbors/japan/original/gpt-4-1_full","textualization_neighbors/japan/original/gpt-5-nano_full"],

    "neighbor_croatia": ["textualization_neighbors/croatia/langchain/gpt-4o-mini","textualization_neighbors/croatia/langchain/gpt-5", "textualization_neighbors/croatia/langchain/gpt-4-1","textualization_neighbors/croatia/langchain/gpt-5-nano","textualization_neighbors/croatia/original/gpt-4o-mini_full","textualization_neighbors/croatia/original/gpt-5_full","textualization_neighbors/croatia/original/gpt-4-1_full","textualization_neighbors/croatia/original/gpt-5-nano_full"],
}

def results_file(case_name: str, kg_paths: List[str]) -> str:
    graph_root = kg_paths[0].split("/")[1]
    outdir = Path("results/")
    outdir.mkdir(parents=True, exist_ok=True)
    return str(outdir / f"{case_name}_vs_{graph_root}.csv")

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
        if "neighbor" in case_key:
            case_key = case_key.replace("neighbor_", "")
        out_csv = results_file(case_key, kg_paths)
        for kg_path in kg_paths:
            for split in SPLITS:
                for model in (args.only_models or MODELS):
                    tasks.append((kg_path, model, args.batch_size, split, case_key, out_csv))

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
