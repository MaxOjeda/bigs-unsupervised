import subprocess
from pathlib import Path
from typing import List, Tuple

# MODELS = [
#     "hiiamsid/sentence_similarity_spanish_es",
#     "paraphrase-multilingual-mpnet-base-v2",
#     "distiluse-base-multilingual-cased-v1",
# ]
MODELS = ["sentence-transformers/all-mpnet-base-v2"]  # For testing in English

BATCH_SIZE = 64         
SPLIT = "sentences" # or "chunks"

CONFIG = {
    # -------- case ---------  #
    # "lonquen": ["lonquen/langchain/gpt-4o-mini","lonquen/langchain/gpt-4o","lonquen/langchain/o3-mini","lonquen/original/gpt-4o-mini","lonquen/original/gpt-4o",
    #             "lonquen/original/o3-mini","lonquen/original/gpt-3","lonquen/original_res/gpt-4o-mini","lonquen/original_res/gpt-4o","lonquen/original_res/o3-mini","lonquen/original_res/gpt-3"],

    # "20_docs": ["20_docs/langchain/gpt-4o-mini","20_docs/langchain/gpt-4o","20_docs/langchain/o3-mini","20_docs/original/gpt-4o-mini","20_docs/original/gpt-4o",
    #             "20_docs/original/o3-mini","20_docs/original/gpt-3","20_docs/original_res/gpt-4o-mini","20_docs/original_res/gpt-4o","20_docs/original_res/o3-mini","20_docs/original_res/gpt-3"],

    # "san_gregorio": ["san_gregorio/langchain/gpt-4o-mini","san_gregorio/langchain/gpt-4o","san_gregorio/langchain/o3-mini","san_gregorio/original/gpt-4o-mini","san_gregorio/original/gpt-4o",
    #                  "san_gregorio/original/o3-mini","san_gregorio/original/gpt-3.5-turbo","san_gregorio/original_res/gpt-4o-mini","san_gregorio/original_res/gpt-4o","san_gregorio/original_res/o3-mini","san_gregorio/original_res/gpt-3.5-turbo"],

    "japan": ["japan/langchain/gpt-4o-mini","japan/langchain/gpt-5","japan/original/gpt-4o-mini","japan/original/gpt-5"],

    "croatia": ["croatia/langchain/gpt-4o-mini","croatia/langchain/gpt-5","croatia/original/gpt-4o-mini","croatia/original/gpt-5"],

    # --- neighbour variants --- #
    # "lonquen_neighbors": ["lonquen/neighbors/langchain/gpt-4o-mini","lonquen/neighbors/langchain/gpt-4o","lonquen/neighbors/langchain/o3-mini","lonquen/neighbors/original/gpt-4o-mini","lonquen/neighbors/original/gpt-4o",
    #                       "lonquen/neighbors/original/o3-mini","lonquen/neighbors/original/gpt-3.5-turbo","lonquen/neighbors/original_res/gpt-4o-mini","lonquen/neighbors/original_res/gpt-4o","lonquen/neighbors/original_res/o3-mini","lonquen/neighbors/original/gpt-3.5-turbo"],

    # "20_docs_neighbors": ["20_docs/neighbors/langchain/gpt-4o-mini","20_docs/neighbors/langchain/gpt-4o","20_docs/neighbors/langchain/o3-mini","20_docs/neighbors/original/gpt-4o-mini","20_docs/neighbors/original/gpt-4o",
    #                       "20_docs/neighbors/original/o3-mini","20_docs/neighbors/original/gpt-3.5-turbo","20_docs/neighbors/original_res/gpt-4o-mini","20_docs/neighbors/original_res/gpt-4o","20_docs/neighbors/original_res/o3-mini","20_docs/neighbors/original/gpt-3.5-turbo"],

    # "san_gregorio_neighbors": ["san_gregorio/neighbors/langchain/gpt-4o-mini","san_gregorio/neighbors/langchain/gpt-4o","san_gregorio/neighbors/langchain/o3-mini","san_gregorio/neighbors/original/gpt-4o-mini","san_gregorio/neighbors/original/gpt-4o",
    #                            "san_gregorio/neighbors/original/o3-mini","san_gregorio/neighbors/original/gpt-3.5-turbo","san_gregorio/neighbors/original_res/gpt-4o-mini","san_gregorio/neighbors/original_res/gpt-4o","san_gregorio/neighbors/original_res/o3-mini","san_gregorio/neighbors/original/gpt-3.5-turbo"]
}

# ---------------------------------------------------------------------------- #
def results_file(case_name: str, kg_paths: List[str]):
    """
    Produce the output CSV name.

    * neighbours_* prefix is added iff *every* path contains 'neighbors'.
    * otherwise the file is simply <case>_vs_<graph>.csv where <graph> is the
      first folder component of the KG path.
    """
    neighbor_prefix = "neighbors_" if all("neighbors" in p.split("/") for p in kg_paths) else ""
    graph_root = kg_paths[0].split("/")[0]
    Path("test_results/results_emb_models").mkdir(parents=True, exist_ok=True)
    return f"test_results/results_emb_models/{neighbor_prefix}{case_name}_vs_{graph_root}.csv"


def launch_run(kg_path: str, model: str, batch_size: int, split: str, case: str, out_csv: str):
    cmd = ["python", "bigs_score.py",
        "--model_name", model,
        "--batch_size", str(batch_size),
        "--kg_text_path", kg_path,
        "--split", split,
        "--case", case,
        "--filename", out_csv,
    ]
    print("»", " ".join(cmd))
    subprocess.run(cmd, check=True)

# ---------------------------------------------------------------------------- #
if __name__ == "__main__":
    for case_key, kg_paths in CONFIG.items():
        base_case = case_key.replace("_neighbors", "")
        out_csv = results_file(base_case, kg_paths)

        for kg_path in kg_paths:
            for model in MODELS:
                launch_run(
                    kg_path=kg_path,
                    model=model,
                    batch_size=BATCH_SIZE,
                    split=SPLIT,
                    case=base_case,
                    out_csv=out_csv,
                )