import argparse
import json
import os
import pickle
from pathlib import Path
from typing import Iterable, List

import openai
from tqdm import tqdm

# --------------------------------------------------------------------------- #
# Sentence generation using OpenAI
# --------------------------------------------------------------------------- #

def generate_sentence_from_triplet(triplet: str, lang: str, model: str) -> str:
    """
    Turn one ``s | p | o`` triplet into a descriptive sentence
    using the exact prompts from the notebook.
    """
    try:
        subject, relation, obj = (part.strip() for part in triplet.split("|"))
    except ValueError as exc:
        raise ValueError(f"Triplet '{triplet}' is not in `s | p | o` format.") from exc

    if lang == "es":
        prompt = (
            f"Dado el triplet del grafo de conocimiento a continuación, genera una oración descriptiva en español.\n\n"
            f"Triplet: \"{subject} | {relation} | {obj}\"\n\n"
            f"Oración:"
        )
    else:  # lang == "en"
        prompt = (
            f"Given the following knowledge graph triplet, generate a descriptive sentence in English.\n\n"
            f"Triplet: \"{subject} | {relation} | {obj}\"\n\n"
            f"Sentence:"
        )

    completion = openai.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=50,
        temperature=0.0,
        n=1,
        stop=["\n"],
    )
    return completion.choices[0].message.content.strip()

# --------------------------------------------------------------------------- #
def triplet_files(base: Path, folders: Iterable[str]):
    """Return every *.txt file under the requested folders."""
    files: List[Path] = []
    for folder in folders:
        dir_path = base / folder
        files.extend(sorted(dir_path.glob("*_full.txt")))
    return files


def graph_to_text(triplets: List[str], lang: str, model: str):
    """
    Convert a list of triplet strings to sentences.
    Returns the list *without* indices (only raw sentences).
    """
    sentences: List[str] = []
    for triplet in tqdm(triplets, desc="Triplets"):
        sentences.append(generate_sentence_from_triplet(triplet, lang, model))
    return sentences


# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    parser = argparse.ArgumentParser("KG-to-text converter")
    parser.add_argument("--case", required=True, help="Name of the document collection (e.g. lonquen, 20_docs, san_gregorio, japan or croatia)")
    parser.add_argument("--folders", nargs="+", default=["original"], help="Sub-folders inside data/triplet_lists/<case>/ that contain triplet *.txt files") # ["langchain", "original"]
    parser.add_argument("--lang", choices=["es", "en"], default="en", help="Language for the generated sentences")
    parser.add_argument("--model", default="gpt-4o-mini", help="Any chat-completion model available to your OpenAI account")
    args = parser.parse_args()
    openai.api_key = os.getenv("OPENAI_API_KEY")
    if not openai.api_key:
        raise EnvironmentError("OPENAI_API_KEY must be set in the environment.")

    base_triplet_dir = Path("data/triplet_lists") / args.case
    out_dir = Path("data/textualization")
    out_dir.mkdir(parents=True, exist_ok=True)

    for txt_path in triplet_files(base_triplet_dir, args.folders):
        with txt_path.open(encoding="utf-8") as fh:
            triplets = [line.strip() for line in fh]

        # Convert and save
        sentences = graph_to_text(triplets, lang=args.lang, model=args.model)
        outfile = out_dir / f"{txt_path.stem}_full.pkl"
        with outfile.open("wb") as fh:
            pickle.dump(sentences, fh)

        print(f"Saved {len(sentences):,} sentences → {outfile}")

    print("Done!")
