import argparse
import os
import pickle
from pathlib import Path
from typing import List, Dict

import openai
from tqdm import tqdm

def generate_sentence_with_neighbors(triplet: str, neighbors: List[str], lang: str, model: str) -> str:
    """
    Generate a sentence for a single triplet, providing all triplets with the same subject as context.
    """
    try:
        subject, relation, obj = (part.strip() for part in triplet.split("|"))
    except ValueError as exc:
        raise ValueError(f"Triplet '{triplet}' is not in `s | p | o` format.") from exc

    if lang == "es":
        prompt = (
            f"Dado el siguiente triplete del grafo de conocimiento, genera una oración descriptiva en español. "
            f"Considera también el contexto de los otros tripletes que comparten el mismo sujeto.\n\n"
            f"Triplete objetivo: \"{subject} | {relation} | {obj}\"\n"
            f"Tripletes con el mismo sujeto:\n"
            f"{chr(10).join(neighbors)}\n\n"
            f"Oración:"
        )
    else:
        prompt = (
            f"Given the following knowledge graph triplet, generate a descriptive sentence in English. "
            f"Also consider the context of the other triplets sharing the same subject.\n\n"
            f"Target triplet: \"{subject} | {relation} | {obj}\"\n"
            f"Triplets with the same subject:\n"
            f"{chr(10).join(neighbors)}\n\n"
            f"Sentence:"
        )

    completion = openai.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=80,
        temperature=0.0,
        n=1,
        stop=["\n"],
    )
    return completion.choices[0].message.content.strip()

def txt_files(base: Path, folders: List[str]):
    """Return every *.txt file under the requested folders."""
    files: List[Path] = []
    for folder in folders:
        dir_path = base / folder
        files.extend(sorted(dir_path.glob("*.txt")))
    return files

def group_triplets_by_subject(triplets: List[str]) -> Dict[str, List[str]]:
    """Group triplets by their subject."""
    groups: Dict[str, List[str]] = {}
    for t in triplets:
        try:
            subject = t.split("|")[0].strip()
        except Exception:
            continue
        groups.setdefault(subject, []).append(t)
    return groups

def graph_to_text_with_neighbors(triplets: List[str], lang: str, model: str) -> List[str]:
    """
    For each triplet, generate a sentence using all triplets with the same subject as context.
    Returns a list of sentences (one per input triplet).
    """
    subj_groups = group_triplets_by_subject(triplets)
    results = []
    for triplet in tqdm(triplets, desc="Triplets"):
        subject = triplet.split("|")[0].strip()
        neighbors = subj_groups[subject]
        sent = generate_sentence_with_neighbors(triplet, neighbors, lang, model)
        results.append(sent)
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser("KG-to-text with neighbor context")
    parser.add_argument("--case", required=True, help="Name of the document collection (e.g. lonquen, 20_docs, san_gregorio, japan or croatia)")
    parser.add_argument("--folders", nargs="+", default=["original"], help="Sub-folders inside data/triplet_lists/<case>/ that contain triplet *.txt files")
    parser.add_argument("--lang", choices=["es", "en"], default="en", help="Language for the generated sentences")
    parser.add_argument("--model", default="gpt-4o-mini", help="Any chat-completion model available to your OpenAI account")
    args = parser.parse_args()
    openai.api_key = os.getenv("OPENAI_API_KEY")
    if not openai.api_key:
        raise EnvironmentError("OPENAI_API_KEY must be set in the environment.")

    base_triplet_dir = Path("data/triplet_lists") / args.case
    out_dir = Path("data/textualization_neighbors")
    out_dir.mkdir(parents=True, exist_ok=True)

    for txt_path in txt_files(base_triplet_dir, args.folders):
        with txt_path.open(encoding="utf-8") as fh:
            triplets = [line.strip() for line in fh if line.strip()]

        # Convert and save
        sentences = graph_to_text_with_neighbors(triplets, lang=args.lang, model=args.model)
        outfile = out_dir / f"{txt_path.stem}.pkl"
        with outfile.open("wb") as fh:
            pickle.dump(sentences, fh)

        print(f"Saved {len(sentences):,} sentences → {outfile}")

    print("Done!")
