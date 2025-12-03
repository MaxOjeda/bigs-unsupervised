import argparse
import os
import pickle
from pathlib import Path
from typing import List

import openai
from tqdm import tqdm

# --------------------------------------------------------------------------- #
# Sentence generation                                                         #
# --------------------------------------------------------------------------- #
PROMPT_EN = (
    'Given the following knowledge graph triplet, generate a descriptive '
    'sentence in English.\n\nTriplet: "{triplet}"\n\nSentence:'
)


def generate_sentence(triplet: str, model: str):
    """Return one English sentence describing *triplet* (s | p | o)."""
    prompt = PROMPT_EN.format(triplet=triplet)
    response = openai.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=50,
        temperature=0.0,
        n=1,
        stop=["\n"],
    )
    return response.choices[0].message.content.strip()


# --------------------------------------------------------------------------- #
def triplet_files(bench: str, split: str):
    """
    Collect all triplet files for *bench* (wiki | webnlg) and *split*
    (train | valid | test).
    """
    root = Path("data/text2kgbench") / bench / split / "triplets"
    return sorted(root.glob("*.txt"))


def convert_file(txt_path: Path, out_dir: Path, bench: str, split: str, model: str):
    """
    Convert one TXT file of triplets into a pickle with sentences.
    """
    with txt_path.open(encoding="utf-8") as fh:
        triplets = [line.strip() for line in fh if line.strip()]

    sentences = [generate_sentence(t, model) for t in tqdm(triplets, desc=txt_path.stem)]

    out_path = out_dir / f"triples_{bench}_{split}_{txt_path.stem.split('_')[-1]}.pkl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("wb") as fh:
        pickle.dump(sentences, fh)

    print(f"✓  {len(sentences):,} sentences → {out_path.relative_to(Path.cwd())}")


# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    parser = argparse.ArgumentParser("Benchmark KG→text converter")
    parser.add_argument("--bench", required=True, choices=["webwiki", "webnlg"])
    parser.add_argument("--split", required=True, choices=["train", "valid", "test"])
    parser.add_argument("--model", default="gpt-4o-mini", help="OpenAI chat model")
    args = parser.parse_args()

    openai.api_key = os.getenv("OPENAI_API_KEY")
    if not openai.api_key:
        raise EnvironmentError("OPENAI_API_KEY must be set.")

    out_base = Path("data/textualization") / args.bench / args.split
    for txt_file in triplet_files(args.bench, args.split):
        convert_file(txt_file, out_base, args.bench, args.split, args.model)

    print("Done!")
