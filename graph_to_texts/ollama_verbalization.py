import ollama
from typing import Sequence
from pathlib import Path
import os
from tqdm.auto import tqdm

PROMPT_TEMPLATE = """You are a data-to-text model.
You will receive ONE RDF triple: (subject, predicate, object).
Write ONE short, natural English sentence that expresses ONLY that triple.
- Keep the subject and object names exactly as given (do not translate or paraphrase them).
- If the predicate is like "birthPlace" or "place of birth", say: "<subj>'s place of birth is <obj>."
- If the predicate looks verbal (e.g. "influenced", "wrote"), say: "<subj> <predicate> <obj>."
- Do NOT add extra facts.
Triple: ({subj}, {pred}, {obj})
Sentence:"""


def textualize_triples_ollama(t: Sequence[str], model: str = "llama3.2") -> str:
    """
    Turn a single triple (s, p, o) into a sentence using a local Ollama model.
    Requires:
      - `pip install ollama`
      - `ollama serve` running
      - model already pulled: `ollama pull llama3.2`
    """
    if len(t) != 3:
        raise ValueError(
            "Triple must have exactly 3 elements: (subject, predicate, object)"
        )
    s, p, o = t
    prompt = PROMPT_TEMPLATE.format(subj=s, pred=p, obj=o)

    resp = ollama.chat(
        model=model,
        messages=[
            {
                "role": "system",
                "content": "You turn RDF triples into natural sentences.",
            },
            {"role": "user", "content": prompt},
        ],
    )
    # ollama.chat(...) returns a dict with message -> content
    text = resp["message"]["content"].strip()
    # sometimes models add quotes or final period twice – quick clean
    if text.startswith('"') and text.endswith('"'):
        text = text[1:-1].strip()
    return text


save_path = Path("../data/verbalization_methods_sentences/llm_local")
save_path.mkdir(parents=True, exist_ok=True)

triples_root = Path("/data/textualization/")

TRIPLE_DELIMITER = ","


def iter_triple_files(root: Path):
    for sub in ["croatia", "japan"]:
        folder = root / sub
        if not folder.exists():
            continue
        for path in sorted(folder.rglob("*.csv")):
            if path.stem.endswith("full"):
                continue
            yield path


def read_triples(csv_path: Path):
    triples = []
    with csv_path.open(encoding="utf-8") as f:
        header = f.readline()
        columns = [col.strip() for col in header.split(TRIPLE_DELIMITER)]
        expected = {"s", "p", "o"}
        if set(columns) != expected:
            raise ValueError(f"Unexpected columns in {csv_path}: {columns}")
        for line in f:
            parts = [part.strip() for part in line.rstrip(" ").split(TRIPLE_DELIMITER)]
            if len(parts) != 3:
                continue
            triples.append(tuple(parts))
    return triples


def build_output_name(csv_path: Path) -> str:
    parts = csv_path.relative_to(triples_root).with_suffix("").parts
    return "_".join(parts) + ".txt"


if __name__ == "__main__":
    all_files = list(iter_triple_files(triples_root))
    print(f"Found {len(all_files)} triple files")

    for csv_path in tqdm(all_files, desc="Triple files"):
        triples = read_triples(csv_path)
        if not triples:
            print(f"Skipping empty file: {csv_path}")
            continue

        sentences = []
        triple_iter = tqdm(triples, desc=f"Triples for {csv_path.stem}", leave=False)
        for triple in triple_iter:
            sentence = textualize_triples_ollama(triple)
            sentences.append(sentence)

        output_name = build_output_name(csv_path)
        output_path = save_path / output_name
        output_path.write_text(" ".join(sentences) + " ", encoding="utf-8")
        print(f"Saved {len(sentences)} sentences to {output_path}")
