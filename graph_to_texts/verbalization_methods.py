import random
import openai
import re
from tqdm.auto import tqdm
import csv
from pathlib import Path
from typing import List, Tuple, Optional

Triplet = Tuple[str, str, str]


def _normalize_relation(relation: str) -> str:
    """Normalize relation tokens to human-readable lowercase phrases."""
    mapping = {
        "ISPARTOF": "is part of",
        "ISRELATEDTO": "is related to",
        "ISCONTAINEDIN": "is contained in",
        "OCCURSAT": "occurs at",
        "WASPRESENTAT": "was present at",
    }

    key = relation.strip().replace(" ", "").upper()
    if key in mapping:
        return mapping[key]

    # Fallback: split camel case / underscores, lowercase
    relation = relation.strip()
    relation = re.sub(r"[_\s]+", " ", relation)
    relation = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", relation)
    relation = re.sub(r"(?<=[A-Z])(?=[A-Z][a-z])", " ", relation)
    relation = " ".join(relation.split())
    return relation.lower()


def read_triplets_from_csv(path: str) -> List[Triplet]:
    """
    Read a CSV file with header: s,p,o
    and return a list of (subject, relation, object) triplets.
    Relation values are normalised to lowercase with words separated by spaces.
    """
    triplets: List[Triplet] = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            subject = row["s"].strip()
            relation = _normalize_relation(row["p"])
            obj = row["o"].strip()
            triplets.append((subject, relation, obj))
    return triplets


def _iter_with_progress(triplets: List[Triplet], desc: str, show_progress: bool):
    if show_progress:
        return tqdm(triplets, desc=desc, leave=False, total=len(triplets))
    return triplets


def generate_sentences_from_triplets_llm(
    triplets: List[Triplet],
    model: str,
    show_progress: bool = False,
    progress_desc: Optional[str] = None,
) -> List[str]:
    """Generate descriptive sentences via an LLM for each triplet."""
    sentences: List[str] = []
    iterator = _iter_with_progress(
        triplets, progress_desc or "LLM variant", show_progress
    )

    for subject, relation, obj in iterator:
        prompt = (
            "Given the following knowledge graph triplet, "
            "generate a descriptive sentence in English.\n\n"
            f'Triplet: "{subject} | {relation} | {obj}"\n\n'
            "Sentence:"
        )

        completion = openai.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=50,
            temperature=0.0,
            n=1,
            stop=["\n"],
        )

        sentences.append(completion.choices[0].message.content.strip())

    return sentences


def generate_sentences_from_triplets_concat(
    triplets: List[Triplet],
    show_progress: bool = False,
    progress_desc: Optional[str] = None,
) -> List[str]:
    """Generate sentences by simple concatenation: "s p o"."""
    sentences = []
    iterator = _iter_with_progress(
        triplets, progress_desc or "Concat variant", show_progress
    )
    for s, p, o in iterator:
        sentence = f"{s} {p} {o}"
        sentences.append(sentence)
    return sentences


def generate_sentences_from_triplets_prefix(
    triplets: List[Triplet],
    show_progress: bool = False,
    progress_desc: Optional[str] = None,
) -> List[str]:
    """Generate sentences with prefix format: "S:{s} | P:{p} | O:{o}"."""
    sentences = []
    iterator = _iter_with_progress(
        triplets, progress_desc or "Prefix variant", show_progress
    )
    for s, p, o in iterator:
        sentence = f"S:{s} | P:{p} | O:{o}"
        sentences.append(sentence)
    return sentences


def generate_sentences_from_triplets_random(
    triplets: List[Triplet],
    show_progress: bool = False,
    progress_desc: Optional[str] = None,
) -> List[str]:
    """Sanity-check baseline. Randomly permutes each triplet's components."""
    if not triplets:
        return []

    sentences: List[str] = []
    iterator = _iter_with_progress(
        triplets, progress_desc or "Random variant", show_progress
    )

    for s, p, o in iterator:
        original = (s, p, o)
        triplet = list(original)
        while True:
            random.shuffle(triplet)
            if tuple(triplet) != original:
                break
        sentences.append(" ".join(triplet))

    return sentences


def generate_sentences_from_triplets_concat_sop(
    triplets: List[Triplet],
    show_progress: bool = False,
    progress_desc: Optional[str] = None,
) -> List[str]:
    """Generate sentences by simple concatenation: "s o p"."""
    sentences = []
    iterator = _iter_with_progress(
        triplets, progress_desc or "Concat variant", show_progress
    )
    for s, p, o in iterator:
        sentence = f"{s} {o} {p}"
        sentences.append(sentence)
    return sentences


def generate_sentences_from_triplets_concat_pso(
    triplets: List[Triplet],
    show_progress: bool = False,
    progress_desc: Optional[str] = None,
) -> List[str]:
    """Generate sentences by simple concatenation: "p s o"."""
    sentences = []
    iterator = _iter_with_progress(
        triplets, progress_desc or "Concat variant", show_progress
    )
    for s, p, o in iterator:
        sentence = f"{p} {s} {o}"
        sentences.append(sentence)
    return sentences


def generate_sentences_from_triplets_concat_pos(
    triplets: List[Triplet],
    show_progress: bool = False,
    progress_desc: Optional[str] = None,
) -> List[str]:
    """Generate sentences by simple concatenation: "p o s"."""
    sentences = []
    iterator = _iter_with_progress(
        triplets, progress_desc or "Concat variant", show_progress
    )
    for s, p, o in iterator:
        sentence = f"{p} {o} {s}"
        sentences.append(sentence)
    return sentences


def generate_sentences_from_triplets_concat_osp(
    triplets: List[Triplet],
    show_progress: bool = False,
    progress_desc: Optional[str] = None,
) -> List[str]:
    """Generate sentences by simple concatenation: "o s p"."""
    sentences = []
    iterator = _iter_with_progress(
        triplets, progress_desc or "Concat variant", show_progress
    )
    for s, p, o in iterator:
        sentence = f"{o} {s} {p}"
        sentences.append(sentence)
    return sentences


def generate_sentences_from_triplets_concat_ops(
    triplets: List[Triplet],
    show_progress: bool = False,
    progress_desc: Optional[str] = None,
) -> List[str]:
    """Generate sentences by simple concatenation: "o p s"."""
    sentences = []
    iterator = _iter_with_progress(
        triplets, progress_desc or "Concat variant", show_progress
    )
    for s, p, o in iterator:
        sentence = f"{o} {p} {s}"
        sentences.append(sentence)
    return sentences


def generate_sentences_from_triplets_concat_spo(
    triplets: List[Triplet],
    show_progress: bool = False,
    progress_desc: Optional[str] = None,
) -> List[str]:
    """Generate sentences by simple concatenation: "s p o"."""
    sentences = []
    iterator = _iter_with_progress(
        triplets, progress_desc or "Concat variant", show_progress
    )
    for s, p, o in iterator:
        sentence = f"{s} {p} {o}"
        sentences.append(sentence)
    return sentences


TRIPLES_DIR = Path("data/triplet_list")
GENERATED_TRIPLES_DIR = Path("data/generated_sentences")
OUTPUT_VARIANTS = (
    "llm",
    "prefix",
    "random",
    "concat_spo",
    "concat_sop",
    "concat_pos",
    "concat_pso",
    "concat_osp",
    "concat_ops",
)
LLM_MODEL = "gpt-4o-mini"

GENERATED_TRIPLES_DIR.mkdir(exist_ok=True)
for variant in OUTPUT_VARIANTS:
    (GENERATED_TRIPLES_DIR / variant).mkdir(exist_ok=True)


def should_use_file(algo_name: str, csv_path: Path) -> bool:
    if csv_path.suffix.lower() != ".csv":
        return False
    if algo_name == "custom_algo":
        return csv_path.stem.endswith("clean")
    return True


def iter_triplet_files(root: Path):
    """Yield dataset, algorithm, and csv path for each triples file."""
    for dataset_dir in sorted(root.iterdir()):
        if not dataset_dir.is_dir():
            continue
        for algo_dir in sorted(dataset_dir.iterdir()):
            if not algo_dir.is_dir():
                continue
            for csv_path in sorted(algo_dir.glob("*.csv")):
                if should_use_file(algo_dir.name, csv_path):
                    yield dataset_dir.name, algo_dir.name, csv_path


def _normalise_sentence_line(sentence: str) -> str:
    """Strip and collapse whitespace so each sentence is single-line."""
    return " ".join(sentence.split())


def sentences_for_variant(
    variant: str,
    triplets: List[Triplet],
    *,
    show_progress: bool = False,
    progress_desc: str = "",
) -> List[str]:
    dispatch = {
        "llm": lambda: generate_sentences_from_triplets_llm(
            triplets,
            model=LLM_MODEL,
            show_progress=show_progress,
            progress_desc=progress_desc,
        ),
        "concat": lambda: generate_sentences_from_triplets_concat(
            triplets,
            show_progress=show_progress,
            progress_desc=progress_desc,
        ),
        "prefix": lambda: generate_sentences_from_triplets_prefix(
            triplets,
            show_progress=show_progress,
            progress_desc=progress_desc,
        ),
        "random": lambda: generate_sentences_from_triplets_random(
            triplets,
            show_progress=show_progress,
            progress_desc=progress_desc,
        ),
        "concat_sop": lambda: generate_sentences_from_triplets_concat_sop(
            triplets,
            show_progress=show_progress,
            progress_desc=progress_desc,
        ),
        "concat_pso": lambda: generate_sentences_from_triplets_concat_pso(
            triplets,
            show_progress=show_progress,
            progress_desc=progress_desc,
        ),
        "concat_pos": lambda: generate_sentences_from_triplets_concat_pos(
            triplets,
            show_progress=show_progress,
            progress_desc=progress_desc,
        ),
        "concat_osp": lambda: generate_sentences_from_triplets_concat_osp(
            triplets,
            show_progress=show_progress,
            progress_desc=progress_desc,
        ),
        "concat_ops": lambda: generate_sentences_from_triplets_concat_ops(
            triplets,
            show_progress=show_progress,
            progress_desc=progress_desc,
        ),
        "concat_spo": lambda: generate_sentences_from_triplets_concat_spo(
            triplets,
            show_progress=show_progress,
            progress_desc=progress_desc,
        ),
    }

    if variant not in dispatch:
        raise ValueError(f"Unknown variant: {variant}")
    return dispatch[variant]()


def write_generated_triplet_sentences(show_progress: bool = True):
    jobs = list(iter_triplet_files(TRIPLES_DIR))
    if not jobs:
        print("No triples found under", TRIPLES_DIR)
        return

    def notify(message: str):
        if show_progress:
            tqdm.write(message)
        else:
            print(message)

    notify(f"Processing {len(jobs)} triples files from {TRIPLES_DIR}...")
    notify(f"Available triple→sentence methods: {', '.join(OUTPUT_VARIANTS)}")

    file_iterable = (
        tqdm(jobs, desc="Triples files", unit="file") if show_progress else jobs
    )
    for dataset_name, algo_name, csv_path in file_iterable:
        triplets = read_triplets_from_csv(str(csv_path))
        output_name = f"{dataset_name}_{algo_name}_{csv_path.stem}.txt"
        notify(f"{csv_path} -> {output_name} ({len(triplets)} triplets)")

        for variant in OUTPUT_VARIANTS:
            variant_desc = f"{variant.upper()} | {csv_path.stem}"
            notify(f"  • Using method: {variant.upper()}")
            sentences = sentences_for_variant(
                variant,
                triplets,
                show_progress=show_progress,
                progress_desc=variant_desc,
            )
            clean_sentences = [_normalise_sentence_line(s) for s in sentences]
            output_path = GENERATED_TRIPLES_DIR / variant / output_name
            output_text = "\n".join(clean_sentences) + ("\n" if clean_sentences else "")
            output_path.write_text(output_text, encoding="utf-8")

        notify(f"Finished {output_name}")

    notify("Triples-to-sentences generation complete.")


if __name__ == "__main__":
    write_generated_triplet_sentences()
