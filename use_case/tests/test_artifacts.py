from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "results"
INPUTS = ROOT / "data" / "figure_inputs"
PROGRAMS = {"graphify", "graphrag", "lightrag", "neo4j"}
DOCUMENT_UNITS = 1_083_279


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def test_all_packaged_tables_and_json_are_readable() -> None:
    for path in sorted((ROOT / "data").rglob("*.csv")):
        assert _rows(path), path
    for path in sorted((ROOT / "data").rglob("*.json")):
        assert json.loads(path.read_text(encoding="utf-8")), path


def test_graph_and_threshold_counts_agree() -> None:
    graph_rows = {row["program"]: row for row in _rows(DATA / "graph_bigs_summary.csv")}
    coverage_rows = {
        row["program"]: row
        for row in _rows(DATA / "threshold_coverage.csv")
        if row["program"] in PROGRAMS
    }
    plot_rows = {
        row["program"]: row for row in _rows(INPUTS / "winrate_tau_coverage.csv")
    }
    assert graph_rows.keys() == coverage_rows.keys() == plot_rows.keys() == PROGRAMS

    for program in PROGRAMS:
        graph = graph_rows[program]
        coverage = coverage_rows[program]
        plotted = plot_rows[program]
        covered = int(coverage["covered_document_units"])
        uncovered = int(coverage["uncovered_document_units"])
        assert covered + uncovered == DOCUMENT_UNITS
        assert int(graph["graph_units"]) == int(coverage["graph_units"])
        assert int(graph["graph_units"]) == int(plotted["graph_units"])
        assert uncovered == int(plotted["uncovered_document_units"])
        assert np.isclose(
            100 * covered / DOCUMENT_UNITS,
            float(coverage["covered_percent"]),
            atol=1e-6,
        )
        assert np.isclose(
            float(coverage["covered_percent"]),
            float(plotted["tau_coverage_percent"]),
            atol=1e-6,
        )


def test_pairwise_table_reproduces_aggregate_win_rates() -> None:
    wins: defaultdict[str, int] = defaultdict(int)
    losses: defaultdict[str, int] = defaultdict(int)
    ties: defaultdict[str, int] = defaultdict(int)

    for row in _rows(DATA / "paired_win_loss.csv"):
        a = row["program_a"]
        b = row["program_b"]
        a_wins = int(row["a_wins"])
        b_wins = int(row["b_wins"])
        tied = int(row["ties"])
        assert a_wins + b_wins + tied == DOCUMENT_UNITS
        wins[a] += a_wins
        losses[a] += b_wins
        ties[a] += tied
        wins[b] += b_wins
        losses[b] += a_wins
        ties[b] += tied

    aggregate = {
        row["program"]: row for row in _rows(INPUTS / "winrate_tau_coverage.csv")
    }
    for program in PROGRAMS:
        row = aggregate[program]
        assert wins[program] == int(row["pairwise_wins"])
        assert losses[program] == int(row["pairwise_losses"])
        assert ties[program] == int(row["pairwise_ties"])
        denominator = wins[program] + losses[program] + ties[program]
        assert denominator == 3 * DOCUMENT_UNITS
        assert np.isclose(
            100 * wins[program] / denominator,
            float(row["pairwise_win_rate_percent"]),
            atol=1e-6,
        )


def test_threshold_calibration_arithmetic() -> None:
    [row] = _rows(DATA / "threshold_calibration.csv")
    sample = int(row["sample_size"])
    related = int(row["related"])
    unrelated = int(row["unrelated"])
    assert sample == related + unrelated
    assert np.isclose(
        100 * related / sample,
        float(row["observed_related_percent"]),
        atol=1e-4,
    )
    assert int(row["tail_population"]) == 111_743
    assert sample >= 0.10 * int(row["tail_population"])


def test_distance_summaries_account_for_every_query() -> None:
    for path in sorted((INPUTS / "distance_distributions").glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        binned = sum(int(row["count"]) for row in payload["bins"])
        out_of_support = int(payload["out_of_support_count"])
        assert binned + out_of_support == int(payload["summary"]["count"])
        if payload["direction"] == "document_to_graph":
            assert int(payload["summary"]["count"]) == DOCUMENT_UNITS


def test_lorenz_curves_are_complete_and_monotone() -> None:
    paths = sorted((INPUTS / "concentration_lorenz").glob("*.csv"))
    assert len(paths) == 8
    for path in paths:
        rows = _rows(path)
        x = np.asarray([float(row["cumulative_targets_percent"]) for row in rows])
        y = np.asarray([float(row["cumulative_assignments_percent"]) for row in rows])
        assert np.isclose(x[0], 0.0) and np.isclose(y[0], 0.0)
        assert np.isclose(x[-1], 100.0) and np.isclose(y[-1], 100.0)
        assert np.all(np.diff(x) >= 0)
        assert np.all(np.diff(y) >= 0)


def test_noise_rates_match_review_counts() -> None:
    rows = _rows(DATA / "noise_by_distance_band.csv")
    assert len(rows) == 16
    for row in rows:
        reviewed = int(row["reviewed"])
        noise = int(row["noise"])
        assert np.isclose(noise / reviewed, float(row["noise_rate"]), atol=1e-12)


def test_compact_package_uses_portable_paths() -> None:
    roots = [DATA, INPUTS, ROOT / "scripts"]
    forbidden = tuple(
        str(Path("/") / name) + "/" for name in ("home", "local_scratch", "nfs")
    )
    for root in roots:
        for path in root.rglob("*"):
            if path.is_file() and path.suffix.lower() in {".csv", ".json", ".md", ".py"}:
                text = path.read_text(encoding="utf-8").lower()
                assert not any(marker in text for marker in forbidden), path


def test_all_recorded_checksums_match() -> None:
    import hashlib

    checksum_path = ROOT / "data" / "checksums.sha256"
    for line in checksum_path.read_text(encoding="utf-8").splitlines():
        expected, relative = line.split(maxsplit=1)
        path = ROOT / relative
        assert path.is_file(), path
        assert hashlib.sha256(path.read_bytes()).hexdigest() == expected, path
