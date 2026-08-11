#!/usr/bin/env python3
"""Build the two use-case figures whose canonical sources are LaTeX."""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path

from common import FIGURES, ROOT


SOURCES = ROOT / "figure_sources"
FIGURES_TO_BUILD = {
    "use_case_nearest_match_lorenz": 180,
    "uc_tail_selected_examples": 300,
}


def _required_program(name: str) -> str:
    path = shutil.which(name)
    if path is None:
        raise RuntimeError(f"Required program is not available: {name}")
    return path


def reproduce() -> tuple[Path, ...]:
    latexmk = _required_program("latexmk")
    pdftoppm = _required_program("pdftoppm")
    FIGURES.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []

    with tempfile.TemporaryDirectory(prefix="bigs-use-case-figures-") as raw_temp:
        build_dir = Path(raw_temp)
        for name, png_dpi in FIGURES_TO_BUILD.items():
            source = SOURCES / f"{name}.tex"
            subprocess.run(
                [
                    latexmk,
                    "-silent",
                    "-pdf",
                    "-interaction=nonstopmode",
                    "-halt-on-error",
                    f"-outdir={build_dir}",
                    source.name,
                ],
                cwd=SOURCES,
                check=True,
            )
            built_pdf = build_dir / f"{name}.pdf"
            target_pdf = FIGURES / f"{name}.pdf"
            target_png = FIGURES / f"{name}.png"
            shutil.copyfile(built_pdf, target_pdf)
            subprocess.run(
                [
                    pdftoppm,
                    "-png",
                    "-singlefile",
                    "-r",
                    str(png_dpi),
                    str(built_pdf),
                    str(target_png.with_suffix("")),
                ],
                check=True,
            )
            outputs.extend((target_pdf, target_png))

    return tuple(outputs)


def main() -> None:
    for path in reproduce():
        print(path)


if __name__ == "__main__":
    main()
