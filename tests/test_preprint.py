"""Tests for the Quarto manuscript compilation.

Verifies that all .qmd source files parse correctly and that the
main.qmd orchestrator compiles to HTML and PDF without errors.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PREPRINT_DIR = PROJECT_ROOT / "preprint"

# All section files that should exist and be included in main.qmd
SECTION_FILES = [
    "abstract.qmd",
    "introduction.qmd",
    "methods.qmd",
    "results.qmd",
    "discussion.qmd",
    "conclusion.qmd",
    "references.qmd",
    "supplementary.qmd",
]

# All expected ancillary files
ANCILLARY_FILES = [
    "bibliography.bib",
]


def test_quarto_cli_available() -> None:
    """Quarto CLI must be installed to compile the manuscript."""
    result = subprocess.run(
        ["quarto", "--version"],
        capture_output=True, text=True, cwd=PROJECT_ROOT,
    )
    assert result.returncode == 0, (
        "Quarto CLI not found. Install from https://quarto.org"
    )
    version = result.stdout.strip()
    parts = version.split(".")
    assert len(parts) >= 2, f"Unrecognised quarto version: {version}"
    major, minor = int(parts[0]), int(parts[1])
    assert (major, minor) >= (1, 4), (
        f"Quarto >=1.4 required, found {version}"
    )


def test_preprint_dir_exists() -> None:
    """The preprint/ directory must exist."""
    assert PREPRINT_DIR.is_dir(), f"{PREPRINT_DIR} not found"


@pytest.mark.parametrize("filename", SECTION_FILES + ANCILLARY_FILES)
def test_section_file_exists(filename: str) -> None:
    """Each required section .qmd and ancillary file must exist."""
    path = PREPRINT_DIR / filename
    assert path.is_file(), f"Missing file: {path}"
    # Files should be non-empty
    assert path.stat().st_size > 0, f"Empty file: {path}"


def test_main_qmd_exists() -> None:
    """main.qmd orchestrator must exist."""
    path = PREPRINT_DIR / "main.qmd"
    assert path.is_file(), f"Missing orchestrator: {path}"
    assert path.stat().st_size > 0


def test_main_qmd_includes_all_sections() -> None:
    """main.qmd must {{< include >}} every section file."""
    text = (PREPRINT_DIR / "main.qmd").read_text()
    for section in SECTION_FILES:
        include_stmt = "{{< include " + section + " >}}"
        assert include_stmt in text, (
            f"main.qmd missing include for {section}"
        )


def test_bibliography_is_valid_bibtex() -> None:
    """bibliography.bib should parse (basic check for @ entries)."""
    bib_path = PREPRINT_DIR / "bibliography.bib"
    text = bib_path.read_text()
    # Count @article, @conference, @techreport entries
    entries = [
        line for line in text.splitlines()
        if line.strip().startswith("@")
    ]
    assert len(entries) >= 4, (
        f"Expected >=4 BibTeX entries, found {len(entries)}"
    )


def test_bibliography_referenced_in_main() -> None:
    """main.qmd must reference the bibliography file."""
    text = (PREPRINT_DIR / "main.qmd").read_text()
    assert "bibliography:" in text or "bibliography" in text


@pytest.mark.slow
def test_quarto_compile_html() -> None:
    """Compile main.qmd to HTML and verify zero exit code."""
    result = subprocess.run(
        ["quarto", "render", "preprint/main.qmd", "--to", "html"],
        capture_output=True, text=True, cwd=PROJECT_ROOT,
        timeout=120,
    )
    if result.returncode != 0:
        print(result.stdout, file=sys.stderr)
        print(result.stderr, file=sys.stderr)
    assert result.returncode == 0, (
        f"Quarto HTML compilation failed:\n{result.stderr}"
    )


@pytest.mark.slow
def test_quarto_compile_pdf() -> None:
    """Compile main.qmd to PDF and verify zero exit code.

    PDF requires a LaTeX distribution (e.g. TeX Live / tinytex).
    If not available, quarto will exit with an error message about
    a missing engine, which we catch gracefully.
    """
    """Compile main.qmd to PDF and verify zero exit code.

    PDF requires a LaTeX distribution (e.g. TeX Live / tinytex).
    If not available, quarto will error with a missing-engine message,
    which we catch gracefully.
    """
    result = subprocess.run(
        ["quarto", "render", "preprint/main.qmd", "--to", "pdf"],
        capture_output=True, text=True, cwd=PROJECT_ROOT,
        timeout=180,
    )
    if result.returncode != 0:
        # Not a hard failure — LaTeX may not be installed in CI
        stderr_lower = result.stderr.lower()
        latex_missing = any(
            kw in stderr_lower
            for kw in ("pdflatex", "xelatex", "lualatex", "tinytex", "tlmgr")
        )
        if latex_missing:
            pytest.skip("LaTeX distribution not available — PDF not tested")
        print(result.stdout, file=sys.stderr)
        print(result.stderr, file=sys.stderr)
        pytest.fail(f"Quarto PDF compilation failed:\n{result.stderr}")
