# ─────────────────────────────────────────────────────────────────────────
# Epinal Diurnal Peak Temperature — Pipeline Makefile
# ─────────────────────────────────────────────────────────────────────────
# Pipeline stages:
#   acquire → preprocess → analyze → visualize → report
#   test (independent of data pipeline)
#   all (full pipeline end-to-end)
# ─────────────────────────────────────────────────────────────────────────

.PHONY: install acquire preprocess analyze visualize report
.PHONY: test clean all data_dirs preprint

# ── Data directories (created on demand) ────────────────────────────────
data_dirs:
	mkdir -p data/raw data/processed data/external results/figures logs

# ── Package installation ────────────────────────────────────────────────
install:
	uv pip install -e ".[dev]"

# ── Pipeline stages ─────────────────────────────────────────────────────
acquire: data_dirs
	python -m epinal_peak.acquire

preprocess: acquire
	python -m epinal_peak.preprocess

analyze: preprocess
	python -m epinal_peak.analysis

visualize: analyze
	python -m epinal_peak.visualize

report: visualize
	python -m epinal_peak.report

# ── Testing (independent of data pipeline) ──────────────────────────────
test:
	python -m pytest tests/ -v --cov=epinal_peak --cov-report=term-missing

# ── Manuscript compilation (Quarto CLI required) ────────────────────────
# Quarto renders preprint/main.qmd to HTML and PDF.
# Install quarto from https://quarto.org
preprint:
	quarto render preprint/main.qmd --to html
	quarto render preprint/main.qmd --to pdf
	@echo "=== Manuscript compiled ==="

# ── Housekeeping ────────────────────────────────────────────────────────
clean:
	rm -rf logs/*.log results/figures/* data/processed/*

# ── Full pipeline ───────────────────────────────────────────────────────
all: install acquire preprocess analyze visualize report
	@echo "=== Full pipeline complete ==="
