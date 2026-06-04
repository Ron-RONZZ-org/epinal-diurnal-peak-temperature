# AGENTS.md — Root Project Rules for Epinal Diurnal Peak Temperature

This is the canonical, repo-wide instruction file for AI agents working on **Epinal Diurnal Peak Temperature** — a climate data science project investigating whether the clock hour of daily maximum temperature at Épinal, France, has shifted over recent decades.

## Hierarchical Context Model

Agents **must** follow this rule:

> When working inside a directory, load the nearest `AGENTS.md` file and merge it with parent `AGENTS.md` files up to root.  
> Local rules override global rules.

Context resolution order (highest priority first):
1. `AGENTS-[module].md` in module directories — module-specific context
2. `AGENTS.md` in current working directory (if present)
3. Root `AGENTS.md` — global project rules

---

## Project Structure

```
epinal-diurnal-peak-temperature/
├── src/
│   └── epinal_peak/
│       ├── __init__.py
│       ├── config.py          # Station IDs, thresholds, analysis params (dataclass)
│       ├── acquire.py         # Data download via pooch + local fallback
│       ├── preprocess.py      # QC, UTC→local conversion, daily peak extraction
│       ├── _preprocess_qc.py  # Private QC helpers (imported by preprocess.py)
│       ├── _preprocess_schemas.py  # Private pandera schemas (imported by preprocess.py)
│       ├── analysis.py        # von Mises circular regression + sensitivities
│       ├── visualize.py       # Publication-quality figures
│       └── report.py          # Summary statistics tables
├── notebooks/
│   ├── 01-data-exploration.qmd
│   ├── 02-power-analysis.qmd
│   └── 03-sensitivity-analysis.qmd
├── tests/
│   ├── conftest.py
│   ├── test_preprocess.py
│   └── test_analysis.py
├── data/
│   ├── raw/                   # git-ignored — archived on OSF
│   ├── processed/             # git-ignored
│   └── external/              # Station metadata, reference lookups
├── results/
│   └── figures/               # Output figures (PDF + PNG)
├── logs/                      # Pipeline execution logs
├── preprint/                  # Quarto manuscript
├── pyproject.toml             # Source of truth for dependencies
├── Makefile                   # Pipeline orchestration
└── AGENTS.md                  # This file
```

---

## Language and Naming Conventions

- **Language**: Python 3.10+
- **Naming**: `snake_case` for functions/variables, `PascalCase` for classes, `UPPER_CASE` for constants
- **Imports**: Standard library first, third-party second, local package third, each group alphabetically sorted
- **Types**: Use type hints on all function signatures (`def foo(x: int) -> str:`)
- **Docstrings**: Google-style docstrings for all public functions

---

## Tech Stack

| Component | Choice |
|-----------|--------|
| Language | Python ≥ 3.10 |
| Package manager | pip via pyproject.toml |
| Data handling | pandas, numpy |
| Circular statistics | scipy (von Mises), statsmodels (GLM) |
| Data acquisition | pooch |
| Visualization | matplotlib, seaborn |
| Schema validation | pandera |
| Testing | pytest, pytest-cov |
| Pipeline | Makefile |
| Manuscript | Quarto (.qmd) |
| Time zones | zoneinfo (stdlib) |

---

## Dependency Management

This project uses `pyproject.toml` for dependency management. All dependencies are declared there.

- **Core**: pandas, numpy, scipy, statsmodels, pooch, matplotlib, seaborn, zoneinfo (stdlib)
- **Dev/Test**: pytest, pytest-cov
- **Optional**: pandera (schema validation), pingouin (supplementary correlation), quarto (manuscript render)

Install: `pip install -e ".[dev]"`

---

## Coding Guidelines

1. **Every script must be runnable as a module**: `python -m epinal_peak.<module>`
2. **Every module uses Python `logging`**: Log to both `logs/` file and stdout with format `%(asctime)s | %(levelname)s | %(name)s | %(message)s`
3. **Configuration lives in `config.py`** as validated dataclasses — never hardcode thresholds in pipeline code
4. **Pipeline stages communicate through files** (CSV/JSON in `data/` or `results/`), never through in-memory state between `make` targets
5. **Schema validation at every stage boundary**: Use `pandera.DataFrameModel` to validate column names, dtypes, and value ranges before and after each processing step
6. **Prefer vectorized pandas operations over explicit loops** for data transformation
7. **Unit test DST edge cases**: 23-hour (spring) and 25-hour (autumn) transition days must have explicit test coverage
8. **Synthetic data tests**: `analysis.py` must include a test with known synthetic circular data that verifies the regression returns the expected coefficient
9. **Tie-breaking**: Earliest peak hour is primary; latest is a sensitivity. Do NOT average tied hours (circular average of 23 and 0 is 11.5 — meaningless)
10. **Diurnal amplitude filter**: Days with ``T_max - T_min < config.min_diurnal_amplitude`` (default 2.0 °C) must be excluded before trend analysis. This removes flat-trace instrument artefacts.

---

## Documentation Standards

- **AGENTS.md** at root defines global project rules
- Module-level `AGENTS-[module].md` files define domain-specific rules for each module
- Every public function has a Google-style docstring
- Data processing decisions (thresholds, exclusion rules) are documented in `config.py` dataclass docstrings
- Pipeline stages are documented in `Makefile` comments

---

## Commit Message Format

Use [Conventional Commits](https://www.conventionalcommits.org/):
- `feat:` — New capability (e.g., "feat: add von Mises circular regression")
- `fix:` — Bug fix (e.g., "fix: DST transition day crashes on 23-hour day")
- `docs:` — Documentation (e.g., "docs: add AGENTS.md with hierarchical context model")
- `chore:` — Maintenance (e.g., "chore: pin scipy to 1.12 in pyproject.toml")
- `test:` — Testing (e.g., "test: add synthetic data test for circular regression")
- `refactor:` — Code restructuring (e.g., "refactor: extract QC checks into separate functions")

Reference GitHub issues by number: `feat(#5): implement pooch-based data acquisition`

---

## What to Avoid

- **Do NOT use `sys.path` hacks** — The `src/epinal_peak/` package is always installable via `pip install -e .`
- **Do NOT use `pytz`** — It is deprecated; use `zoneinfo` (stdlib) for all timezone handling
- **Do NOT use linear statistics on circular data** — Hour-of-day is circular (0–23 wraps); always use circular mean, circular std, and circular regression
- **Do NOT hardcode file paths** — All paths come from `config.py` or `pyproject.toml` metadata
- **Do NOT commit raw/processed data to git** — Data goes to OSF; local copies are git-ignored
- **Do NOT add dependencies without updating `pyproject.toml`** — All dependencies must be declared and pinned
- **Do NOT skip pre-registration** — Analysis must not begin before the OSF pre-registration is submitted

---

## Module-Level AGENTS Files

| Module | AGENTS File | Purpose |
|--------|-------------|---------|
| `src/epinal_peak/acquire.py` | `AGENTS-acquire.md` | Data acquisition rules |
| `src/epinal_peak/preprocess.py` | `AGENTS-preprocess.md` | QC and preprocessing rules |
| `src/epinal_peak/_preprocess_qc.py` | — | Private QC helpers (imported by `preprocess.py`) |
| `src/epinal_peak/_preprocess_schemas.py` | — | Private pandera schemas (imported by `preprocess.py`) |
| `src/epinal_peak/analysis.py` | `AGENTS-analysis.md` | Circular statistics and regression rules |
| `src/epinal_peak/visualize.py` | `AGENTS-visualize.md` | Figure generation rules |
| `notebooks/` | `AGENTS-notebooks.md` | Notebook conventions |
| `tests/` | `AGENTS-tests.md` | Testing conventions |

(Update this table as new modules are added)

---

## Dependency and Inheritance Map

```
Root AGENTS.md (global rules — this file)
    │
    ├── AGENTS-acquire.md     — data download + pooch + validation rules
    ├── AGENTS-preprocess.md  — QC thresholds, DST handling, peak extraction rules
    ├── AGENTS-analysis.md    — circular regression, sensitivity analysis rules
    ├── AGENTS-visualize.md   — figure style, export format rules
    ├── AGENTS-notebooks.md   — notebook conventions
    └── AGENTS-tests.md       — test coverage targets, synthetic data conventions
```

Local rules override global rules. Module-level files focus on domain-specific behavior, constraints, and invariants.

---

## Issue Tracking

The project plan is maintained on GitHub:
- **Issue #1**: Master research plan with sub-issue dependency map
- **Issues #2–#9**: Actionable work items (M0–M7 milestones)

All work should reference these issues in commit messages and PRs.
