# AGENTS-notebooks.md — Notebook Conventions

## Scope
All Quarto notebooks (`.qmd`) under `notebooks/` — exploratory analysis, power analysis, sensitivity analysis.

## Conventions

- **Header metadata**: Every notebook starts with a YAML header specifying `title`, `author`, `date`, `format`, and `execute` options.
- **Cache**: Use `execute: cache: true` for computationally expensive notebooks. Set `cache-refresh: true` to force re-run.
- **Code readability**: Chunks have descriptive labels (`#| label: short-description`). Output cells are controlled via `#| output: true/false`.
- **Output files**: Figures are saved to `results/figures/` as PDF + PNG. Tabular results go to `data/processed/`.
- **Seeds**: All stochastic simulations use an explicit RNG seed (`numpy.random.default_rng(seed=42)` or similar) for reproducibility.
- **No side effects**: Notebooks should not modify source code or committed data files.
- **Dependencies**: Notebooks import from `epinal_peak.*` modules. Inline utility functions are acceptable for notebook-specific logic.
- **Format**: Prefer HTML output (`format: html`) for interactive exploration. PDF is reserved for manuscript figures.

## References
- [Root AGENTS.md](AGENTS.md) — global coding conventions
