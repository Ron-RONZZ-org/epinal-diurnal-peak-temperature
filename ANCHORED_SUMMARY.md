# Anchored Summary

## Goal
Restructure pipeline and manuscript so seasonal stratification is the primary analysis, pooled is supplementary, with per-season sensitivity variants.

## Constraints & Preferences
- `main` branch must not be worked on directly — feature branch required.
- Monolith files >500 lines forbidden; already split in prior work.
- Pre-registration was **never submitted** — no constraint on analysis design.
- Seasonal primary is justified by significant interaction test (χ²=13.71, p=0.0033).
- Per-season sensitivity variants (6 variants × 4 seasons) must be added.
- Manuscript narrative must refocus: four distinct seasons, not one pooled result.
- All code to Python3 conventions: English names/comments, type hints, no `sys.path` hacks.

## Progress

### Done
- Interaction test implemented (LR test, full/reduced von Mises GLM, BH FDR).
- `analysis.py` split into `_analysis_regression.py`, `_analysis_interaction.py`, `_analysis_weighted.py`, slimmer `analysis.py`.
- 17 automated tests — all pass (except pre-existing data-dependent timeout).
- User-simulation testing on real data: interaction significant (χ²=13.71, p=0.0033); winter BH-rejected (−0.19 h/decade, p=0.0015).
- Merged `feat/interaction-test` → `main`, pushed.
- Created and closed GitHub issue #11 documenting the interaction test.
- `AGENTS.md` updated: removed pre-registration constraint, added hierarchical inference rules, added issue #11 reference, added `_analysis_weighted.py` to module table.
- `README.md` updated with interaction test methodology.
- Figure placement fixed (`fig-pos: "htb!"` in `main.qmd` PDF format).
- Season-blind narrative identified as problematic by user (Simpson's paradox: winter/autumn cancel in pooled estimate).
- **Weighted regression extracted**: `_analysis_variants.py` dropped from 535→328 lines; new `_analysis_weighted.py` at 239 lines. All imports updated.

### In Progress
- Adding `seasonal_sensitivity_analysis()` to `_analysis_variants.py`.
- Reordering `analysis.py` `main()`: interaction test (diagnostic) → seasonal (primary) → seasonal sensitivity → pooled (supplementary) → pooled sensitivity → other supplementary.
- Updating `_build_output_json` to reflect new nesting: `interaction`, `seasonal`, `seasonal_sensitivity`, `pooled`, `sensitivity`, `supplementary`.

### Blocked
- None.

## Key Decisions
- **Seasonal becomes primary analysis** — pooled is supplementary, presented as "average of opposing signals." The interaction test (p=0.0033) justifies this stratification.
- **Sensitivity variants run per season** — 6 variants × 4 seasons = 24 models, testing robustness of each seasonal estimate.
- **No pre-registration constraint** — the draft was never submitted; analysis design is free to evolve.
- **Branch changed** from `feat/interaction-test` (merged) to `refactor/seasonal-primary-analysis` (current).
- **`_analysis_weighted.py` extracted** from `_analysis_variants.py` to keep all modules <500 lines.

## Next Steps
1. Finish `seasonal_sensitivity_analysis()` and reordered `main()`.
2. Add automated tests for seasonal sensitivity.
3. Run all tests, verify pass.
4. User-simulation: run pipeline on real data, verify new JSON structure.
5. Restructure manuscript sections (results.qmd, methods.qmd, abstract.qmd, conclusion.qmd).
6. Recompile PDF and HTML.
7. Commit, merge `refactor/seasonal-primary-analysis` → `main`, push.
8. Create/close GitHub issue for structural change; update `AGENTS.md`, `README.md`.

## Critical Context
- **Simpson's paradox**: Pooled estimate (−0.028 h/decade) hides winter (−0.19) and autumn (+0.09) trends cancelling. User found this confusing.
- Interaction LR test: χ²=13.71, df=3, p=0.0033 → significant. Justifies per-season analysis as primary.
- Per-season slopes from interaction model: winter −0.19 (p=0.0015, BH-rejected), spring −0.07 (p=0.17), autumn +0.09 (p=0.10), summer +0.02 (p=0.68).
- The existing `_analysis_variants.sensitivity_analysis()` handles the pooled 8 variants. New `seasonal_sensitivity_analysis()` wraps `_run_variant()` per season with 6 variants each.
- `main()` reordering: interaction test first (diagnostic), seasonal second (primary), pooled last (supplementary).
- Output JSON restructured: `{interaction, seasonal, seasonal_sensitivity, pooled, sensitivity, supplementary}`.
- `_analysis_weighted.py` created with `temperature_weighted_regression()`, `_bootstrap_weighted_engine()`, `_run_weighted()`.

## Relevant Files
- `src/epinal_peak/_analysis_variants.py`: adding `seasonal_sensitivity_analysis()` after `sensitivity_analysis()`.
- `src/epinal_peak/_analysis_weighted.py`: extracted weighted regression functions.
- `src/epinal_peak/analysis.py`: reordering `main()`, updating `_build_output_json()` and import paths.
- `tests/test_analysis.py`: needs new test for `seasonal_sensitivity_analysis()`.
- `preprint/results.qmd`, `methods.qmd`, `abstract.qmd`, `conclusion.qmd`: need full restructure for seasonal-primary narrative.
- `AGENTS.md`, `README.md`: need update after restructure is complete.
