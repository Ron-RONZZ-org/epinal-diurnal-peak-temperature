# AGENTS-analysis.md — Circular Statistics & Regression Rules

## Scope
`src/epinal_peak/analysis.py` — circular mean and standard deviation, von
Mises GLM regression with year as linear predictor, bootstrap confidence
intervals, and multi-variant sensitivity analysis.

## Constraints & Invariants
- **Always use circular statistics**: Hour-of-day is circular (0 wraps to 23).
  Never compute linear mean, linear std, or linear regression on raw hours.
- **Von Mises parameterization**: Model the GLM with
  `statsmodels.GLM(endog, exog, family=statsmodels.genmod.families.Binomial())`
  using a von Mises / wrapped Cauchy link or a linear-circular conversion.
  The predictor variable is **year** (centered).
- **Year centering**: Center year at the midpoint of the data range to reduce
  correlation between intercept and slope.
- **Bootstrap methodology**: Resample daily peak hours with replacement,
  refit the GLM each iteration, and extract percentile confidence intervals.
  Default: 1000 iterations, 95 % CI.
- **Sensitivity variants** (all must be implemented):
  1. Tie-breaking rule: `"latest"` instead of `"earliest"`
  2. Minimum years reduced to 20 (from 30)
  3. Outlier threshold: 3.0 MAD instead of 5.0
  4. Excluding post-2000 data only

## Data Contracts
- **Input**: CSV with columns `date`, `year`, `peak_hour`, `peak_temperature`,
  `season`.
- **Output**: JSON file with regression coefficients, bootstrap CI bounds,
  and per-variant sensitivity results.

## Edge Cases
- **Fewer than 2 unique years**: Return `None` with a logged warning.
- **All peak hours identical**: Circular variance is zero; GLM may fail to
  converge. Return `np.nan` coefficients.
- **Empty DataFrame after filtering**: Raise `ValueError`.

## References
- [Root AGENTS.md](AGENTS.md) — global coding conventions
- `config.py`: `EpinalPeakConfig.n_bootstrap`, `.bootstrap_ci_level`,
  `.min_years_for_trend`, `.sensitivity_tie_rule`
