# AGENTS-analysis.md — Circular Statistics & Regression Rules

## Scope
`src/epinal_peak/analysis.py` — circular mean and standard deviation, von
Mises GLM regression with year as linear predictor, bootstrap confidence
intervals, and multi-variant sensitivity analysis.

## Constraints & Invariants
- **Always use circular statistics**: Hour-of-day is circular (0 wraps to 23).
  Never compute linear mean, linear std, or linear regression on raw hours.
- **Von Mises regression by direct MLE**: Maximise the von Mises
  log-likelihood directly using ``scipy.optimize.minimize``
  (Fisher & Lee 1992, *Biometrics*).  The model is:

  ```math
  \theta_i = 2\pi \times \text{peak\_hour}_i / 24
  \mu_i = \beta_0 + \beta_1 \times \text{year\_centered}_i
  \log L = \sum_i \kappa \cos(\theta_i - \mu_i) - n \log(2\pi I_0(\kappa))
  ```

  where $I_0$ is the modified Bessel function
  (``scipy.special.i0``).  Enforce $\kappa > 0$ via
  $\kappa = \exp(\text{log\_kappa})$.

  **Do NOT use** ``statsmodels.GLM`` with a ``Binomial`` family —
  that family is for binary data, not circular data.  Statsmodels
  has no built-in von Mises family; custom MLE via scipy is the
  correct approach.
- **Year centering**: Center year at the midpoint of the data range to reduce
  correlation between intercept and slope.
- **Year centering**: Center year at the midpoint of the data range to reduce
  correlation between intercept and slope.
- **Bootstrap methodology**: Resample daily peak hours with replacement,
  refit the GLM each iteration, and extract percentile confidence intervals.
  Default: 1000 iterations, 95 % CI.
- **Sensitivity variants** (all must be implemented; all are pre-registered):
   1. Tie-breaking rule: `"latest"` instead of `"earliest"`
      (`config.sensitivity_tie_rule`)
   2. Minimum years reduced to 20 (from 30)
      (`config.sensitivity_min_years`)
   3. Outlier threshold: 3.0 MAD instead of 5.0
      (`config.sensitivity_mad_threshold`)
   4. Excluding post-2000 data only
      (`config.sensitivity_exclude_post_2000`)

## Primary Endpoint
- **Primary outcome**: Slope β from von Mises GLM, expressed in hours per
  decade (`config.primary_endpoint = "slope_beta_hours_per_decade"`).
- Report β, 95 % bootstrap CI, and p-value (or CI-exclusivity-of-zero).

## Data Contracts
- **Input**: CSV with columns `date`, `year`, `peak_hour`, `peak_temperature`,
  `season` (one of ``"spring"``, ``"summer"``, ``"autumn"``, ``"winter"``).
- **Output**: JSON file with regression coefficients, bootstrap CI bounds,
  and per-variant sensitivity results.

## Edge Cases
- **Fewer than 2 unique years**: Return `None` with a logged warning.
- **All peak hours identical**: Circular variance is zero; MLE may fail to
  converge. Return `np.nan` coefficients.
- **MLE convergence failure**: Fall back to **circular–linear rank
  correlation** (Mardia 1976) — a non-parametric test using all
  observations.  Report $\rho_c$ and bootstrap p-value.
- **Empty DataFrame after filtering**: Raise `ValueError`.

## References
- [Root AGENTS.md](AGENTS.md) — global coding conventions
- `config.py`: `EpinalPeakConfig.n_bootstrap`, `.bootstrap_ci_level`,
  `.min_years_for_trend`, `.sensitivity_tie_rule`,
  `.sensitivity_min_years`, `.sensitivity_mad_threshold`,
  `.sensitivity_exclude_post_2000`, `.primary_endpoint`
- Fisher, N. I. & Lee, A. J. (1992). Regression models for an angular
  response. *Biometrics* **48**, 665–677.
  doi:10.2307/2532334
- Mardia, K. V. (1976). Linear-circular correlation coefficients and
  rhythmometry. *Biometrika* **63**(3), 403–405.
  doi:10.1093/biomet/63.3.403
- Gill, J. & Hangartner, D. (2010). Circular data in political science
  and how to handle it. *Political Analysis* **18**(3), 316–336.
  doi:10.1093/pan/mpq009
