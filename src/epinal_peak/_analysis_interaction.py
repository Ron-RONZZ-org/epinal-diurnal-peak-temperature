"""Year × season interaction test for von Mises regression.

Tests whether the year-slope (trend in peak hour of daily maximum
temperature) differs across meteorological seasons.

Workflow
--------
1. Fit a **reduced model** (6 params): season-specific intercepts +
   a **common** year slope.
2. Fit a **full model** (9 params): season-specific intercepts +
   **season-specific** year slopes.
3. Likelihood-ratio test with 3 degrees of freedom.
4. If significant (p < α), compute per-season Wald p-values from the
   full-model Hessian and apply Benjamini-Hochberg FDR correction.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import i0
from scipy.stats import chi2, norm

_logger = logging.getLogger(__name__)

_HOURS_TO_RAD = 2.0 * np.pi / 24.0
_RAD_TO_HOURS = 24.0 / (2.0 * np.pi)

_SEASON_ORDER = ["spring", "summer", "autumn", "winter"]
"""Canonical season order — must match values in the ``season`` column."""

_INTERACTION_ALPHA = 0.05
"""Gatekeeper significance level for the interaction LR test."""

_FDR_ALPHA = 0.05
"""Target FDR for Benjamini-Hochberg correction on per-season p-values."""

# ── Low-level helpers ─────────────────────────────────────────────────


def _hours_to_radians(hours: np.ndarray) -> np.ndarray:
    """Convert hour-of-day values to radians."""
    return hours * _HOURS_TO_RAD


def _radians_to_hours(radians: np.ndarray | float) -> np.ndarray | float:
    """Convert radians to hour-of-day values, normalised to [0, 24)."""
    return (radians * _RAD_TO_HOURS) % 24.0


def _build_design_matrix(
    season: pd.Series,
    year_centered: np.ndarray,
    *,
    include_interaction: bool,
) -> np.ndarray:
    """Build the von Mises GLM design matrix.

    Reduced model (``include_interaction=False``):
        [I(spring), I(summer), I(autumn), I(winter), year_centered]
    → 5 columns  (→ 6 params with log_kappa).

    Full model (``include_interaction=True``):
        [I(spring), I(summer), I(autumn), I(winter),
         I(spring)×yr, I(summer)×yr, I(autumn)×yr, I(winter)×yr]
    → 8 columns (→ 9 params with log_kappa).

    Args:
        season: Categorical season labels matched to ``_SEASON_ORDER``.
        year_centered: Year values centred at the sample mean.
        include_interaction: If True, add season × year columns.

    Returns:
        Design matrix, shape ``(n, n_cols)``.
    """
    n = len(season)
    cols: list[np.ndarray] = []

    # Season dummies (one-hot, no reference category)
    for s in _SEASON_ORDER:
        cols.append((season == s).astype(float).values.reshape(-1, 1))

    if include_interaction:
        for i in range(len(_SEASON_ORDER)):
            cols.append(
                (cols[i].flatten() * year_centered).reshape(-1, 1)
            )
    else:
        cols.append(year_centered.reshape(-1, 1))

    return np.column_stack(cols)


def _neg_log_likelihood_design(
    params: np.ndarray,
    theta: np.ndarray,
    X: np.ndarray,
) -> float:
    """Negative log-likelihood for von Mises regression with design matrix.

    Args:
        params: Coefficient vector ``[β_0, …, β_{{p-1}}, log_κ]``.
        theta: Circular response in radians, shape ``(n,)``.
        X: Design matrix, shape ``(n, p)``.

    Returns:
        Negative log-likelihood.
    """
    beta = params[:-1]
    log_kappa = params[-1]
    kappa = np.exp(log_kappa)
    mu = X @ beta
    n = len(theta)
    ll = np.sum(kappa * np.cos(theta - mu)) - n * np.log(2.0 * np.pi * i0(kappa))
    return float(-ll)


# ── Numerical Hessian ────────────────────────────────────────────────


def _numerical_hessian(
    f: callable, x: np.ndarray, args: tuple = (), eps: float = 1e-5
) -> np.ndarray:
    """Finite-difference Hessian of a scalar function at *x*.

    Uses central differences.  Requires ``f : (x, *args) → float``.

    Args:
        f: Scalar function.
        x: Point at which to evaluate the Hessian.
        args: Additional arguments passed to *f*.
        eps: Step size for finite differences.

    Returns:
        Hessian matrix, shape ``(len(x), len(x))``.
    """
    n = len(x)
    H = np.zeros((n, n))
    f0 = f(x, *args)
    for i in range(n):
        ei = np.zeros(n)
        ei[i] = eps
        fp = f(x + ei, *args)
        fm = f(x - ei, *args)
        H[i, i] = (fp - 2.0 * f0 + fm) / (eps * eps)
        for j in range(i + 1, n):
            ej = np.zeros(n)
            ej[j] = eps
            fpp = f(x + ei + ej, *args)
            fpm = f(x + ei - ej, *args)
            fmp = f(x - ei + ej, *args)
            fmm = f(x - ei - ej, *args)
            H[i, j] = (fpp - fpm - fmp + fmm) / (4.0 * eps * eps)
            H[j, i] = H[i, j]
    return H


# ── Initialisation helpers ───────────────────────────────────────────


def _init_params(
    theta: np.ndarray, year_centered: np.ndarray, season: pd.Series
) -> dict[str, np.ndarray]:
    """Compute smart starting values for reduced and full models.

    Season intercepts are initialised to the circular mean of each
    season (in radians).  Slopes are initialised to zero.  log_kappa
    is estimated from the pooled data.

    Returns:
        Dict with ``reduced`` and ``full`` initial parameter vectors.
    """
    n_seasons = len(_SEASON_ORDER)

    # Per-season circular mean (intercept init)
    alpha_init = np.zeros(n_seasons)
    for i, s in enumerate(_SEASON_ORDER):
        mask = season == s
        if mask.any():
            h = theta[mask]
            s_sum = np.sin(h).sum()
            c_sum = np.cos(h).sum()
            alpha_init[i] = np.arctan2(s_sum, c_sum)
        else:
            alpha_init[i] = 0.0

    # Pooled kappa from mean resultant length
    sin_sum = np.sin(theta).sum()
    cos_sum = np.cos(theta).sum()
    n = float(len(theta))
    r = np.sqrt(sin_sum**2 + cos_sum**2) / n
    if r < 0.99:
        kappa_init = max(r * (2.0 - r**2) / (1.0 - r**2), 0.01)
    else:
        kappa_init = 100.0
    log_kappa_init = np.log(kappa_init)

    # Reduced: [α_s, α_s, α_s, α_s, β_common, log_κ]
    reduced = np.concatenate([alpha_init, [0.0], [log_kappa_init]])

    # Full: [α_s, α_s, α_s, α_s, γ_s, γ_s, γ_s, γ_s, log_κ]
    full = np.concatenate([alpha_init, np.zeros(n_seasons), [log_kappa_init]])

    return {"reduced": reduced, "full": full}


# ── Fit helpers ──────────────────────────────────────────────────────


def _fit_vm_design(
    theta: np.ndarray,
    X: np.ndarray,
    x0: np.ndarray,
) -> dict[str, Any]:
    """Fit a von Mises regression with a design matrix.

    Args:
        theta: Circular response in radians.
        X: Design matrix, shape ``(n, p)``.
        x0: Initial parameter vector (length ``p + 1`` for log_kappa).

    Returns:
        Dict with ``params``, ``log_likelihood``, ``converged``, ``status``.
    """
    result = minimize(
        _neg_log_likelihood_design,
        x0=x0,
        args=(theta, X),
        method="Nelder-Mead",
        options={"maxiter": 5000, "xatol": 1e-8, "fatol": 1e-8},
    )
    return {
        "params": result.x,
        "log_likelihood": float(-result.fun),
        "converged": bool(result.success),
        "status": "ok" if result.success else "mle_did_not_converge",
    }


# ── Public API ───────────────────────────────────────────────────────


def interaction_lr_test(df: pd.DataFrame) -> dict[str, Any]:
    """Likelihood-ratio test for year × season interaction.

    Compares a reduced von Mises GLM (season intercepts + common year
    slope) to a full model (season intercepts + season-specific slopes).

    Args:
        df: DataFrame with ``year``, ``peak_hour``, and ``season`` columns.
            The ``season`` column must use values in ``_SEASON_ORDER``.

    Returns:
        Dict with keys:

        - ``lr_stat``: Likelihood-ratio :math:`\\chi^2` statistic.
        - ``df``: Degrees of freedom (3).
        - ``p_value``: p-value from :math:`\\chi^2_3`.
        - ``significant``: ``True`` if ``p_value < 0.05``.
        - ``full_model``: Full-model fit results.
        - ``reduced_model``: Reduced-model fit results.
        - ``per_season``: Per-season estimates with Wald p-values and
          FDR flags (only present when ``significant=True``).
        - ``n_obs``: Number of valid observations used.
        - ``status``: ``"ok"`` or error description.
    """
    # ── Validate ─────────────────────────────────────────────────
    required = {"year", "peak_hour", "season"}
    missing = required - set(df.columns)
    if missing:
        return {
            "status": f"missing_columns: {missing}",
            "lr_stat": None,
            "df": None,
            "p_value": None,
            "significant": None,
        }

    peak_hour = df["peak_hour"].values.astype(float)
    year = df["year"].values.astype(float)
    season = df["season"]

    valid = ~(np.isnan(peak_hour) | np.isnan(year))
    if not valid.any():
        return {
            "status": "no_valid_observations",
            "lr_stat": None,
            "df": None,
            "p_value": None,
            "significant": None,
        }

    theta = _hours_to_radians(peak_hour[valid])
    year_subset = year[valid]
    season_subset = season.iloc[valid].reset_index(drop=True)
    year_center = float(np.mean(year_subset))
    year_centered = year_subset - year_center
    n_obs = len(theta)

    # ── Check season coverage ────────────────────────────────────
    missing_seasons = [s for s in _SEASON_ORDER if not (season_subset == s).any()]
    if missing_seasons:
        return {
            "status": f"missing_seasons: {missing_seasons}",
            "lr_stat": None,
            "df": None,
            "p_value": None,
            "significant": None,
        }

    # ── Build design matrices ────────────────────────────────────
    X_reduced = _build_design_matrix(
        season_subset, year_centered, include_interaction=False
    )
    X_full = _build_design_matrix(
        season_subset, year_centered, include_interaction=True
    )

    # ── Smart initial values ─────────────────────────────────────
    inits = _init_params(theta, year_centered, season_subset)

    # ── Reduced model ────────────────────────────────────────────
    reduced_fit = _fit_vm_design(theta, X_reduced, inits["reduced"])
    if reduced_fit["status"] != "ok":
        return {
            "status": f"reduced_model_{reduced_fit['status']}",
            "lr_stat": None,
            "df": None,
            "p_value": None,
            "significant": None,
        }

    # ── Full model ───────────────────────────────────────────────
    full_fit = _fit_vm_design(theta, X_full, inits["full"])
    if full_fit["status"] != "ok":
        return {
            "status": f"full_model_{full_fit['status']}",
            "lr_stat": None,
            "df": None,
            "p_value": None,
            "significant": None,
        }

    # ── Likelihood-ratio test ────────────────────────────────────
    ll_reduced = reduced_fit["log_likelihood"]
    ll_full = full_fit["log_likelihood"]
    lr_stat = 2.0 * (ll_full - ll_reduced)
    df_test = 3  # 4 season slopes - 1 common slope = 3
    p_value = 1.0 - chi2.cdf(lr_stat, df_test)
    significant = bool(p_value < _INTERACTION_ALPHA)

    # ── Assemble result ──────────────────────────────────────────
    result: dict[str, Any] = {
        "lr_stat": float(lr_stat),
        "df": df_test,
        "p_value": float(p_value),
        "significant": significant,
        "full_model": {
            "params": full_fit["params"].tolist(),
            "log_likelihood": ll_full,
            "converged": full_fit["converged"],
        },
        "reduced_model": {
            "params": reduced_fit["params"].tolist(),
            "log_likelihood": ll_reduced,
            "converged": reduced_fit["converged"],
        },
        "n_obs": n_obs,
        "year_center": float(year_center),
        "status": "ok",
    }

    # ── Per-season inference (only when significant) ─────────────
    if significant:
        _logger.info(
            "Interaction LR test significant (χ²=%.2f, df=%d, p=%.4f) "
            "— computing per-season Wald p-values",
            lr_stat,
            df_test,
            p_value,
        )
        result["per_season"] = _per_season_inference(
            theta, X_full, full_fit["params"], n_obs
        )

    return result


def _per_season_inference(
    theta: np.ndarray,
    X_full: np.ndarray,
    params: np.ndarray,
    n_obs: int,
) -> dict[str, Any]:
    """Compute per-season Wald p-values and BH-adjusted significance.

    The full model has 9 params:
        params[0:4] = season intercepts [spring … winter]
        params[4:8] = season slopes     [spring … winter]
        params[8]   = log_kappa

    We extract the 4 slope estimates, compute standard errors from the
    observed Fisher information (Hessian of NLL), then derive Wald
    p-values and apply Benjamini-Hochberg FDR.

    Args:
        theta: Circular response in radians.
        X_full: Full-model design matrix.
        params: MLE parameter vector (length 9).
        n_obs: Number of observations (for reference).

    Returns:
        Dict with ``slopes`` (per-season details), ``fdr_alpha``, and
        ``n_seasons_rejected``.
    """
    n_seasons = len(_SEASON_ORDER)

    # ── Hessian → covariance ─────────────────────────────────────
    hess = _numerical_hessian(
        _neg_log_likelihood_design, params, args=(theta, X_full)
    )
    try:
        cov = np.linalg.inv(hess)
    except np.linalg.LinAlgError:
        cov = None

    # ── Per-season slope table ───────────────────────────────────
    slopes: list[dict[str, Any]] = []
    for i, season_name in enumerate(_SEASON_ORDER):
        gamma_idx = n_seasons + i  # params[4]…params[7]
        gamma_est = float(params[gamma_idx])
        gamma_hpd = gamma_est * _RAD_TO_HOURS * 10.0  # hours/decade

        se = float(np.sqrt(cov[gamma_idx, gamma_idx])) if cov is not None else np.nan
        if se > 0 and not np.isnan(se):
            wald_z = gamma_est / se
            p = 2.0 * norm.sf(abs(wald_z))
        else:
            p = 1.0

        slopes.append({
            "season": season_name,
            "slope_rad_per_year": gamma_est,
            "slope_hours_per_decade": gamma_hpd,
            "se_rad_per_year": se,
            "p_value": p,
        })

    # ── Benjamini-Hochberg FDR ───────────────────────────────────
    p_vals = np.array([s["p_value"] for s in slopes])
    order = np.argsort(p_vals)
    m = len(p_vals)
    reject = np.zeros(m, dtype=bool)
    for rank, idx in enumerate(order, start=1):
        reject[idx] = p_vals[idx] <= (rank / m) * _FDR_ALPHA

    for i, r in enumerate(reject):
        slopes[i]["bh_reject"] = bool(r)

    return {
        "slopes": slopes,
        "fdr_alpha": _FDR_ALPHA,
        "n_seasons_rejected": int(reject.sum()),
    }



