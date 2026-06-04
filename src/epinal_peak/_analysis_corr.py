"""Circular-linear correlation (Mardia 1976) and pingouin wrapper.

Provides a supplementary circular-linear association measure (Mardia 1976)
and a fallback when the primary von Mises MLE fails to converge.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

_logger = logging.getLogger(__name__)


def _mardia_correlation(
    hours: np.ndarray, years: np.ndarray, n_bootstrap: int = 10000
) -> dict[str, Any]:
    """Mardia (1976) circular-linear rank correlation (pure-scipy).

    Implements:

    .. math::

        \\rho_c^2 = \\frac{r_{cx}^2 + r_{sx}^2 - 2\\, r_{cx}\\, r_{sx}\\, r_{cs}}
                         {1 - r_{cs}^2}

    where :math:`r_{cx}` = Spearman rank correlation of
    :math:`\\cos(\\theta)` with year, :math:`r_{sx}` of
    :math:`\\sin(\\theta)` with year, and :math:`r_{cs}` between
    :math:`\\cos(\\theta)` and :math:`\\sin(\\theta)`.

    Args:
        hours: Peak hour values (0--23).
        years: Year values.
        n_bootstrap: Number of bootstrap resamples for p-value.

    Returns:
        Dict with ``rho_c``, ``p_value_bootstrap``, ``n_obs``, ``method``.
    """
    from scipy.stats import spearmanr

    theta = hours * (2.0 * np.pi / 24.0)
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)
    x = years.astype(float)

    valid = ~(np.isnan(cos_t) | np.isnan(sin_t) | np.isnan(x))
    if not valid.any():
        return {
            "rho_c": np.nan,
            "p_value_bootstrap": np.nan,
            "n_obs": 0,
            "method": "mardia_1976",
        }

    cos_t = cos_t[valid]
    sin_t = sin_t[valid]
    x = x[valid]
    n = len(cos_t)

    r_cx, _ = spearmanr(cos_t, x)
    r_sx, _ = spearmanr(sin_t, x)
    r_cs, _ = spearmanr(cos_t, sin_t)

    denom = 1.0 - r_cs**2
    if denom <= 0.0:
        rho_c = 0.0
    else:
        rho_c = np.sqrt(max(0.0, (r_cx**2 + r_sx**2 - 2.0 * r_cx * r_sx * r_cs) / denom))

    # Bootstrap p-value: permute year labels
    rng = np.random.default_rng(42)
    count_extreme = 0
    observed = rho_c

    for _ in range(n_bootstrap):
        perm_x = rng.permutation(x)
        r_cx_p, _ = spearmanr(cos_t, perm_x)
        r_sx_p, _ = spearmanr(sin_t, perm_x)
        denom_p = 1.0 - r_cs**2  # invariant under permutation of x
        if denom_p <= 0.0:
            rho_p = 0.0
        else:
            rho_p = np.sqrt(max(0.0, (r_cx_p**2 + r_sx_p**2 - 2.0 * r_cx_p * r_sx_p * r_cs) / denom_p))
        if rho_p >= observed:
            count_extreme += 1

    p_value = (count_extreme + 1.0) / (n_bootstrap + 1.0)

    return {
        "rho_c": float(rho_c),
        "p_value_bootstrap": float(p_value),
        "n_bootstrap": n_bootstrap,
        "n_obs": n,
        "method": "mardia_1976",
    }


def circular_linear_correlation(
    hours: np.ndarray, years: np.ndarray, n_bootstrap: int = 10000
) -> dict[str, Any]:
    """Circular-linear correlation, using ``pingouin`` if available.

    Tries ``pingouin.circular_linear_corr`` first.  Falls back to a
    pure-scipy implementation of Mardia (1976) if ``pingouin`` is not
    installed.

    Args:
        hours: Peak hour values (0--23).
        years: Year values.
        n_bootstrap: Bootstrap resamples for Mardia fallback (ignored
            when ``pingouin`` is used).

    Returns:
        Dict with ``rho_c``, ``p_value``, ``n_obs``, ``method``.
    """
    try:
        import pingouin as pg  # noqa: F811

        result = pg.circular_linear_corr(x=years, y=hours, low=0, high=24)
        return {
            "rho_c": float(result["r"].iloc[0]),
            "p_value": float(result["pval"].iloc[0]),
            "n_obs": len(hours),
            "method": "pingouin",
        }
    except ImportError:
        _logger.info("pingouin not installed, falling back to Mardia 1976")
        return _mardia_correlation(hours, years, n_bootstrap=n_bootstrap)
