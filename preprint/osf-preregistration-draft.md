# OSF Preregistration — Epinal Diurnal Peak Temperature

> **Status**: Draft — ready for copy-paste into OSF Prereg form
> **Date**: 2026-06-04
> **Corresponding issue**: [#2](https://github.com/Ron-RONZZ-org/epinal-diurnal-peak-temperature/issues/2)

---

## 1. Title

**Has the Clock Hour of Daily Maximum Temperature at Épinal, France, Shifted Over Recent Decades?**

## 2. Authors

Rong Zhou

## 3. Description

We investigate whether the timing (clock hour) of the daily maximum
temperature at a single station — Épinal, France (Météo-France RADOME
station 88136001, 48.21°N, 6.45°E) — has undergone a systematic shift
over the 40-year period 1986–2025.

Using sub-daily (hourly) temperature records from the Météo-France
Synop Essentielles / RADOME network, we:
1. Convert UTC timestamps to local time (CET/CEST via ``zoneinfo``).
2. Extract, for each calendar day, the clock hour of the maximum
   recorded temperature.
3. Model the long-term trend using a **von Mises circular GLM**,
   appropriate for the circular (0–23 h) nature of the response
   variable.

The primary result is the slope coefficient β expressed in **hours per
decade**, with a 95 % bootstrap confidence interval obtained from 1 000
resamples.

## 4. Hypotheses

- **H₁ (alternative)**: The circular mean clock hour of daily maximum
  temperature at Épinal has changed between 1986 and 2025. Equivalently,
  the slope β of the von Mises GLM differs from zero (β ≠ 0).
- **H₀ (null)**: No systematic change — β = 0.

A positive β indicates the daily maximum occurs **later** in the day
over time; a negative β indicates it occurs **earlier**.

## 5. Design Plan

- **Type**: Observational study of a single meteorological station.
- **Randomization**: None (observational).
- **Blinding**: None.
- **Control group**: None (single-station time series).

## 6. Sampling Plan

### Data source
Météo-France Synop Essentielles / RADOME network, station **88136001**
(Épinal). The station opened on 1 June 1986 — no sub-daily data exist
before this date.

### Inclusion criteria
- All calendar days from **1 January 1986** through **31 December 2025**
  (inclusive).
- Days with ≥ 18 valid hourly observations (i.e., ≤ 6 missing hours).

### Exclusion criteria
Applied **per day**, sequentially:
1. **Missing data**: Fewer than 18 valid hourly observations
   (``max_missing_hours_per_day = 6``).
2. **Diurnal amplitude**: Days where
   ``T_max - T_min < 2.0 °C`` (``min_diurnal_amplitude``) — excludes
   flat-trace instrument artefacts.
3. **Outlier flagging**: Observations exceeding
   ``outlier_mad_threshold`` (5.0 median absolute deviations from the
   series median) are flagged. The flagged day is excluded from the
   primary analysis.

### Minimum data requirement
If fewer than 30 unique calendar years contain valid daily peak-hour
observations, the trend analysis will **not** be performed (the sample
is deemed insufficient for a meaningful trend estimate).

## 7. Variables

| Variable | Role | Type | Range | Details |
|---|---|---|---|---|
| **Peak hour** | Dependent | Circular (integer) | 0–23 | Clock hour (local time CET/CEST) of daily T_max. Ties resolved by **earliest** occurrence for primary analysis. |
| **Year** | Independent | Continuous (integer) | 1986–2025 | Calendar year, **centred** at midpoint of data range before regression. |
| **Season** | Stratification (exploratory) | Categorical | Spring / Summer / Autumn / Winter | Meteorological seasons: MAM (3–5), JJA (6–8), SON (9–11), DJF (12–2). |

### Tie-breaking rule
- **Primary**: Earliest peak hour when multiple hours share the daily
  maximum temperature.
- **Sensitivity**: Latest peak hour (see Section 8, sensitivity #1).

## 8. Analysis Plan

### 8.1 Primary analysis

Fit a **von Mises circular regression** by **direct maximum likelihood
estimation (MLE)** via ``scipy.optimize``, following Fisher & Lee
(1992, *Biometrics*).

**Model** (log-likelihood to be maximised):

```math
\log L = \sum_{i=1}^{n} \kappa \cos(\theta_i - \beta_0 - \beta_1 x_i)
          - n \log\bigl(2\pi I_0(\kappa)\bigr)
```

where:
- $\theta_i = 2\pi \times \text{peak\_hour}_i / 24$ (radians)
- $x_i = \text{year}_i - \text{year\_midpoint}$ (centred year)
- $I_0(\kappa)$ = modified Bessel function of order 0
- $\beta_1$ = trend coefficient (radians/year)

Numerical optimisation via ``scipy.optimize.minimize`` with
$\kappa = \exp(\text{log\_kappa})$ to enforce $\kappa > 0$.
Standard errors from the inverse Hessian of the negative log-likelihood.

The primary endpoint $\beta_1$ is converted to **hours per decade**:

```math
\beta\ (\text{hours/decade}) = \beta_1 \times \frac{24}{2\pi} \times 10
```

**Confidence interval**: Bootstrap — resample daily observations with
replacement (1 000 iterations), refit the model each iteration, and
extract the 2.5th and 97.5th percentiles of the bootstrap distribution.

**Reporting**: β (hours/decade), 95 % bootstrap CI, number of
valid days/years included.

### 8.2 Seasonal sub-analyses (exploratory, flagged secondary)

Repeat the primary analysis separately for each meteorological season:

| Season | Months | Label |
|---|---|---|
| Spring | March–May | MAM |
| Summer | June–August | JJA |
| Autumn | September–November | SON |
| Winter | December–February | DJF |

These are **exploratory** — no multiple-testing correction is applied.
Results are presented as supplementary information.

### 8.3 Sensitivity analyses (all pre-registered)

All four sensitivity variants are pre-registered to constrain
researcher degrees of freedom:

| # | Variant | Parameter changed | Rationale |
|---|---|---|---|
| 1 | Tie-breaking **latest** | ``sensitivity_tie_rule = "latest"`` | Tests whether tie-choice affects the trend |
| 2 | Reduced minimum years (20) | ``sensitivity_min_years = 20`` | Tests robustness to shorter record |
| 3 | Strict outlier flagging (3.0 MAD) | ``sensitivity_mad_threshold = 3.0`` | Tests sensitivity to extreme events |
| 4 | Exclude post-2000 data | ``sensitivity_exclude_post_2000 = True`` | Tests whether recent decades drive the trend |

Each sensitivity is run independently; only the varied parameter
changes. All other parameters match the primary analysis.

### 8.4 Contingency plan

If the von Mises regression MLE fails to converge (e.g., all peak
hours identical, flat likelihood, or extreme data sparsity), we will
fall back to a **non-parametric circular–linear association test**:

**Primary contingency**: **Circular–linear rank correlation**
(Mardia 1976, *JRSS B*)

```math
\rho_c^2 = \frac{r_{cx}^2 + r_{sx}^2 - 2\, r_{cx}\, r_{sx}\, r_{cs}}
               {1 - r_{cs}^2}
```

where $r_{cx}$ is the rank correlation of $\cos\theta$ with year,
$r_{sx}$ of $\sin\theta$ with year, and $r_{cs}$ between
$\cos\theta$ and $\sin\theta$.
- Uses **all daily observations** — no aggregation loss.
- Makes **no distributional assumptions** about $\kappa$.
- Bootstrap p-value from 10 000 resamples of year–hour pairs.
- Reports the correlation coefficient $\rho_c$ and its p-value.

**Secondary (descriptive only)**: Compute circular mean peak hour
per decade (1986–1995, 1996–2005, 2006–2015, 2016–2025) and display
as a simple table. Do **not** fit a regression through these 4 points
(aggregation bias — see reasoning in project documentation).

Report both the convergence failure and the contingency results.

## 9. Software

| Component | Version |
|---|---|
| Python | ≥ 3.10 |
| numpy | ≥ 1.24 |
| pandas | ≥ 2.0 |
| scipy | ≥ 1.12 |
| statsmodels | ≥ 0.14 |
| pooch | ≥ 1.8 |
| matplotlib | ≥ 3.8 |
| seaborn | ≥ 0.13 |
| pandera | ≥ 0.18 |

All package versions are pinned in ``pyproject.toml``.

## 10. Data Availability

The raw data are publicly available from the Météo-France public data
portal (Synop Essentielles / RADOME network). Processed data and
analysis code are archived in the associated OSF repository.

---

*This pre-registration was created using the OSF Prereg template and
archived before any data analysis began. See the project repository
at https://github.com/Ron-RONZZ-org/epinal-diurnal-peak-temperature
for the full analysis code and AGENTS.md documentation.*

---

## Submission Instructions

To submit this pre-registration on OSF:

1. Go to https://osf.io and log in (or create an account).
2. Create a new project named **"Epinal Diurnal Peak Temperature"**.
3. Navigate to **Registrations** → **New Registration**.
4. Select the **OSF Preregistration** template.
5. Copy-paste the content of this document into the form, section by
   section.
6. Once submitted, OSF will assign a **DOI** (e.g.,
   `https://doi.org/10.17605/OSF.IO/XXXXX`).
7. Add the DOI to `README.md` in the project repository and update the
   OSF project link.
8. Close issue #2 with a comment linking to the DOI.
