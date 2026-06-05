# Coding Scheme — Épinal Diurnal Peak Temperature

## Overview

This document describes the complete data-analysis pipeline for the project
**"The Shifting Peak: Analyzing Diurnal Maximum Temperature Trends in
Épinal (Grand Est, France)"**, intended for OSF pre-registration.

The pipeline is implemented as a Python package (`src/epinal_peak/`),
orchestrated via `make` targets, and rendered as a Quarto manuscript.

---

## 1. Pipeline Stages

```
acquire → preprocess → analyze → visualize → report → preprint
```

### 1.1 Acquire (`make acquire`)
- **Input**: Météo-France Synop Essentielles API (station 88136001, Épinal).
- **Download**: Via `pooch` with a registry file (`data/external/registry.txt`)
  mapping URLs to SHA256 hashes.  Fallback to a local directory for offline use.
- **Output**: `data/raw/epinal_temperature_raw.csv` — raw sub-daily temperature
  records (one row per hour) in UTC.

### 1.2 Preprocess (`make preprocess`)
**UTC → Local time conversion**
- Parse `AAAAMMJJHH` (YYYYMMDDHH) → UTC datetime → local datetime
  (`Europe/Paris`, CET/CEST via `zoneinfo`).
- DST transitions handled transparently: 25-hour autumn days and 23-hour
  spring days are processed correctly.

**Quality control**
1. **Physical limits filter**: Drop hourly readings outside
   [−30 °C, +50 °C] (configurable).
2. **Flatline detection**: Flag any run of ≥6 consecutive identical
   hourly temperature readings (stuck sensor).
3. **Temporal gap flag**: Flag observations after ≥30 consecutive
   missing days.
4. **Circular outlier detection**: Flag peak hours >5 MAD from the
   circular median (angular deviation in hours). Threshold configurable.
5. **Minimum valid observations**: A day is valid only if ≥18
   (24 − 6) hourly observations are non-null.
6. **Diurnal amplitude filter**: Exclude days where
   `T_max − T_min < 2.0 °C` (flat trace / radiation error).

**Peak extraction**
- For each calendar day (local date), select the hour with the
  highest temperature.
- **Tie-breaking (primary)**: Earliest peak hour.
- **Tie-breaking (sensitivity)**: Latest peak hour.
- Tied hours occur on ~15 % of days (multiple hours with same maximum).

**Schema validation**: `pandera.DataFrameModel` validates column names,
dtypes, and value ranges at each stage boundary.

**Output**: `data/processed/epinal_daily_peak_hour.csv` — one row per day
with `peak_hour`, `peak_temperature`, `diurnal_amplitude`, and QC flags.

### 1.3 Analyze (`make analyze`)
All models are **von Mises circular GLMs** fit by direct maximum
likelihood estimation (Fisher & Lee 1992).  The model is:

$$
\begin{aligned}
\theta_i &= 2\pi \times \text{peak\_hour}_i / 24 \\
\mu_i &= \beta_0 + \beta_1 \times (\text{year}_i - \text{year\_center}) \\
\log L &= \sum_i \kappa \cos(\theta_i - \mu_i) - n \log(2\pi I_0(\kappa))
\end{aligned}
$$

where $\kappa = \exp(\text{log\_kappa}) > 0$ and $I_0$ is the modified
Bessel function of order 0.  Optimisation via `scipy.optimize.minimize`
(L-BFGS-B).

#### 1.3.1 Year × Season Interaction Test (gatekeeper)
- **Reduced model**: 6 parameters — 4 season intercepts + 1 common year
  slope + log_kappa.
- **Full model**: 9 parameters — 4 season intercepts + 4 season-specific
  year slopes + log_kappa.
- **Test**: Likelihood-ratio, χ² with 3 degrees of freedom.
- **α**: 0.05.
- If **significant**, seasonal stratification is the primary framework.
  Per-season Wald p-values from the full-model Hessian are corrected
  via **Benjamini-Hochberg FDR** (α = 0.05, m = 4).

#### 1.3.2 Primary Analysis: Seasonal Stratification
- Fit the von Mises GLM independently to each meteorological season:
  - Spring (March–May)
  - Summer (June–August)
  - Autumn (September–November)
  - Winter (December–February)
- **Bootstrap CI**: 1 000 resamples, percentile interval at 95 %.
  Each resample draws `n` rows with replacement, using
  `numpy.random.Generator` with independent seeds per iteration.
  The primary tie-breaking rule (earliest peak) is used.
- **Filter**: Only years with ≥30 observations are included
  (configurable: `min_years_for_trend = 30`).
- **Primary endpoint**: Slope β in hours per decade.

#### 1.3.3 Pooled Analysis (supplementary)
- Same von Mises GLM fit to all observations pooled across seasons.
- Bootstrap CI with 1 000 resamples.

#### 1.3.4 Supplementary Analyses
| Analysis | Description |
|----------|-------------|
| **Temperature-weighted regression** | Each day weighted by diurnal amplitude (`T_max − T_min`) in the log-likelihood. Per season + pooled. |
| **Circular–linear rank correlation** | Mardia (1976) non-parametric correlation between peak hour and year. Bootstrap p-value with 10 000 resamples. Per season + pooled. |
| **Autocorrelation diagnostic** | Circular ACF (Fisher 1993, §6.3.4) at lags 1–7, computed per season and pooled on model residuals sorted by date. |

### 1.4 Sensitivity Analyses

#### Pooled Sensitivity (6 variants)
| Variant | Modification | Rationale |
|---------|-------------|-----------|
| Latest tie | Use `peak_hour_sensitivity` (latest-peak rule) | Test sensitivity to tie-breaking |
| Min years = 20 | Reduce `min_years_for_trend` from 30 to 20 | Test sensitivity to record length |
| 3-hour subsampling | Round peak hours to nearest 3-hour bin | Test sensitivity to temporal resolution |
| 6-hour subsampling | Round peak hours to nearest 6-hour bin | Lower-resolution check |
| Amp ≥ 1.0 °C | Reduce diurnal amplitude threshold to 1.0 °C | More permissive inclusion |
| Amp ≥ 3.0 °C | Increase diurnal amplitude threshold to 3.0 °C | More restrictive inclusion |

#### Seasonal Sensitivity (4 seasons × 6 variants = 24 models)
Each season independently runs all 6 pooled sensitivity variants above.

#### Temperature-Weighted Sensitivity
- Per-season temperature-weighted regression (see §1.3.4).

### 1.5 Visualize (`make visualize`)
Publication-quality figures:
- Seasonal trend panels with regression line + 95 % CI band
- Rose diagrams (polar histograms) per season
- 5-year moving circular mean per season
- Wrapped scatter plot (pooled)
- Context time series (annual mean daily maximum temperature)

### 1.6 Report (`make report`)
Summary statistics tables (printed to stdout / log).

### 1.7 Preprint (`./preprint/recompile.sh`)
Quarto manuscript compiled to PDF (xelatex) and HTML.

---

## 2. Key Parameters

| Parameter | Value | Purpose |
|-----------|-------|---------|
| Station ID | 88136001 | Météo-France RADOME (Épinal) |
| Timezone | Europe/Paris | CET/CEST via zoneinfo |
| Analysis period | 1992–2025 | After QC filtering (35 years) |
| Min hourly obs / day | 18 (24 − 6) | Valid-day threshold |
| Physical temp bounds | −30 °C, +50 °C | Instrument error filter |
| Flatline threshold | 6 consecutive identical hours | Stuck sensor |
| Temporal gap threshold | 30 consecutive missing days | Post-gap flag |
| Circular outlier MAD | 5.0 | Multiples of MAD from circular median |
| Min diurnal amplitude | 2.0 °C | Flat-trace / radiation error exclusion |
| Min years for trend | 30 | Per-season sample adequacy |
| Primary tie rule | earliest | Tie-breaking for peak hour |
| Bootstrap iterations | 1 000 | CI precision |
| Bootstrap CI level | 95 % | Percentile interval |
| Interaction LR α | 0.05 | Gatekeeper significance |
| FDR α | 0.05 | Benjamini-Hochberg target |
| Seasons | Meteorological (MAM, JJA, SON, DJF) | Northern Hemisphere |

---

## 3. Software & Dependencies

### Python (≥3.10)
| Package | Version (approx.) | Use |
|---------|-------------------|-----|
| `numpy` | ≥1.24 | Numerical arrays |
| `pandas` | ≥2.0 | Data handling |
| `scipy` | ≥1.12 | Von Mises MLE, special functions |
| `statsmodels` | — | GLM framework (reference only) |
| `pooch` | — | Data download + caching |
| `matplotlib` | ≥3.7 | Figures |
| `seaborn` | — | Figure styling |
| `pandera` | — | Schema validation (optional) |
| `zoneinfo` | stdlib | Timezone handling (no pytz) |

### External tools
| Tool | Version | Use |
|------|---------|-----|
| Quarto CLI | ≥1.4 | Manuscript compilation |
| XeLaTeX | TeX Live 2026 | PDF rendering |

Install: `pip install -e ".[dev]"`

---

## 4. Data Flow

```
Météo-France API
       │
       ▼ pooch (registry.txt)
 data/raw/epinal_temperature_raw.csv
       │
       ▼ preprocess.py
 data/processed/epinal_daily_peak_hour.csv
       │
       ▼ analysis.py
 results/epinal_analysis_results.json
       │
       ▼ visualise.py
 results/figures/*.pdf
       │
       ▼ report.py
 stdout (tables)
       │
       ▼ recompile.sh (Quarto)
 preprint/main.pdf
 preprint/main.html
```

---

## 5. Project Structure

```
src/epinal_peak/           # Python package
├── config.py              # All parameters (EpinalPeakConfig dataclass)
├── acquire.py             # Data download
├── preprocess.py          # QC + peak extraction
├── _preprocess_qc.py      # QC helpers (private)
├── _preprocess_schemas.py # pandera schemas (private)
├── analysis.py            # Orchestrator (public API)
├── _analysis_regression.py # Von Mises MLE math
├── _analysis_interaction.py # LR test + FDR
├── _analysis_bootstrap.py   # Bootstrap engine
├── _analysis_variants.py    # Sensitivity variants
├── _analysis_weighted.py    # Temperature-weighted regression
├── _analysis_corr.py        # Circular-linear correlation
├── _circular_utils.py       # Shared circular helpers
├── visualize.py             # Figure generation
└── report.py                # Summary tables
```

