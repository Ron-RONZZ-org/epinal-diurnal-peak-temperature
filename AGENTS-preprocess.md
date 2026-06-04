# AGENTS-preprocess.md — Preprocessing & QC Rules

## Scope
`src/epinal_peak/preprocess.py` — quality control, UTC-to-local time
conversion, daily peak temperature extraction, and output validation.

## Constraints & Invariants
- **Never use `pytz`**: Use `zoneinfo` (stdlib) for all timezone operations.
- **Pandera input validation**: Call `validate_input_schema()` immediately on
  raw data; call `validate_output_schema()` before returning peak-hour output.
- **Tie-breaking**: Earliest peak hour is the *primary* analysis variable.
  Latest peak hour is a *sensitivity* variable. Do NOT average tied hours
  (circular average of 23 and 0 = 11.5 — meaningless).
- **DST edge cases**: 23-hour (spring-forward) and 25-hour (autumn-back) days
  must be handled correctly. Do not drop or duplicate records on transition
  days.
- **Missing data threshold**: A day is valid only if at least
  `config.min_valid_fraction` (default 75 %) of hourly observations are
  non-null.

## Data Contracts
- **Input**: Raw CSV with UTC timestamps and temperature readings.
- **Output**: CSV with columns `date` (YYYY-MM-DD), `peak_hour` (int, 0--23),
  `peak_temperature` (float).

## Edge Cases
- **Spring-forward day (23 hours)**: The missing hour (02:00 CET -> 03:00
  CEST) should not cause an alignment error.
- **Autumn-back day (25 hours)**: Duplicate 02:00--03:00 CEST -> CET records
  must be distinguished by timezone fold.
- **All-missing day**: Return row with `peak_hour = NaN`; do NOT drop.
- **Station gap**: If consecutive days are missing, leave gaps — do NOT
  interpolate across them.

## References
- [Root AGENTS.md](AGENTS.md) — global coding conventions
- `config.py`: `EpinalPeakConfig.max_missing_hours_per_day`,
  `.min_valid_fraction`, `.outlier_mad_threshold`, `.primary_tie_rule`,
  `.sensitivity_tie_rule`
