# AGENTS-acquire.md — Data Acquisition Rules

## Scope
`src/epinal_peak/acquire.py` — downloads sub-daily temperature records from
Météo-France via `pooch`, with a local fallback if the remote source is
unreachable.

## Data Source

| Item | Detail |
|------|--------|
| **Station** | EPINAL — MF `88136001` (RADOME network) |
| **Coordinates** | 48.210833°N, 6.451667°E, 317 m |
| **Period** | 1 June 1986 – present |
| **Product** | Météo-France *Observations in situ* (RADOME) — hourly temperature |
| **Access** | `portail-api.meteofrance.fr` (Observations API, free tier, API token required) |
| **Fallback** | `meteo.data.gouv.fr` bulk CSV downloads (no token, less structured) |

**Note**: The Météo-France SYNOP Essentielles OMM dataset (monthly files since 1996)
is **not** the target. Use the RADOME Observations API instead for hourly/sub-daily
data from this station.

## Constraints & Invariants
- **Always validate after download**: Call `validate_raw_schema()` before
  returning data to the pipeline.
- **Hash verification**: The pooch registry file must contain SHA256 hashes
  for every tracked URL. Do NOT skip hash checks.
- **Local fallback**: If the remote download fails (HTTP error, timeout,
  DNS failure), fall back to a local copy in `external_dir/fallback/`.
- **Retry**: Implement at least 1 retry with exponential backoff before
  falling back.

## Data Contracts
- **Input**: Registry file (`external_dir / pooch_registry_file`) mapping
  filenames to URL + SHA256 hash.
- **Output**: CSV file saved to `raw_dir / raw_data_filename` with columns
  matching the Météo-France RADOME API schema (station, date, temperature,
  quality flags).

## Edge Cases
- **Empty registry**: Raise a clear `FileNotFoundError`.
- **Hash mismatch**: Log the expected vs. actual hash; raise `ValueError`.
- **Network timeout**: Catch `requests.Timeout`, log warning, try fallback.

## References
- [Root AGENTS.md](AGENTS.md) — global coding conventions
- `config.py`: `EpinalPeakConfig.pooch_registry_file`, `.local_fallback_dir`,
  `.raw_dir`, `.raw_data_filename`
- Issue #5 — data acquisition implementation
- Issue #1 — research plan and architectural decisions
