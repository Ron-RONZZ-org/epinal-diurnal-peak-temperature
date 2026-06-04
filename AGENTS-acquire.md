# AGENTS-acquire.md — Data Acquisition Rules

## Scope
`src/epinal_peak/acquire.py` — downloads sub-daily temperature records from
Météo-France via `pooch`, with a local fallback if the remote source is
unreachable.

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
  matching the Météo-France public API schema.

## Edge Cases
- **Empty registry**: Raise a clear `FileNotFoundError`.
- **Hash mismatch**: Log the expected vs. actual hash; raise `ValueError`.
- **Network timeout**: Catch `requests.Timeout`, log warning, try fallback.

## References
- [Root AGENTS.md](AGENTS.md) — global coding conventions
- `config.py`: `EpinalPeakConfig.pooch_registry_file`, `.local_fallback_dir`,
  `.raw_dir`, `.raw_data_filename`
