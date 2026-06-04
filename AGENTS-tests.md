# AGENTS-tests.md — Testing Conventions

## Scope
All tests under `tests/` — unit tests, integration tests, and regression
tests for every pipeline module.

## Constraints & Invariants
- **Coverage targets**: >= 80 % line coverage overall; 100 % coverage on
  critical paths (DST handling, tie-breaking, circular mean/std).
- **DST edge cases**: Every DST transition type must have at least one test:
  - 23-hour day (spring-forward)
  - 25-hour day (autumn-back)
- **Synthetic data tests**: `test_analysis.py` must include a test with known
  synthetic circular data that verifies the regression returns the expected
  coefficient.
- **Slow tests**: Tests that take >2 seconds must be marked with
  ``@pytest.mark.slow``. Run with ``pytest tests/ -m "not slow"`` for a
  quick check.
- **Logging suppression**: The autouse fixture in `conftest.py` keeps test
  output clean. Do NOT log at INFO or below during tests unless explicitly
  testing logging behaviour.

## Test Structure
Use `class Test*` for logical grouping. Each test function is a single
assertion or a small set of related assertions.

## Fixtures
Shared fixtures live in `conftest.py`:
- `sample_temperature_data`: 48-hour synthetic data with a DST transition.
- `sample_peak_hours`: 30 years of synthetic daily peak hours with a known
  weak trend.

Module-specific fixtures go in the module's own test file using
``conftest.py``-level fixtures where possible.

## Naming
Test functions use `test_<scenario>_<expected_behaviour>` pattern (snake
case).  Examples:
- `test_dst_spring_forward_retains_23_records`
- `test_tie_earliest_picks_first_maximum`
- `test_circular_mean_of_uniform_data_is_nan`

## References
- [Root AGENTS.md](AGENTS.md) — global coding conventions
- `conftest.py` — fixture definitions and pytest configuration
