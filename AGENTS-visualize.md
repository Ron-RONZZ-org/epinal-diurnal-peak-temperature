# AGENTS-visualize.md — Figure Generation Rules

## Scope
`src/epinal_peak/visualize.py` — publication-quality figures: wrapped
scatter, rose diagram, and seasonal trend plots.

## Constraints & Invariants
- **Figure dimensions**: Single-column journal width = 3.5 in (88.9 mm);
  full-page width = 7.0 in (177.8 mm). Use these as default widths.
- **DPI**: Minimum 300 DPI for raster export.
- **Color palette**: Use a colorblind-safe palette (e.g.,
  `seaborn.color_palette("colorblind")`). Avoid red-green contrasts.
- **Dual export**: Every figure is saved as **PDF** (vector, for publication)
  and **PNG** (raster, for quick viewing) at 300 DPI.
- **Font sizes**: Axis labels >= 10 pt, tick labels >= 8 pt, title >= 12 pt.
  Use sans-serif font (Helvetica or DejaVu Sans).
- **File naming**: Lowercase, underscored, descriptive:
  `{variant}_{description}.{ext}`

## Specific Plots

### Wrapped scatter (`plot_wrapped_scatter`)
- X-axis: year
- Y-axis: peak hour, zoomed to data concentration range (8--22 h) with annotation noting the full circular range (0h ↔ 24h)
- Overlay: Von Mises regression trend line (from ``analysis.py`` fitted model) with 95 % CI band from bootstrap slope distribution
- Rationale: The full circular y-axis (0--24) would make the ~0.1 h trend over 40 years invisible. Zooming to 8--22 h shows the data concentration (77 % of observations fall in 12--18 h) while keeping the circular context visible.

### Rose diagram (`plot_rose_diagram`)
- Circular histogram of peak-hour frequency
- N bins: 24

### Seasonal trend (`plot_seasonal_trend`)
- Faceted: one panel per meteorological season
  — spring (MAM), summer (JJA), autumn (SON), winter (DJF)
- Each panel: scatter + von Mises regression trend line + 95 % CI band, matching wrapped-scatter style
- Y-axis: zoomed to 8--22 h (same as wrapped scatter)

## Data Contracts
- **Input**: Peak-hour CSV (DataFrame with `year`, `peak_hour`, `month`,
  `date` columns).
- **Output**: Figure files saved to `config.figures_dir`.

## Edge Cases
- **Zero variance**: If all peak hours are identical, the trend line is flat.
  Plot it as a horizontal line.
- **Gappy years**: Do NOT interpolate across multi-year gaps in the trend line.

## References
- [Root AGENTS.md](AGENTS.md) — global coding conventions
- `config.py`: `EpinalPeakConfig.figures_dir`,
  `.spring_start`, `.spring_end`, `.summer_start`, `.summer_end`,
  `.autumn_start`, `.autumn_end`, `.winter_start`, `.winter_end`
