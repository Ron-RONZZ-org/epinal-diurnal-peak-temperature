# Diurnal Peak Temperature Trends at Épinal, France

Investigating long-term trends in the diurnal timing of the daily maximum temperature for **Épinal**, located in the Grand Est region of France.

## Research Question

**Has the clock hour of the daily maximum temperature at Épinal undergone a detectable change over the past several decades?**

Whereas existing research has predominantly examined *regional warming trends*, this study focuses specifically on the diurnal cycle to determine whether the hour of peak temperature exhibits a systematic shift over time.

## Methodology

1. **Data Acquisition** — Historical sub-daily temperature records from a publicly accessible source (e.g., Météo-France public data portal or Infoclimat) for the meteorological station nearest to Épinal.

2. **Data Processing** — Raw data are preprocessed using a Python script (see `code/`). Steps include:
   - Conversion of UTC timestamps to local time (CET/CEST).
   - Extraction, for each calendar day, of the hour corresponding to the highest recorded temperature.

3. **Analysis & Visualization** — The primary output is a time-series plot of the daily peak temperature hour across the observation period. Secondary analyses examine seasonal variations (summer vs. winter) and compute a long-term trend line using appropriate statistical methods.

## Expected Outputs

- Curated and validated dataset.
- Complete Python script for data processing and analysis.
- Final visualization illustrating the trend in diurnal peak temperature.
- Project report detailing the methodological approach and empirical findings.

## Project Management

This project is managed on the **Open Science Framework (OSF)**. The Git repository serves as an external storage component linked to the OSF project.

- OSF Project: [Add OSF link here]

## License

This repository is **dual-licensed** depending on the material type:

| Component | License | File |
|-----------|---------|------|
| **Source code** (Python scripts, `code/`) | **MIT** | [`LICENSE`](LICENSE) |
| **Non-software materials** (data, figures, report, documentation) | **Creative Commons Attribution 4.0 International (CC BY 4.0)** | [`LICENSE.content`](LICENSE.content) |

[![CC BY 4.0](https://licensebuttons.net/l/by/4.0/88x31.png)](https://creativecommons.org/licenses/by/4.0/)
