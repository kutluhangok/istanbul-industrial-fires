# DSA210 Industrial Fire & Explosion Incident Analysis

This repository contains a reproducible DSA 210 project analyzing industrial fire and explosion incidents reported by TMMOB Kimya Mühendisleri Odası İstanbul Şubesi.

## Research Question

How do industrial fire and explosion incidents vary across time, location, sector, OSB concentration, weather conditions, and severity?

## Data

Primary data comes from the annual KMO industrial fire and explosion reports. The pipeline extracts incident-level rows from the Ek/Ek-1 appendix tables.

Extracted years:

| Year | Rows |
|---:|---:|
| 2017 | 153 |
| 2018 | 436 |
| 2019 | 541 |
| 2020 | 493 |
| 2021 | 394 |
| 2022 | 587 |
| 2023 | 528 |

The 2024 PDF appendix pages do not expose an extractable text/table layer in the supplied file, so the pipeline logs it as OCR-required in `data/raw/extraction_log.csv`.

External enrichment uses Open-Meteo archive daily weather data for Istanbul.

## Reproduce

Install dependencies:

```bash
python3 -m pip install -r requirements.txt
```

Run the full pipeline:

```bash
MPLCONFIGDIR=/private/tmp/mplconfig python3 run_pipeline.py
```

The script creates:

- `data/raw/kmo_incidents_raw.xlsx`
- `data/processed/kmo_incidents_clean.xlsx`
- `data/enrichment/weather_daily.xlsx`
- `data/processed/istanbul_enriched.xlsx`
- `figures/*.png`
- `reports/model_comparison.xlsx`
- `reports/hypothesis_results.json`
- `reports/pipeline_summary.json`

## Notebooks

- `notebooks/00_pdf_extraction.ipynb`
- `notebooks/01_eda.ipynb`
- `notebooks/02_hypothesis_testing.ipynb`
- `notebooks/03_ml_models.ipynb`

## Key Results

- Total parsed incident rows: 3,132.
- Istanbul incident rows: 826.
- 2021 validation: 394 total rows, 358 fires, 36 explosions, 78 Istanbul incidents.
- H1 seasonality test: Kruskal-Wallis p = 0.191, so month-level differences are not statistically significant across 2017-2023 at alpha 0.05.
- H2 OSB spatial concentration: Mann-Whitney p = 2.99e-06, suggesting Istanbul OSB districts have higher incident counts than non-OSB districts.
- H4 sector vs severity: chi-square p = 1.03e-09, suggesting severity distribution differs by sector.
- Best ML model in the current run: Random Forest, CV macro-F1 about 0.481.

## Ethics and Limitations

The source reports compile publicly reported incidents, not a complete official national registry. Some records have missing firm names, unknown ignition causes, and ambiguous location fields. Weather enrichment is city-level for Istanbul and should not be interpreted as facility-level meteorology. The 2024 appendix requires OCR before it can be included at incident level.
