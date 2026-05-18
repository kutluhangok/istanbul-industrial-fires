# DSA210 Industrial Fire & Explosion Incident Analysis

This repository contains a reproducible DSA 210 project analyzing industrial fire and explosion incidents reported by TMMOB Kimya Mühendisleri Odası İstanbul Şubesi.

## Research Question

How do industrial fire and explosion incidents vary across time, province, sector, OSB industrial capacity, and severity across Turkey?

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
| 2024 | 720 |

The 2024 PDF appendix pages did not expose an extractable text/table layer, but a manually provided 2024 Excel table is now included as `data/raw/kmo2024_manual.xlsx`.

External enrichment now uses two low-friction exposure layers. OSBÜK province-level organized industrial zone data adds OSB count, operational OSB count, OSB area in hectares, and total parcel counts. Daily province-level weather data from NASA POWER adds temperature, relative humidity, precipitation, and wind speed. This lets the analysis separate raw incident volume from industrial exposure and test whether temporal environmental conditions add signal beyond month/weekday patterns.

OSBÜK source: https://osbuk.org.tr/view/sayilarlaosb/osbliste.php

Weather source: https://power.larc.nasa.gov/

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
- `data/enrichment/weather_daily_by_province.xlsx`
- `data/enrichment/province_coordinates.xlsx`
- `data/enrichment/osbuk_osb_list.xlsx`
- `data/enrichment/osb_exposure_by_city.xlsx`
- `data/processed/istanbul_enriched.xlsx`
- `data/processed/city_year_osb_panel.xlsx`
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

- Total parsed incident rows: 3,852.
- Province-day weather panel: 216,228 daily rows for 74 provinces; 3,430 of 3,852 incidents have matched weather covariates.
- Turkey-wide OSB exposure rows: 418 OSBs aggregated to all 81 provinces.
- OSB exposure totals in the downloaded OSBÜK table: 141,637.5 hectares and 63,860 parcels.
- 2021 validation: 394 total rows, 358 fires, 36 explosions, 78 Istanbul incidents.
- 2024 validation: 720 total rows, 694 fires, 26 explosions.
- H1 seasonality test: Kruskal-Wallis p = 0.378, so month-level differences are not statistically significant across 2017-2024 at alpha 0.05.
- H2 OSB exposure: provinces with OSB exposure have higher incident counts than provinces without matched exposure in the KMO data, Mann-Whitney p = 1.70e-29.
- H3 exposure correlation: province incident count correlates strongly with OSB parcel count, Spearman rho = 0.807, p = 1.17e-17; and OSB area, rho = 0.764, p = 6.02e-15.
- H3b national weather: province-day incident frequency has statistically detectable but very small correlations with mean temperature (rho = 0.034), wind speed (rho = 0.048), humidity (rho = 0.009), and precipitation (rho = -0.011). These should be interpreted as weak temporal covariates, not standalone causal effects.
- H4 sector vs severity: chi-square p = 6.51e-11, suggesting severity distribution differs by sector.
- Best ML model in the current run: Random Forest, CV macro-F1 about 0.496.

## Ethics and Limitations

The source reports compile publicly reported incidents, not a complete official national registry. Some records have missing firm names, unknown ignition causes, and ambiguous location fields. Weather is matched at province-city-center resolution rather than exact facility coordinates, and about 11% of incidents remain unmatched because PDF location extraction was not reliable enough to assign a valid province. OSBÜK exposure is current/static rather than year-specific historical OSB capacity, so exposure-adjusted rates should be interpreted as approximate. The 2024 incident table is included from the manually provided Excel export, not from PDF OCR.
