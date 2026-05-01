# Final Report — Industrial Fire & Explosion Incident Analysis

## Motivation

Industrial fires and explosions are preventable events with high human, operational, and environmental cost. This project studies when, where, and under what sectoral or environmental conditions these incidents cluster, with special attention to Istanbul and OSB-heavy districts.

## Data Source

The primary source is the annual TMMOB Kimya Mühendisleri Odası İstanbul Şubesi industrial fire and explosion reports. Incident-level rows were extracted from Ek/Ek-1 appendix tables for 2017-2023, producing 3,132 rows. The supplied 2024 PDF appendix does not contain an extractable text/table layer, so it is documented as requiring OCR before inclusion.

The dataset was enriched with Open-Meteo daily Istanbul weather variables: mean/max/min temperature, relative humidity, precipitation, and wind speed.

## Data Preparation

The pipeline standardizes dates, event types, casualty counts, severity labels, city/district fields, OSB flags, sectors, calendar features, and weather joins. Missing or unknown source cells are retained rather than force-filled.

Important validation check: the 2021 appendix extraction produced 394 rows, including 358 fires, 36 explosions, and 78 Istanbul incidents.

## Analysis

Exploratory analysis produced monthly, geographic, sectoral, severity, OSB, weather, ignition-source, and cause visualizations in `figures/`.

Hypothesis tests:

- Seasonality: Kruskal-Wallis p = 0.191. Across 2017-2023, monthly count differences are not statistically significant at alpha 0.05.
- Istanbul OSB clustering: Mann-Whitney p = 2.99e-06. OSB districts show substantially higher incident counts than non-OSB districts.
- Weather association: daily Istanbul incident counts show weak correlations with temperature, humidity, wind, and precipitation in this dataset.
- Severity associations: sector vs severity is statistically significant with chi-square p = 1.03e-09; OSB vs severity is not significant in this run.

## Machine Learning

Severity classification used temporal, spatial, sector, event type, OSB, and weather features. Models tested were logistic regression, random forest, and XGBoost. Random Forest achieved the best cross-validated macro-F1 in the current run at about 0.481, with test accuracy around 0.840. SHAP output is saved as `figures/14_shap_summary.png`.

## Findings

Incident counts are concentrated in industrial cities and Istanbul districts with strong manufacturing/OSB presence. Sector appears more informative for severity than the OSB flag alone. Weather variables, measured at city level, show only weak association with incident counts, suggesting facility processes, sector, equipment, and reporting coverage may dominate over broad daily weather in this dataset.

## Limitations and Future Work

The source is compiled from public reports, so it should not be treated as a complete official registry. Many ignition and formation causes are unknown. Weather enrichment is not facility-specific. The 2024 appendix needs OCR. Future work should add facility coordinates, official fire department response data, richer Turkish holiday calendars, OCR for scanned appendices, and district-level exposure denominators such as number of factories or industrial employment.
