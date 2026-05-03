# Final Report — Exposure-Adjusted Industrial Fire & Explosion Incident Analysis in Turkey

## Motivation

Industrial fires and explosions are preventable events with high human, operational, and environmental cost. This project studies where industrial incidents cluster across Turkey and whether those patterns align with organized industrial zone exposure, measured by OSB area and parcel capacity.

## Data Source

The primary source is the annual TMMOB Kimya Mühendisleri Odası İstanbul Şubesi industrial fire and explosion reports. Incident-level rows were extracted from Ek/Ek-1 appendix tables for 2017-2023, producing 3,132 rows. The supplied 2024 PDF appendix is image-based. OCR tests with macOS Vision and EasyOCR did not produce reliable cell-level text, so 2024 is not merged into the clean incident dataset unless a manually verified `data/raw/kmo2024_manual.xlsx` file is supplied.

The dataset was enriched with OSBÜK organized industrial zone exposure data, including province-level OSB counts, operational OSB counts, total OSB area in hectares, and total OSB parcel counts.

## Data Preparation

The pipeline standardizes dates, event types, casualty counts, severity labels, city/district fields, OSB flags, sectors, calendar features, and OSB exposure joins. Missing or unknown source cells are retained rather than force-filled.

Important validation check: the 2021 appendix extraction produced 394 rows, including 358 fires, 36 explosions, and 78 Istanbul incidents.

## Analysis

Exploratory analysis produced monthly, geographic, sectoral, severity, OSB exposure, ignition-source, and cause visualizations in `figures/`.

Hypothesis tests:

- Seasonality: Kruskal-Wallis p = 0.191. Across 2017-2023, monthly count differences are not statistically significant at alpha 0.05.
- OSB exposure concentration: provinces with matched OSB exposure have higher incident counts than provinces without matched OSB exposure, Mann-Whitney p = 1.03e-28.
- Exposure correlation: province incident counts correlate with OSB parcel count, Spearman rho = 0.771, p = 3.81e-15; and OSB area, rho = 0.737, p = 2.24e-13.
- Severity associations: sector vs severity is statistically significant with chi-square p = 1.03e-09. Province-level OSB exposure vs severity is also significant in this run, p = 0.009.

## Machine Learning

Severity classification used temporal, spatial, sector, event type, OSB flags, and OSB exposure features. Models tested were logistic regression, random forest, and XGBoost. Random Forest achieved the best cross-validated macro-F1 in the current run at about 0.525, with test accuracy around 0.877. SHAP output is saved as `figures/14_shap_summary.png`.

## Findings

Raw incident counts are strongly aligned with industrial exposure: provinces with more OSB parcels and larger OSB area report more industrial fire and explosion incidents. This does not prove OSBs are more dangerous; rather, it shows that industrial capacity is a necessary denominator. The exposure-adjusted view helps distinguish provinces with high raw counts because they host more industry from provinces with unusually high incident intensity per OSB parcel.

## Limitations and Future Work

The source is compiled from public reports, so it should not be treated as a complete official registry. Many ignition and formation causes are unknown. OSBÜK exposure is static/current and not reconstructed year by year. The 2024 appendix needs either a reliable OCR workflow or a manually verified table export. Future work should add facility coordinates, official fire department response data, historical OSB capacity, district-level industrial employment, and satellite night-light measures.
