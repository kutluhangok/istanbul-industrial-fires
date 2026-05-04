# Project Proposal: Spatiotemporal Dynamics of Industrial Fire Incidents in Istanbul

**Course:** DSA 210 — Introduction to Data Science  
**Term:** Spring 2026, Sabancı University

---

## Overview

This project investigates the temporal and spatial patterns of fire incidents in Istanbul using multi-source data analysis. The primary data source is fire incident records from the Istanbul Fire Department (Istanbul İtfaiyesi), including incident-level reports with fields such as date, time, district, building/property type, fire cause category, and casualty information.

## Data Sources and Enrichment

The IBB Open Data Portal currently provides verified aggregate records covering 2017–2019 (3 years). Combined with the 2020–2024 statistical reports, the guaranteed open-source baseline covers 8 years (2017–2024) of monthly district-level fire counts across Istanbul's 39 districts, yielding a panel dataset of approximately 39 districts × 96 months = 3,744 district-month observations.
If the institutional request to Istanbul Itfaiyesi is successful, the dataset will be upgraded to incident-level records. Based on Istanbul Itfaiyesi's published statistics, the city records approximately 22,000–25,000 fire incidents per year across all categories. Filtering for structural and industrial fires is estimated to yield roughly 8,000–12,000 incident-level records per year, or 64,000–96,000 records over 8 years. This estimate will be confirmed and revised once the data request receives a response. The analysis is designed to be valid under either scenario; the aggregate panel dataset constitutes the minimum viable dataset, while incident-level records enable the full machine learning pipeline described in the methodology.
## Methodology

The analysis proceeds in four stages:

1. **Exploratory Data Analysis** — Geospatial visualization, temporal trend analysis, and missing data assessment
2. **Hypothesis Testing** — Kruskal-Wallis test for seasonal patterns, Moran's I for spatial clustering, negative binomial regression for weather-fire associations
3. **Machine Learning** — Fire severity classification using logistic regression, random forest, and XGBoost, with SHAP-based feature importance analysis
4. **Time Series Forecasting** *(extension)* — Short-term fire incident count prediction

## Objective

The project aims to identify high-risk temporal windows, spatial hotspots, and environmental co-factors for fire incidents in Istanbul, with findings relevant to fire prevention resource allocation and inspection prioritization.
