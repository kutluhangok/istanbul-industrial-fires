# Project Proposal: Spatiotemporal Dynamics of Industrial Fire Incidents in Istanbul

**Course:** DSA 210 — Introduction to Data Science  
**Term:** Spring 2026, Sabancı University

---

## Overview

This project investigates the temporal and spatial patterns of fire incidents in Istanbul using multi-source data analysis. The primary data source is fire incident records from the Istanbul Fire Department (Istanbul İtfaiyesi), including incident-level reports with fields such as date, time, district, building/property type, fire cause category, and casualty information.

## Data Sources and Enrichment

The fire incident dataset is enriched with three external sources:

1. **Meteorological data** — Daily temperature, humidity, wind speed, and precipitation from the Open-Meteo archive API (2017–2024)
2. **Demographic and building data** — District-level population, density, and building stock characteristics from TUIK and IBB Open Data Portal
3. **Industrial zone geographies** — Organized Industrial Zone (OSB) locations and boundaries from the Istanbul OSB portal and OpenStreetMap

The enriched dataset is expected to contain several thousand fire incident records spanning multiple years, with approximately 15–20 features after engineering (temporal, spatial, environmental, and structural variables).

## Methodology

The analysis proceeds in four stages:

1. **Exploratory Data Analysis** — Geospatial visualization, temporal trend analysis, and missing data assessment
2. **Hypothesis Testing** — Kruskal-Wallis test for seasonal patterns, Moran's I for spatial clustering, negative binomial regression for weather-fire associations
3. **Machine Learning** — Fire severity classification using logistic regression, random forest, and XGBoost, with SHAP-based feature importance analysis
4. **Time Series Forecasting** *(extension)* — Short-term fire incident count prediction

## Objective

The project aims to identify high-risk temporal windows, spatial hotspots, and environmental co-factors for fire incidents in Istanbul, with findings relevant to fire prevention resource allocation and inspection prioritization.
