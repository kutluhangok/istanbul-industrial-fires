# Istanbul Industrial Fires

This project investigates the temporal, spatial, and environmental patterns of industrial fire incidents in Istanbul using a multi-source data analysis approach.

## Research Question
Are industrial fire incidents in Istanbul temporally and spatially clustered, and can their frequency and severity be explained by environmental, temporal, and district-level factors?

## Data Strategy
This project follows a dual-track data strategy:
- Track A: incident-level fire reports if accessible
- Track B: aggregated fire statistics from IBB Open Data and Istanbul Fire Department publications

## Planned Workflow
1. Collect fire incident data
2. Clean and standardize datasets
3. Enrich with weather, district, and industrial zone data
4. Perform exploratory data analysis
5. Test temporal, spatial, and environmental hypotheses
6. Apply ML models if feasible
7. Produce final report and visual outputs

## Repository Structure
- `data/raw/`: untouched source files
- `data/processed/`: cleaned and merged data
- `data/enrichment/`: weather, demographics, geo data
- `notebooks/`: analysis notebooks
- `src/`: reusable Python code
- `reports/`: proposal, notes, final report
- `figures/`: plots and maps
