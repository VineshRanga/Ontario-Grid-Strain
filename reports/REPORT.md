# Ontario Grid Strain Scenario Engine - Analysis Report

## Overview

This report summarizes baseline model performance, neural network forecasts, and scenario analysis for Ontario electricity demand forecasting with incremental AI/data center load projections.

## Data Window

- **Historical data**: 2019-2025 (training: 2019-2023, validation: 2024, test: 2025 partial)

- **Forecast horizon**: 2026-2027 (scenario analysis)

## Model Performance Summary

- **Best test RMSE**: MLPRegressor (376 MW)

- **Best validation RMSE**: MLPRegressor (278 MW)


## Strain Proxy Definition

- **Peak demand**: Maximum hourly demand within each quarter

- **High-demand hours**: Count of hours above 95th percentile threshold (computed from training data: 2019-2023)


## Scenario Summary

- **Maximum peak uplift**: high scenario, 2026Q1 (+800 MW)

- **Maximum high-demand hours increase**: high scenario, 2026Q3 (+194 hours)


## Limitations

- **Demand-only proxy**: This analysis focuses on demand forecasting and does not model transmission constraints, generation capacity, or grid stability.

- **No transmission constraints**: Peak demand increases do not account for regional transmission bottlenecks or local capacity limits.

- **Scenario-based AI load**: AI/data center load projections are exogenous scenarios, not learned from historical patterns.


## Generated Artifacts

### Metrics CSVs

- `baseline_metrics.csv` - Baseline model performance metrics

- `nn_metrics.csv` - Neural network performance metrics

- `strain_quarterly_*.csv` - Quarterly strain metrics by model

- `scenario_strain_quarterly.csv` - Scenario analysis results


### Figures

- `model_rmse_comparison.png` - Model RMSE comparison

- `*_peak_window_actual_vs_pred.png` - Time series around peak days

- `*_monthly_peak_scatter.png` - Monthly peak scatter plots

- `scenario_quarterly_peak.png` - Scenario peak demand by quarter

- `scenario_quarterly_high_hours_delta.png` - Scenario high-demand hours delta
