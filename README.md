# Ontario Grid Strain Scenario Engine

Forecasts Ontario electricity demand using weather and seasonality. Adds AI/data center load scenarios on top to see how grid stress changes. Uses IESO demand data and ECCC Toronto Pearson weather from 2019-2025. Outputs backtest results, scenario tables, figures, and a markdown report.

Grid strain here means two things: quarterly peak demand and the count of "high-demand hours" above a threshold. The threshold is the 95th percentile of training demand. More high-demand hours means more time the grid is under stress.

2026 forecasts are synthetic for now. We build the 2026 calendar from 2024's seasonal profile since we don't have real 2026 weather yet.

## Definitions

See [docs/GLOSSARY.md](docs/GLOSSARY.md) for full definitions. Quick reference:

**MW**: Megawatts. Ontario demand is total provincial load. Market demand is what's actually traded.

**Train/Val/Test split**: 2019-2023 is training, 2024 is validation, 2025 is test. We never look at future data when training.

**95th percentile threshold**: The demand level that 95% of training hours stayed below. We use this to count "high-demand hours" in forecasts. It's a simple way to flag stress periods.

**MAE**: Mean absolute error. Average prediction error in MW.

**RMSE**: Root mean squared error. Penalizes big errors more than MAE.

**Monthly Peak MAE**: How wrong we are on the highest demand hour of each month. Useful for planning since peaks matter most.

**High-demand hours**: Hours where demand exceeds the 95th percentile threshold. More of these means more stress.

**HDD/CDD**: Heating degree days and cooling degree days. HDD is how much colder it is than 18°C (heating load). CDD is how much hotter (cooling load). Base temp is 18°C.

**Lag1 / Lag24 / Lag168**: Previous hour, same hour yesterday, same hour last week. Demand has strong patterns.

**Rolling mean**: Average of recent hours. We use 24h and 168h rolling means as features.

**SeasonalNaive168**: Baseline that just predicts "same hour last week". Simple but often works.

**LinearWeatherRidge**: Ridge regression with weather features (HDD/CDD) and time features (hour, day of week, etc). Includes a trend term.

**LagRidge**: Ridge regression using lag features (previous hour, yesterday, last week, rolling means).

**MLPRegressor**: Neural network from scikit-learn. Best model so far.

## What the forecasts are telling us

The neural network gets test RMSE around 376 MW. That's about 1.5% error on typical demand.

Scenarios add MW on top of baseline forecasts. Low adds 200 MW, base adds 350 MW, high adds 800 MW. These are flat across the year for now.

Example from 2026: Baseline Q1 peak might be 21,300 MW with 113 high-demand hours. Low scenario adds 200 MW (21,500 MW peak, +29 high hours). Base adds 350 MW (21,650 MW peak, +64 high hours). High adds 800 MW (22,100 MW peak, +188 high hours).

For planning, you care about two things: peak uplift (how much higher the max gets) and how many more hours cross the threshold. Both matter. Peak tells you if you need more capacity. High-hours tells you how often you'll be stressed.

## Charts (what they show + why they matter)

### Model RMSE Comparison

![Model RMSE Comparison](reports/figures/model_rmse_comparison.png)

- **What it is**: Bar chart comparing test RMSE across all models
- **Why it matters**: Shows which model is most accurate. Lower RMSE is better
- **What it's saying**: MLPRegressor wins at 376 MW. LagRidge is second at 498 MW. LinearWeatherRidge and SeasonalNaive168 are much worse (1297 MW and 1683 MW). The neural network learns patterns the simple models miss

### Peak Window: Actual vs Predicted (Validation)

![Val Peak Window](reports/figures/val_peak_window_actual_vs_pred.png)

- **What it is**: 14-day time series around the highest demand day in validation set. Shows actual vs predicted hour by hour
- **Why it matters**: Peaks are what break grids. If we miss peak timing or magnitude, that's a problem
- **What it's saying**: Model tracks the peak day well. Predictions follow actual demand patterns. Small errors during peak hours, but captures the overall shape

### Peak Window: Actual vs Predicted (Test)

![Test Peak Window](reports/figures/test_peak_window_actual_vs_pred.png)

- **What it is**: Same as validation, but for test set (2025 data the model never saw during training)
- **Why it matters**: Test performance is what matters. If it works on test, it should work on 2026
- **What it's saying**: Model still tracks peaks well on unseen data. No major overfitting. Ready for forecasting

### Monthly Peak Scatter (Validation)

![Val Monthly Peak Scatter](reports/figures/val_monthly_peak_scatter.png)

- **What it is**: Scatter plot. Each dot is one month's peak demand. X-axis is actual, Y-axis is predicted
- **Why it matters**: Monthly peaks drive capacity planning. If dots are close to the diagonal line, peaks are predicted well
- **What it's saying**: Most dots cluster near the y=x line. Some months are slightly overpredicted (dots above line) or underpredicted (dots below line), but overall the model captures monthly peaks

**How to read scatter plots**: Dots close to the y=x line mean peaks are predicted well. Dots above the line mean overprediction (model says peak is higher than it was). Dots below the line mean underprediction (model says peak is lower than it was)

### Monthly Peak Scatter (Test)

![Test Monthly Peak Scatter](reports/figures/test_monthly_peak_scatter.png)

- **What it is**: Same scatter plot for test set (2025)
- **Why it matters**: Shows if monthly peak accuracy holds on new data
- **What it's saying**: Dots still cluster near the line. Model generalizes to new months. Good sign for 2026 forecasts

### Scenario Quarterly Peak

![Scenario Quarterly Peak](reports/figures/scenario_quarterly_peak.png)

- **What it is**: Bar chart showing peak demand by quarter for each scenario (baseline, low, base, high)
- **Why it matters**: Shows how much peak demand increases under each AI load scenario
- **What it's saying**: 2026 Q1 baseline peak is 21,305 MW. Low adds 200 MW (21,505 MW). Base adds 350 MW (21,655 MW). High adds 800 MW (22,105 MW). Q3 baseline is 23,475 MW, scenarios add the same amounts. Summer peaks (Q3) are already higher, so absolute increases are the same but relative impact is smaller

### Scenario High-Demand Hours Delta

![Scenario High Hours Delta](reports/figures/scenario_quarterly_high_hours_delta.png)

- **What it is**: Bar chart showing how many more hours cross the threshold under each scenario vs baseline
- **Why it matters**: Peak tells you about capacity. High-hours tells you about frequency of stress. Both matter for planning
- **What it's saying**: 2026 Q1 baseline has 113 high-demand hours. Low adds 29 hours (142 total). Base adds 64 hours (177 total). High adds 188 hours (301 total). Q3 baseline has 411 high-hours, scenarios add 43/77/194 hours. Summer already has more high-hours, so the deltas are smaller but absolute counts get high (605 hours in high scenario Q3)

## How to run it

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m src all --synthetic-year 2026
```

Outputs land in:
- `reports/REPORT.md` - Main report
- `reports/figures/*.png` - Plots
- `artifacts/mlp/*.joblib` - Trained models
- `data/processed/*.parquet` - Clean datasets

## Pre-commit checks

Before committing, make sure you're not staging huge files or generated artifacts:

```bash
# See what's staged
git status
git diff --cached --stat

# Check for problematic files (should return nothing)
git ls-files | grep -E "(\.parquet|data/raw|data/processed|\.venv|\.env)" || true
```

If you accidentally tracked something that should be ignored:

```bash
# Untrack a file (keeps it on disk, just removes from git)
git rm -r --cached <path>
git commit -m "Stop tracking generated data"
```

Common things to untrack: `data/raw/`, `data/processed/`, `artifacts/`, `reports/figures/`, `reports/*.csv`, `reports/REPORT.md`, any `.parquet` files.

## Repo map

- `src/ingest/` - Load IESO demand CSVs and ECCC weather CSVs, clean them, save parquets
- `src/features/` - Merge demand + weather, add HDD/CDD, time features, lags
- `src/models/` - Baseline models (SeasonalNaive, Ridge variants) and neural network
- `src/backtest/` - Evaluate models, compute strain metrics, generate plots and report
- `src/scenarios/` - Apply AI load scenarios to forecasts, compute scenario strain
- `src/pipeline.py` - Orchestrates everything
- `src/main.py` - CLI router

## Limitations

This is a demand + weather proxy. It doesn't model power flow, transmission constraints, or generation capacity. It just says "if demand goes up this much, here's what peak and high-hours look like."

2026 baseline is synthetic. We copy 2024's seasonal pattern. Real 2026 weather will change things.

AI load is scenario-based. We add flat MW amounts. Real AI load will have daily patterns and might correlate with demand.

Doesn't model outages, transmission bottlenecks, or prices. Just demand forecasting.
