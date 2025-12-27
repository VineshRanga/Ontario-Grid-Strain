# Glossary

Plain English definitions of terms used in this project.

## Data Terms

**IESO**: Independent Electricity System Operator. They publish hourly demand data.

**ECCC**: Environment and Climate Change Canada. They publish hourly weather data.

**Toronto Pearson**: Weather station ID 51459. We use this for all weather features.

**Ontario demand**: Total provincial electricity demand in MW.

**Market demand**: Demand that's actually traded in the market. Usually close to Ontario demand.

## Model Terms

**Baseline model**: Simple model used as a comparison point. SeasonalNaive168, LinearWeatherRidge, LagRidge.

**Neural network**: MLPRegressor from scikit-learn. Multi-layer perceptron. Best performing model.

**Training**: Fitting models on 2019-2023 data.

**Validation**: Testing on 2024 data to pick hyperparameters.

**Test**: Final evaluation on 2025 data. Never used for training decisions.

**Overfitting**: Model memorizes training data but fails on new data. We avoid this with train/val/test splits.

## Feature Terms

**HDD**: Heating degree days. How much colder than 18°C. Drives heating load.

**CDD**: Cooling degree days. How much hotter than 18°C. Drives AC load.

**Lag features**: Previous values. Lag1 is last hour, Lag24 is same hour yesterday, Lag168 is same hour last week.

**Rolling mean**: Average of recent hours. 24h rolling mean is last day's average, 168h is last week's average.

**Time features**: Hour of day, day of week, month, year. Also cyclical encodings (sin/cos) for hour and day of year.

**Trend**: Long-term growth. We use "years since 2019" as a feature.

## Metric Terms

**MAE**: Mean absolute error. Average prediction error.

**RMSE**: Root mean squared error. Bigger errors get penalized more.

**Monthly Peak MAE**: Error on the highest demand hour of each month.

**95th percentile**: The value that 95% of data points fall below. We use training demand's 95th percentile as a threshold.

**High-demand hours**: Hours where demand exceeds the 95th percentile threshold.

## Scenario Terms

**Baseline scenario**: No AI load added. Just the forecast.

**Low scenario**: Adds 200 MW flat across the year.

**Base scenario**: Adds 350 MW flat across the year.

**High scenario**: Adds 800 MW flat across the year.

**Synthetic baseline**: 2026 forecast built by copying 2024's seasonal pattern. Not real 2026 weather.

**Ramp shape**: How AI load changes within a year. "Flat" means constant. "Linear" means gradual increase. "Step quarterly" means jumps each quarter.

## Output Terms

**Peak demand**: Highest hourly demand in a quarter.

**Peak delta**: How much higher peak gets vs baseline.

**High-hours delta**: How many more hours cross the threshold vs baseline.

**Strain metrics**: Peak and high-hours counts by quarter, by scenario.

