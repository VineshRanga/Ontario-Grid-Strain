"""
Apply AI load scenarios to forecasts, compute strain metrics (peak + high-hours), save results.
"""

import argparse
import json
from pathlib import Path
from typing import List, Dict

import numpy as np
import pandas as pd
import pyarrow

from src.scenarios.ai_load_scenarios import (
    yearly_mw_series,
    hourly_ai_adder,
    get_default_scenarios,
    load_scenarios_config,
)


def load_baseline_forecast(path: str) -> pd.DataFrame:
    """
    Load baseline forecast from parquet or CSV.

    Args:
        path: Path to forecast file

    Returns:
        DataFrame with timestamp and y_hat columns

    Raises:
        ValueError: If required columns are missing
    """
    path_obj = Path(path)
    if not path_obj.exists():
        raise FileNotFoundError(f"Baseline forecast not found: {path}")

    if path_obj.suffix == ".parquet":
        df = pd.read_parquet(path)
    elif path_obj.suffix == ".csv":
        df = pd.read_csv(path, parse_dates=["timestamp"])
    else:
        raise ValueError(f"Unsupported file format: {path_obj.suffix}")

    # Validate required columns
    if "timestamp" not in df.columns:
        raise ValueError("Baseline forecast must contain 'timestamp' column")
    if "y_hat" not in df.columns:
        raise ValueError("Baseline forecast must contain 'y_hat' column")

    # Ensure timestamp is datetime
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    return df


def make_synthetic_baseline(
    year: int, template_path: str = "data/processed/preds_mlp_val.parquet"
) -> pd.DataFrame:
    """
    Create synthetic baseline forecast for a target year using template data.

    Args:
        year: Target year
        template_path: Path to template predictions (e.g., 2024 val data)

    Returns:
        DataFrame with timestamp, y_hat, and baseline_is_synthetic=True
    """
    # Load template
    template = pd.read_parquet(template_path)
    template["timestamp"] = pd.to_datetime(template["timestamp"])

    # Create hourly timestamps for target year
    start_ts = pd.Timestamp(f"{year}-01-01 00:00:00")
    end_ts = pd.Timestamp(f"{year}-12-31 23:00:00")
    target_index = pd.date_range(start=start_ts, end=end_ts, freq="h")

    # Map each timestamp to same month/day/hour in template year
    # Template year is the year from template data
    template_year = template["timestamp"].dt.year.iloc[0]

    y_hat_values = []
    for ts in target_index:
        # Find matching timestamp in template (same month, day, hour)
        # Handle Feb 29: use Feb 28 if target is non-leap year
        try:
            template_ts = pd.Timestamp(
                year=template_year,
                month=ts.month,
                day=ts.day,
                hour=ts.hour,
            )
        except ValueError:
            # Feb 29 in non-leap year template
            if ts.month == 2 and ts.day == 29:
                template_ts = pd.Timestamp(
                    year=template_year,
                    month=2,
                    day=28,
                    hour=ts.hour,
                )
            else:
                # Use nearest day
                template_ts = pd.Timestamp(
                    year=template_year,
                    month=ts.month,
                    day=min(ts.day, 28),  # Safe fallback
                    hour=ts.hour,
                )

        # Find matching row in template
        matches = template[template["timestamp"] == template_ts]
        if len(matches) > 0:
            y_hat_values.append(matches["y_hat"].iloc[0])
        else:
            # Fallback: use mean of same hour across template
            hour_mean = template[template["timestamp"].dt.hour == ts.hour]["y_hat"].mean()
            y_hat_values.append(hour_mean if not pd.isna(hour_mean) else 0.0)

    result = pd.DataFrame(
        {
            "timestamp": target_index,
            "y_hat": y_hat_values,
            "baseline_is_synthetic": True,
        }
    )

    return result


def run_scenarios(
    df_baseline: pd.DataFrame, scenarios: List[Dict], ramp_shape: str = "flat", debug: bool = False
) -> pd.DataFrame:
    """
    Apply AI load scenarios to baseline forecast.

    Args:
        df_baseline: DataFrame with timestamp and y_hat columns
        scenarios: List of scenario dictionaries
        ramp_shape: Ramp shape for hourly adders

    Returns:
        Long-form DataFrame with columns:
        - timestamp
        - scenario
        - y_hat
        - ai_adder_mw
        - y_hat_with_ai
    """
    results = []

    # Ensure timestamp is datetime
    if "timestamp" not in df_baseline.columns:
        raise ValueError("df_baseline must have 'timestamp' column")
    df_baseline = df_baseline.copy()
    df_baseline["timestamp"] = pd.to_datetime(df_baseline["timestamp"])

    # Determine forecast year range directly from df_baseline
    years_in_df = sorted(df_baseline["timestamp"].dt.year.unique())
    y0, y1 = years_in_df[0], years_in_df[-1]

    # Add explicit baseline scenario (ai_adder_mw=0.0)
    baseline_result = pd.DataFrame(
        {
            "timestamp": df_baseline["timestamp"],
            "scenario": "baseline",
            "y_hat": df_baseline["y_hat"].values,
            "ai_adder_mw": 0.0,
            "y_hat_with_ai": df_baseline["y_hat"].values,
        }
    )
    results.append(baseline_result)

    # Process each scenario
    for scenario in scenarios:
        # Debug: print full scenario dict
        if debug:
            print(f"\n[DEBUG] Processing scenario:")
            print(f"  Full scenario dict: {scenario}")
        
        # Extract required keys explicitly (not .get)
        required_keys = ["name", "start_mw", "yoy_growth"]
        missing_keys = [k for k in required_keys if k not in scenario]
        if missing_keys:
            raise KeyError(
                f"Missing required keys in scenario: {missing_keys}. "
                f"Required keys: {required_keys}"
            )
        
        name = scenario["name"]
        start_mw = float(scenario["start_mw"])
        yoy_growth = float(scenario["yoy_growth"])
        
        # Handle year keys robustly
        start_year = int(scenario.get("start_year", y0))
        end_year = int(scenario.get("end_year", y1))
        
        if debug:
            print(f"  Extracted: name={name}, start_year={start_year}, end_year={end_year}, "
                  f"start_mw={start_mw}, yoy_growth={yoy_growth}")

        # Validate scenario year range overlaps with baseline
        if end_year < y0 or start_year > y1:
            raise RuntimeError(
                f"Scenario '{name}' year range {start_year}-{end_year} does not overlap "
                f"baseline years {y0}-{y1}"
            )

        # Generate yearly series
        yearly_mw = yearly_mw_series(start_year, end_year, start_mw, yoy_growth)
        
        # Debug: print yearly_mw info
        if debug:
            print(f"  [DEBUG] After yearly_mw_series:")
            print(f"    yearly_mw: {repr(yearly_mw)}")
            print(f"    yearly_mw.index dtype: {yearly_mw.index.dtype}")
            if start_year in yearly_mw.index:
                print(f"    yearly_mw.loc[{start_year}]: {yearly_mw.loc[start_year]}")

        # Generate hourly adders
        # Convert timestamp Series to DatetimeIndex
        timestamp_index = pd.DatetimeIndex(df_baseline["timestamp"])
        
        ai_adder = hourly_ai_adder(
            timestamp_index, yearly_mw, ramp_shape=ramp_shape, allow_missing_years=False
        )
        
        # Debug: print ai_adder stats
        if debug:
            print(f"  [DEBUG] After hourly_ai_adder:")
            print(f"    ai_adder_mw.describe():\n{ai_adder.describe()}")
            print(f"    ai_adder_mw.nunique(): {ai_adder.nunique()}")
            print(f"    ai_adder_mw.min(): {ai_adder.min()}, max(): {ai_adder.max()}, mean(): {ai_adder.mean()}")

        # Build result for this scenario
        # Start from df_baseline and add columns directly
        df_s = df_baseline[["timestamp", "y_hat"]].copy()
        df_s["scenario"] = name
        df_s["ai_adder_mw"] = ai_adder.to_numpy()
        df_s["y_hat_with_ai"] = df_s["y_hat"] + df_s["ai_adder_mw"]

        # Sanity assertion: ai_adder_mw should be non-zero for non-baseline scenarios
        # Check immediately after building df_s
        if name != "baseline":
            if float(df_s["ai_adder_mw"].max()) <= 0:
                raise RuntimeError(
                    f"ai_adder_mw is zero for scenario '{name}'; check start_year/end_year mapping. "
                    f"Scenario range: {start_year}-{end_year}, Baseline range: {y0}-{y1}"
                )

        results.append(df_s)

    # Concatenate all scenarios
    return pd.concat(results, ignore_index=True)


def quarterly_strain_from_forecast(
    df_long: pd.DataFrame, threshold: float
) -> pd.DataFrame:
    """
    Compute quarterly strain metrics from scenario forecasts.

    Args:
        df_long: Long-form DataFrame from run_scenarios()
        threshold: High-demand threshold (MW)

    Returns:
        DataFrame with quarterly strain metrics
    """
    # Ensure timestamp is datetime
    df = df_long.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    # Add year and quarter
    df["year"] = df["timestamp"].dt.year
    df["quarter"] = df["timestamp"].dt.quarter

    # Compute baseline metrics (y_hat without AI load) for each quarter
    # y_hat is the same for all scenarios, so use first scenario
    baseline_metrics = {}
    first_scenario = df["scenario"].iloc[0]
    baseline_df = df[df["scenario"] == first_scenario]
    for (year, quarter), group in baseline_df.groupby(["year", "quarter"]):
        baseline_metrics[(year, quarter)] = {
            "peak": group["y_hat"].max(),
            "high_hours": (group["y_hat"] > threshold).sum(),
        }

    # Group by scenario, year, quarter
    results = []
    for (scenario, year, quarter), group in df.groupby(["scenario", "year", "quarter"]):
        predicted_peak = group["y_hat_with_ai"].max()
        predicted_high_hours = (group["y_hat_with_ai"] > threshold).sum()

        # Get baseline for this quarter
        baseline = baseline_metrics.get((year, quarter), {"peak": 0.0, "high_hours": 0})
        baseline_peak = baseline["peak"]
        baseline_high_hours = baseline["high_hours"]

        peak_delta = predicted_peak - baseline_peak
        high_hours_delta = predicted_high_hours - baseline_high_hours

        results.append(
            {
                "year": year,
                "quarter": quarter,
                "scenario": scenario,
                "predicted_peak_mw": predicted_peak,
                "predicted_high_hours": predicted_high_hours,
                "threshold_used": threshold,
                "peak_delta_mw": peak_delta,
                "high_hours_delta": high_hours_delta,
            }
        )

    return pd.DataFrame(results)


def compute_baseline_strain(df_long: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """
    Compute baseline strain (no AI load) for comparison.

    Args:
        df_long: Long-form DataFrame from run_scenarios()
        threshold: High-demand threshold

    Returns:
        DataFrame with baseline quarterly metrics
    """
    # Get first scenario's y_hat as baseline (y_hat is same for all scenarios)
    first_scenario = df_long["scenario"].iloc[0]
    baseline = df_long[df_long["scenario"] == first_scenario].copy()
    baseline["timestamp"] = pd.to_datetime(baseline["timestamp"])
    baseline["year"] = baseline["timestamp"].dt.year
    baseline["quarter"] = baseline["timestamp"].dt.quarter

    results = []
    for (year, quarter), group in baseline.groupby(["year", "quarter"]):
        peak = group["y_hat"].max()
        high_hours = (group["y_hat"] > threshold).sum()

        results.append(
            {
                "year": year,
                "quarter": quarter,
                "scenario": "baseline",
                "predicted_peak_mw": peak,
                "predicted_high_hours": high_hours,
                "threshold_used": threshold,
                "peak_delta_mw": 0.0,
                "high_hours_delta": 0,
            }
        )

    return pd.DataFrame(results)


def main(argv=None) -> None:
    """CLI entry point for scenario runner."""
    parser = argparse.ArgumentParser(
        description="Run AI load scenarios on baseline forecasts"
    )
    parser.add_argument(
        "--baseline-path",
        default=None,
        help="Path to baseline forecast file (parquet or CSV)",
    )
    parser.add_argument(
        "--synthetic-year",
        type=int,
        default=None,
        help="Generate synthetic baseline for this year (if baseline-path not provided)",
    )
    parser.add_argument(
        "--scenarios-config",
        default=None,
        help="Path to scenarios JSON config (uses defaults if not provided)",
    )
    parser.add_argument(
        "--ramp-shape",
        default="flat",
        choices=["flat", "linear_within_year", "step_quarterly"],
        help="Ramp shape for AI load adders",
    )
    parser.add_argument(
        "--out-forecast",
        default="data/processed/scenario_forecasts.parquet",
        help="Output path for scenario forecasts",
    )
    parser.add_argument(
        "--out-strain",
        default="reports/scenario_strain_quarterly.csv",
        help="Output path for quarterly strain metrics",
    )
    parser.add_argument(
        "--threshold-dataset",
        default="data/processed/model_dataset_hourly_2019_2025.parquet",
        help="Path to dataset for computing threshold",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug output for scenario processing",
    )

    args = parser.parse_args(argv)

    # Load or create baseline forecast
    if args.baseline_path:
        print(f"Loading baseline forecast from {args.baseline_path}...")
        df_baseline = load_baseline_forecast(args.baseline_path)
        print(f"  Loaded {len(df_baseline):,} rows")
    elif args.synthetic_year:
        print(f"Generating synthetic baseline for year {args.synthetic_year}...")
        df_baseline = make_synthetic_baseline(args.synthetic_year)
        print(f"  Generated {len(df_baseline):,} rows (synthetic placeholder)")
    else:
        raise ValueError(
            "Must provide either --baseline-path or --synthetic-year"
        )

    # Load scenarios
    if args.scenarios_config:
        print(f"\nLoading scenarios from {args.scenarios_config}...")
        scenarios = load_scenarios_config(args.scenarios_config)
    else:
        print("\nUsing default scenarios...")
        scenarios = get_default_scenarios()

    print(f"  Scenarios: {[s['name'] for s in scenarios]}")

    # Compute threshold from training data
    print(f"\nComputing threshold from {args.threshold_dataset}...")
    df_threshold = pd.read_parquet(args.threshold_dataset)
    train_y = df_threshold[df_threshold["split"] == "train"]["y"].dropna()
    threshold = np.percentile(train_y, 95)
    print(f"  Threshold (95th percentile of train): {threshold:,.0f} MW")

    # Run scenarios
    print(f"\nRunning scenarios with ramp_shape='{args.ramp_shape}'...")
    df_scenarios = run_scenarios(df_baseline, scenarios, ramp_shape=args.ramp_shape, debug=args.debug)
    print(f"  Generated {len(df_scenarios):,} scenario-hour rows")

    # Save forecasts
    out_forecast_path = Path(args.out_forecast)
    out_forecast_path.parent.mkdir(parents=True, exist_ok=True)
    df_scenarios.to_parquet(out_forecast_path, engine="pyarrow", index=False)
    print(f"\nSaved scenario forecasts to {out_forecast_path}")

    # Compute quarterly strain
    print("\nComputing quarterly strain metrics...")
    strain_scenarios = quarterly_strain_from_forecast(df_scenarios, threshold)
    strain_baseline = compute_baseline_strain(df_scenarios, threshold)

    # Combine baseline and scenarios
    strain_all = pd.concat([strain_baseline, strain_scenarios], ignore_index=True)

    # Save strain metrics
    out_strain_path = Path(args.out_strain)
    out_strain_path.parent.mkdir(parents=True, exist_ok=True)
    strain_all.to_csv(out_strain_path, index=False)
    print(f"Saved strain metrics to {out_strain_path}")

    # Print summary
    print("\n" + "=" * 60)
    print("SCENARIO SUMMARY")
    print("=" * 60)
    print(f"\nThreshold: {threshold:,.0f} MW")
    print(f"Scenarios: {[s['name'] for s in scenarios]}")
    print(f"\nOutput files:")
    print(f"  Forecasts: {out_forecast_path}")
    print(f"  Strain: {out_strain_path}")

    # Quick summary table: predicted_peak_mw by scenario for Q3 and Q1
    print("\n" + "=" * 60)
    print("QUARTERLY PEAK DEMAND SUMMARY")
    print("=" * 60)

    # Get unique years
    years = sorted(strain_all["year"].unique())
    
    # Build ordered scenario list (baseline first, then others, deduped)
    scenario_order = ["baseline"]
    for s in scenarios:
        name = s["name"]
        if name not in scenario_order:
            scenario_order.append(name)
    
    for year in years:
        year_data = strain_all[strain_all["year"] == year]
        print(f"\n{year}:")
        for quarter in [1, 3]:  # Q1 (winter) and Q3 (summer)
            q_data = year_data[year_data["quarter"] == quarter]
            if len(q_data) > 0:
                print(f"  Q{quarter}:")
                
                # Extract baseline row first
                base_rows = q_data[q_data["scenario"] == "baseline"]
                if len(base_rows) > 0:
                    base_row = base_rows.iloc[0]
                    base_peak = base_row["predicted_peak_mw"]
                    base_high = base_row["predicted_high_hours"]
                else:
                    base_peak = None
                    base_high = None
                
                # Print in scenario_order, ensuring each scenario appears once
                seen_scenarios = set()
                for scenario_name in scenario_order:
                    scenario_rows = q_data[q_data["scenario"] == scenario_name]
                    if len(scenario_rows) > 0 and scenario_name not in seen_scenarios:
                        row = scenario_rows.iloc[0]
                        peak = row["predicted_peak_mw"]
                        high = row["predicted_high_hours"]
                        
                        if scenario_name == "baseline":
                            # Baseline: print as-is (no deltas)
                            print(
                                f"    {row['scenario']:15s}  Peak: {peak:,.0f} MW  "
                                f"High hours: {high:,.0f}"
                            )
                        else:
                            # Non-baseline: calculate and print deltas
                            if base_peak is not None and base_high is not None:
                                peak_delta = peak - base_peak
                                high_delta = high - base_high
                                # Format deltas with sign
                                peak_delta_str = f"({peak_delta:+.0f})" if peak_delta != 0 else "(+0)"
                                high_delta_str = f"({high_delta:+.0f})" if high_delta != 0 else "(+0)"
                                print(
                                    f"    {row['scenario']:15s}  Peak: {peak:,.0f} MW {peak_delta_str:>8s}  "
                                    f"High hours: {high:,.0f} {high_delta_str:>8s}"
                                )
                            else:
                                # Fallback if baseline not found
                                print(
                                    f"    {row['scenario']:15s}  Peak: {peak:,.0f} MW  "
                                    f"High hours: {high:,.0f}"
                                )
                        seen_scenarios.add(scenario_name)

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()

