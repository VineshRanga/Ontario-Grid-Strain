"""
Generate plots and markdown report. Model comparisons, peak windows, scenario results.
"""

import argparse
import warnings
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow


def ensure_dirs() -> Path:
    """
    Create reports/figures directory if missing.

    Returns:
        Path to figures directory
    """
    figures_dir = Path("reports/figures")
    figures_dir.mkdir(parents=True, exist_ok=True)
    return figures_dir


def safe_read_csv(path: str) -> pd.DataFrame:
    """
    Read CSV file with clear error message if missing.

    Args:
        path: Path to CSV file

    Returns:
        DataFrame

    Raises:
        FileNotFoundError: If file doesn't exist
    """
    path_obj = Path(path)
    if not path_obj.exists():
        raise FileNotFoundError(f"Required file not found: {path}")
    return pd.read_csv(path_obj)


def normalize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize column names: lowercase, replace spaces with underscores.

    Args:
        df: DataFrame

    Returns:
        DataFrame with normalized column names
    """
    df = df.copy()
    df.columns = df.columns.str.lower().str.replace(" ", "_")
    return df


def plot_model_rmse_comparison(
    baseline_path: str,
    nn_path: str,
    output_path: Path,
    prefer_split: str = "test",
) -> None:
    """
    Create bar plot comparing model RMSE.

    Args:
        baseline_path: Path to baseline_metrics.csv
        nn_path: Path to nn_metrics.csv
        output_path: Path to save figure
        prefer_split: Preferred split for comparison ("test" or "val")
    """
    try:
        baseline = safe_read_csv(baseline_path)
        baseline = normalize_column_names(baseline)
    except FileNotFoundError as e:
        warnings.warn(f"Skipping model comparison: {e}")
        return

    try:
        nn = safe_read_csv(nn_path)
        nn = normalize_column_names(nn)
    except FileNotFoundError as e:
        warnings.warn(f"Skipping model comparison: {e}")
        return

    # Find RMSE column
    rmse_col = None
    for col in baseline.columns:
        if "rmse" in col.lower():
            if prefer_split in col.lower():
                rmse_col = col
                break
            elif rmse_col is None:
                rmse_col = col

    if rmse_col is None:
        warnings.warn("Could not find RMSE column in metrics files")
        return

    # Extract model names and RMSE values
    models = []
    rmse_values = []

    # From baseline metrics
    for _, row in baseline.iterrows():
        model_name = row.get("model", f"Model_{len(models)}")
        rmse_val = row.get(rmse_col)
        if pd.notna(rmse_val):
            models.append(model_name)
            rmse_values.append(rmse_val)

    # From NN metrics
    for _, row in nn.iterrows():
        model_name = row.get("model", "MLPRegressor")
        rmse_val = row.get(rmse_col)
        if pd.notna(rmse_val):
            models.append(model_name)
            rmse_values.append(rmse_val)

    if not models:
        warnings.warn("No valid RMSE values found")
        return

    # Create bar plot
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(models, rmse_values)
    ax.set_xlabel("Model")
    ax.set_ylabel(f"RMSE ({prefer_split.upper()}) [MW]")
    ax.set_title("Model RMSE Comparison")
    ax.tick_params(axis="x", rotation=45)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    # Print table
    print(f"\nModel RMSE Comparison ({prefer_split.upper()}):")
    comparison_df = pd.DataFrame({"Model": models, "RMSE": rmse_values})
    comparison_df = comparison_df.sort_values("RMSE")
    print(comparison_df.to_string(index=False))


def plot_peak_window_actual_vs_pred(
    preds_path: str, output_path: Path, split_name: str
) -> None:
    """
    Plot actual vs predicted for 14-day window around peak day.

    Args:
        preds_path: Path to predictions parquet file
        output_path: Path to save figure
        split_name: Name of split (for title)
    """
    try:
        df = pd.read_parquet(preds_path)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.sort_values("timestamp")
    except FileNotFoundError:
        warnings.warn(f"Skipping {split_name} peak window plot: file not found")
        return
    except Exception as e:
        warnings.warn(f"Skipping {split_name} peak window plot: {e}")
        return

    if len(df) < 24:  # Need at least 1 day
        warnings.warn(f"Skipping {split_name} peak window plot: insufficient data")
        return

    # Find peak day (max daily average or max hourly)
    df["date"] = df["timestamp"].dt.date
    daily_max = df.groupby("date")["y"].max()
    peak_date = daily_max.idxmax()

    # Get 14-day window around peak
    peak_ts = pd.Timestamp(peak_date)
    window_start = peak_ts - pd.Timedelta(days=7)
    window_end = peak_ts + pd.Timedelta(days=7)

    window_df = df[
        (df["timestamp"] >= window_start) & (df["timestamp"] <= window_end)
    ].copy()

    if len(window_df) == 0:
        warnings.warn(f"No data in window for {split_name}")
        return

    # Plot
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(
        window_df["timestamp"], 
        window_df["y"], 
        label="Actual demand", 
        linewidth=1.5
    )
    ax.plot(
        window_df["timestamp"], 
        window_df["y_hat"], 
        label="Predicted demand (MLP)", 
        linewidth=1.5
    )
    ax.set_xlabel("Timestamp")
    ax.set_ylabel("Demand (MW)")
    ax.set_title(f"{split_name.upper()}: Actual vs Predicted (14-day window around peak)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_monthly_peak_scatter(
    preds_path: str, output_path: Path, split_name: str
) -> None:
    """
    Create scatter plot of monthly peaks (actual vs predicted).

    Args:
        preds_path: Path to predictions parquet file
        output_path: Path to save figure
        split_name: Name of split (for title)
    """
    try:
        df = pd.read_parquet(preds_path)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.sort_values("timestamp")
    except FileNotFoundError:
        warnings.warn(f"Skipping {split_name} monthly peak scatter: file not found")
        return
    except Exception as e:
        warnings.warn(f"Skipping {split_name} monthly peak scatter: {e}")
        return

    if len(df) == 0:
        warnings.warn(f"No data for {split_name} monthly peak scatter")
        return

    # Group by year-month and compute peaks
    df["year_month"] = df["timestamp"].dt.to_period("M")
    monthly_peaks = df.groupby("year_month").agg(
        {"y": "max", "y_hat": "max"}
    ).reset_index()

    if len(monthly_peaks) == 0:
        warnings.warn(f"No monthly peaks computed for {split_name}")
        return

    # Plot
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(
        monthly_peaks["y"], 
        monthly_peaks["y_hat"], 
        alpha=0.6, 
        s=50,
        label="Monthly peak (model)"
    )

    # Add y=x reference line
    min_val = min(monthly_peaks["y"].min(), monthly_peaks["y_hat"].min())
    max_val = max(monthly_peaks["y"].max(), monthly_peaks["y_hat"].max())
    ax.plot(
        [min_val, max_val], 
        [min_val, max_val], 
        "r--", 
        label="Perfect prediction (y=x)", 
        linewidth=1
    )

    ax.set_xlabel("Actual peak demand (MW)")
    ax.set_ylabel("Predicted peak demand (MW)")
    ax.set_title(f"Monthly Peak Demand: Actual vs Predicted ({split_name.upper()})")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_scenario_quarterly_peak(
    scenario_path: str, output_path: Path
) -> None:
    """
    Plot quarterly peak demand across scenarios.

    Args:
        scenario_path: Path to scenario_strain_quarterly.csv
        output_path: Path to save figure
    """
    try:
        df = safe_read_csv(scenario_path)
        df = normalize_column_names(df)
    except FileNotFoundError as e:
        warnings.warn(f"Skipping scenario quarterly peak plot: {e}")
        return

    if "predicted_peak_mw" not in df.columns:
        warnings.warn("predicted_peak_mw column not found in scenario data")
        return

    # Create quarter label
    if "year" in df.columns and "quarter" in df.columns:
        df["quarter_label"] = (
            df["year"].astype(str) + "Q" + df["quarter"].astype(str)
        )
        df = df.sort_values(["year", "quarter"])
    else:
        warnings.warn("year/quarter columns not found, using index")
        df["quarter_label"] = df.index.astype(str)

    # Plot by scenario
    fig, ax = plt.subplots(figsize=(12, 6))
    scenarios = df["scenario"].unique() if "scenario" in df.columns else ["baseline"]

    for scenario in scenarios:
        scenario_df = df[df["scenario"] == scenario] if "scenario" in df.columns else df
        ax.plot(
            scenario_df["quarter_label"],
            scenario_df["predicted_peak_mw"],
            marker="o",
            label=scenario,
            linewidth=2,
        )

    ax.set_xlabel("Quarter")
    ax.set_ylabel("Predicted Peak Demand [MW]")
    ax.set_title("Scenario Quarterly Peak Demand")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_scenario_high_hours_delta(
    scenario_path: str, output_path: Path
) -> None:
    """
    Plot quarterly high-demand hours delta across scenarios.

    Args:
        scenario_path: Path to scenario_strain_quarterly.csv
        output_path: Path to save figure
    """
    try:
        df = safe_read_csv(scenario_path)
        df = normalize_column_names(df)
    except FileNotFoundError as e:
        warnings.warn(f"Skipping scenario high hours delta plot: {e}")
        return

    if "high_hours_delta" not in df.columns:
        warnings.warn("high_hours_delta column not found in scenario data")
        return

    # Create quarter label
    if "year" in df.columns and "quarter" in df.columns:
        df["quarter_label"] = (
            df["year"].astype(str) + "Q" + df["quarter"].astype(str)
        )
        df = df.sort_values(["year", "quarter"])
    else:
        warnings.warn("year/quarter columns not found, using index")
        df["quarter_label"] = df.index.astype(str)

    # Plot by scenario (exclude baseline)
    fig, ax = plt.subplots(figsize=(12, 6))
    if "scenario" in df.columns:
        scenarios = [s for s in df["scenario"].unique() if s != "baseline"]
    else:
        scenarios = []

    if not scenarios:
        warnings.warn("No scenarios found (excluding baseline)")
        return

    for scenario in scenarios:
        scenario_df = df[df["scenario"] == scenario]
        ax.plot(
            scenario_df["quarter_label"],
            scenario_df["high_hours_delta"],
            marker="o",
            label=scenario,
            linewidth=2,
        )

    ax.set_xlabel("Quarter")
    ax.set_ylabel("High-Demand Hours Delta")
    ax.set_title("Scenario Quarterly High-Demand Hours Delta vs Baseline")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color="k", linestyle="--", linewidth=1, alpha=0.5)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def generate_markdown_report(
    baseline_metrics_path: str,
    nn_metrics_path: str,
    scenario_strain_path: str,
    output_path: Path,
) -> None:
    """
    Generate markdown report summarizing results.

    Args:
        baseline_metrics_path: Path to baseline_metrics.csv
        nn_metrics_path: Path to nn_metrics.csv
        scenario_strain_path: Path to scenario_strain_quarterly.csv
        output_path: Path to save REPORT.md
    """
    report_lines = []

    # Header
    report_lines.append("# Ontario Grid Strain Scenario Engine - Analysis Report\n")
    report_lines.append("## Overview\n")
    report_lines.append(
        "This report summarizes baseline model performance, neural network forecasts, "
        "and scenario analysis for Ontario electricity demand forecasting with "
        "incremental AI/data center load projections.\n"
    )

    # Data window
    report_lines.append("## Data Window\n")
    report_lines.append(
        "- **Historical data**: 2019-2025 (training: 2019-2023, validation: 2024, test: 2025 partial)\n"
    )
    report_lines.append(
        "- **Forecast horizon**: 2026-2027 (scenario analysis)\n"
    )

    # Model performance
    report_lines.append("## Model Performance Summary\n")
    try:
        baseline = safe_read_csv(baseline_metrics_path)
        baseline = normalize_column_names(baseline)
        nn = safe_read_csv(nn_metrics_path)
        nn = normalize_column_names(nn)

        # Find best models
        best_test_rmse = None
        best_test_rmse_model = None
        best_val_rmse = None
        best_val_rmse_model = None

        for _, row in baseline.iterrows():
            model = row.get("model", "Unknown")
            test_rmse = row.get("test_rmse") if "test_rmse" in row else None
            val_rmse = row.get("val_rmse") if "val_rmse" in row else None

            if test_rmse is not None and (
                best_test_rmse is None or test_rmse < best_test_rmse
            ):
                best_test_rmse = test_rmse
                best_test_rmse_model = model

            if val_rmse is not None and (
                best_val_rmse is None or val_rmse < best_val_rmse
            ):
                best_val_rmse = val_rmse
                best_val_rmse_model = model

        for _, row in nn.iterrows():
            model = row.get("model", "MLPRegressor")
            test_rmse = row.get("test_rmse") if "test_rmse" in row else None
            val_rmse = row.get("val_rmse") if "val_rmse" in row else None

            if test_rmse is not None and (
                best_test_rmse is None or test_rmse < best_test_rmse
            ):
                best_test_rmse = test_rmse
                best_test_rmse_model = model

            if val_rmse is not None and (
                best_val_rmse is None or val_rmse < best_val_rmse
            ):
                best_val_rmse = val_rmse
                best_val_rmse_model = model

        if best_test_rmse_model:
            report_lines.append(
                f"- **Best test RMSE**: {best_test_rmse_model} ({best_test_rmse:,.0f} MW)\n"
            )
        if best_val_rmse_model:
            report_lines.append(
                f"- **Best validation RMSE**: {best_val_rmse_model} ({best_val_rmse:,.0f} MW)\n"
            )

    except FileNotFoundError:
        report_lines.append("- Model metrics files not found\n")

    # Strain proxy definition
    report_lines.append("\n## Strain Proxy Definition\n")
    report_lines.append(
        "- **Peak demand**: Maximum hourly demand within each quarter\n"
    )
    report_lines.append(
        "- **High-demand hours**: Count of hours above 95th percentile threshold "
        "(computed from training data: 2019-2023)\n"
    )

    # Scenario summary
    report_lines.append("\n## Scenario Summary\n")
    try:
        scenario_df = safe_read_csv(scenario_strain_path)
        scenario_df = normalize_column_names(scenario_df)

        if "peak_delta_mw" in scenario_df.columns:
            max_peak_delta = scenario_df["peak_delta_mw"].max()
            max_peak_row = scenario_df.loc[scenario_df["peak_delta_mw"].idxmax()]
            scenario_name = max_peak_row.get("scenario", "Unknown")
            quarter = f"{max_peak_row.get('year', '?')}Q{max_peak_row.get('quarter', '?')}"
            report_lines.append(
                f"- **Maximum peak uplift**: {scenario_name} scenario, {quarter} "
                f"(+{max_peak_delta:,.0f} MW)\n"
            )

        if "high_hours_delta" in scenario_df.columns:
            max_hours_delta = scenario_df["high_hours_delta"].max()
            max_hours_row = scenario_df.loc[scenario_df["high_hours_delta"].idxmax()]
            scenario_name = max_hours_row.get("scenario", "Unknown")
            quarter = f"{max_hours_row.get('year', '?')}Q{max_hours_row.get('quarter', '?')}"
            report_lines.append(
                f"- **Maximum high-demand hours increase**: {scenario_name} scenario, {quarter} "
                f"(+{max_hours_delta:,.0f} hours)\n"
            )

    except FileNotFoundError:
        report_lines.append("- Scenario strain data not found\n")

    # Limitations
    report_lines.append("\n## Limitations\n")
    report_lines.append(
        "- **Demand-only proxy**: This analysis focuses on demand forecasting and does not "
        "model transmission constraints, generation capacity, or grid stability.\n"
    )
    report_lines.append(
        "- **No transmission constraints**: Peak demand increases do not account for "
        "regional transmission bottlenecks or local capacity limits.\n"
    )
    report_lines.append(
        "- **Scenario-based AI load**: AI/data center load projections are exogenous "
        "scenarios, not learned from historical patterns.\n"
    )

    # Artifacts
    report_lines.append("\n## Generated Artifacts\n")
    report_lines.append("### Metrics CSVs\n")
    report_lines.append("- `baseline_metrics.csv` - Baseline model performance metrics\n")
    report_lines.append("- `nn_metrics.csv` - Neural network performance metrics\n")
    report_lines.append(
        "- `strain_quarterly_*.csv` - Quarterly strain metrics by model\n"
    )
    report_lines.append(
        "- `scenario_strain_quarterly.csv` - Scenario analysis results\n"
    )

    report_lines.append("\n### Figures\n")
    report_lines.append("- `model_rmse_comparison.png` - Model RMSE comparison\n")
    report_lines.append(
        "- `*_peak_window_actual_vs_pred.png` - Time series around peak days\n"
    )
    report_lines.append(
        "- `*_monthly_peak_scatter.png` - Monthly peak scatter plots\n"
    )
    report_lines.append(
        "- `scenario_quarterly_peak.png` - Scenario peak demand by quarter\n"
    )
    report_lines.append(
        "- `scenario_quarterly_high_hours_delta.png` - Scenario high-demand hours delta\n"
    )

    # Write report
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write("\n".join(report_lines))


def main(argv=None) -> None:
    """CLI entry point for reporting."""
    parser = argparse.ArgumentParser(
        description="Generate visualizations and markdown report"
    )
    parser.add_argument(
        "--prefer-split",
        default="test",
        choices=["test", "val"],
        help="Preferred split for model comparison",
    )
    parser.add_argument(
        "--out-dir",
        default="reports/figures",
        help="Output directory for figures",
    )

    args = parser.parse_args(argv)

    # Ensure directories exist
    figures_dir = ensure_dirs()
    if args.out_dir != "reports/figures":
        figures_dir = Path(args.out_dir)
        figures_dir.mkdir(parents=True, exist_ok=True)

    print("Generating visualizations and report...\n")

    # Plot 1: Model RMSE comparison
    print("1. Model RMSE comparison...")
    plot_model_rmse_comparison(
        "reports/baseline_metrics.csv",
        "reports/nn_metrics.csv",
        figures_dir / "model_rmse_comparison.png",
        prefer_split=args.prefer_split,
    )

    # Plot 2: Peak window actual vs predicted
    print("2. Peak window plots...")
    plot_peak_window_actual_vs_pred(
        "data/processed/preds_mlp_val.parquet",
        figures_dir / "val_peak_window_actual_vs_pred.png",
        "val",
    )
    plot_peak_window_actual_vs_pred(
        "data/processed/preds_mlp_test.parquet",
        figures_dir / "test_peak_window_actual_vs_pred.png",
        "test",
    )

    # Plot 3: Monthly peak scatter
    print("3. Monthly peak scatter plots...")
    plot_monthly_peak_scatter(
        "data/processed/preds_mlp_val.parquet",
        figures_dir / "val_monthly_peak_scatter.png",
        "val",
    )
    plot_monthly_peak_scatter(
        "data/processed/preds_mlp_test.parquet",
        figures_dir / "test_monthly_peak_scatter.png",
        "test",
    )

    # Plot 4: Scenario results
    print("4. Scenario plots...")
    plot_scenario_quarterly_peak(
        "reports/scenario_strain_quarterly.csv",
        figures_dir / "scenario_quarterly_peak.png",
    )
    plot_scenario_high_hours_delta(
        "reports/scenario_strain_quarterly.csv",
        figures_dir / "scenario_quarterly_high_hours_delta.png",
    )

    # Generate markdown report
    print("5. Generating markdown report...")
    report_path = Path("reports/REPORT.md")
    generate_markdown_report(
        "reports/baseline_metrics.csv",
        "reports/nn_metrics.csv",
        "reports/scenario_strain_quarterly.csv",
        report_path,
    )

    print("\n" + "=" * 60)
    print("REPORTING COMPLETE")
    print("=" * 60)
    print(f"\nFigures saved to: {figures_dir}")
    print(f"Report saved to: {report_path}")
    print("\nGenerated artifacts:")
    for fig_file in sorted(figures_dir.glob("*.png")):
        print(f"  - {fig_file}")
    print(f"  - {report_path}")


if __name__ == "__main__":
    main()

