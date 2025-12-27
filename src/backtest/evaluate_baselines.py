"""
Evaluate baseline models. Compute MAE, RMSE, monthly peak MAE, quarterly strain metrics.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow
from sklearn.metrics import mean_absolute_error, mean_squared_error

from src.models.baselines import LagRidge, LinearWeatherRidge, SeasonalNaive168


def compute_quarterly_strain(
    df: pd.DataFrame,
    y_actual: pd.Series,
    y_pred: pd.Series,
    high_threshold: float,
) -> pd.DataFrame:
    """
    Compute quarterly strain metrics.

    Args:
        df: DataFrame with timestamp column
        y_actual: Actual demand values
        y_pred: Predicted demand values
        high_threshold: Threshold for high-demand hours

    Returns:
        DataFrame with quarterly metrics
    """
    # Ensure timestamp is available
    if "timestamp" not in df.columns:
        if df.index.name == "timestamp":
            df = df.reset_index()
        else:
            raise ValueError("timestamp column or index required")

    # Align y_actual and y_pred with df
    if df.index.name == "timestamp" or df.index.equals(y_actual.index):
        # Use index alignment
        aligned_actual = y_actual.reindex(df.index)
        aligned_pred = y_pred.reindex(df.index)
    else:
        # Try to align by timestamp if available
        if "timestamp" in df.columns:
            # Create index from timestamp for alignment
            ts_index = pd.to_datetime(df["timestamp"])
            aligned_actual = y_actual.reindex(ts_index)
            aligned_pred = y_pred.reindex(ts_index)
        else:
            aligned_actual = y_actual
            aligned_pred = y_pred

    # Get timestamp column or index
    if "timestamp" in df.columns:
        timestamp_col = df["timestamp"]
    else:
        timestamp_col = df.index

    # Create working DataFrame with aligned data
    work_df = pd.DataFrame(
        {
            "timestamp": timestamp_col,
            "year": pd.to_datetime(timestamp_col).dt.year,
            "quarter": pd.to_datetime(timestamp_col).dt.quarter,
            "y_actual": y_actual.values,
            "y_pred": y_pred.values,
        }
    )

    # Group by year and quarter
    results = []
    for (year, quarter), group in work_df.groupby(["year", "quarter"]):
        actual_peak = group["y_actual"].max()
        predicted_peak = group["y_pred"].max()
        actual_high_hours = (group["y_actual"] > high_threshold).sum()
        predicted_high_hours = (group["y_pred"] > high_threshold).sum()
        peak_uplift = predicted_peak - actual_peak

        results.append(
            {
                "year": year,
                "quarter": quarter,
                "actual_peak_mw": actual_peak,
                "predicted_peak_mw": predicted_peak,
                "actual_high_hours": actual_high_hours,
                "predicted_high_hours": predicted_high_hours,
                "peak_uplift_mw": peak_uplift,
            }
        )

    return pd.DataFrame(results)


def compute_monthly_peak_mae(
    df: pd.DataFrame, y_actual: pd.Series, y_pred: pd.Series
) -> float:
    """
    Compute MAE of monthly peaks.

    Args:
        df: DataFrame with timestamp column
        y_actual: Actual demand values
        y_pred: Predicted demand values

    Returns:
        MAE of monthly peaks
    """
    # Ensure timestamp is available
    if "timestamp" not in df.columns:
        if df.index.name == "timestamp":
            df = df.reset_index()
        else:
            raise ValueError("timestamp column or index required")

    # Create working DataFrame
    work_df = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(df["timestamp"]),
            "y_actual": y_actual.values,
            "y_pred": y_pred.values,
        }
    )

    # Group by year and month
    monthly_peaks_actual = []
    monthly_peaks_pred = []

    for (year, month), group in work_df.groupby(
        [work_df["timestamp"].dt.year, work_df["timestamp"].dt.month]
    ):
        monthly_peaks_actual.append(group["y_actual"].max())
        monthly_peaks_pred.append(group["y_pred"].max())

    if not monthly_peaks_actual:
        return np.nan

    return mean_absolute_error(monthly_peaks_actual, monthly_peaks_pred)


def evaluate_model(
    model,
    df_train: pd.DataFrame,
    df_val: pd.DataFrame,
    df_test: pd.DataFrame,
    high_threshold: float,
) -> dict:
    """
    Evaluate a model on validation and test sets.

    Args:
        model: Model instance with fit() and predict() methods
        df_train: Training DataFrame
        df_val: Validation DataFrame
        df_test: Test DataFrame
        high_threshold: Threshold for high-demand hours

    Returns:
        Dictionary with evaluation results
    """
    results = {"model_name": model.name}

    # Fit model (if needed)
    if hasattr(model, "fit"):
        model.fit(df_train)

    # Evaluate on validation set
    if len(df_val) > 0:
        # Get predictions
        y_hat_val = model.predict(df_val)
        y_actual_val = df_val["y"]

        # Align and drop NaNs
        aligned = pd.DataFrame(
            {"y_actual": y_actual_val, "y_pred": y_hat_val}
        ).dropna()

        if len(aligned) > 0:
            results["val_mae"] = mean_absolute_error(
                aligned["y_actual"], aligned["y_pred"]
            )
            results["val_rmse"] = np.sqrt(
                mean_squared_error(aligned["y_actual"], aligned["y_pred"])
            )
            
            # Align df_val to same index for strain/metrics computation
            df_val_aligned = df_val.loc[aligned.index].copy()
            
            results["val_monthly_peak_mae"] = compute_monthly_peak_mae(
                df_val_aligned, aligned["y_actual"], aligned["y_pred"]
            )
            results["val_n_samples"] = len(aligned)

            # Quarterly strain metrics
            strain_val = compute_quarterly_strain(
                df_val_aligned, aligned["y_actual"], aligned["y_pred"], high_threshold
            )
            results["strain_val"] = strain_val
        else:
            results["val_mae"] = np.nan
            results["val_rmse"] = np.nan
            results["val_monthly_peak_mae"] = np.nan
            results["val_n_samples"] = 0
            results["strain_val"] = pd.DataFrame()

    # Evaluate on test set
    if len(df_test) > 0:
        # Get predictions
        y_hat_test = model.predict(df_test)
        y_actual_test = df_test["y"]

        # Align and drop NaNs
        aligned = pd.DataFrame(
            {"y_actual": y_actual_test, "y_pred": y_hat_test}
        ).dropna()

        if len(aligned) > 0:
            results["test_mae"] = mean_absolute_error(
                aligned["y_actual"], aligned["y_pred"]
            )
            results["test_rmse"] = np.sqrt(
                mean_squared_error(aligned["y_actual"], aligned["y_pred"])
            )
            
            # Align df_test to same index for strain/metrics computation
            df_test_aligned = df_test.loc[aligned.index].copy()
            
            results["test_monthly_peak_mae"] = compute_monthly_peak_mae(
                df_test_aligned, aligned["y_actual"], aligned["y_pred"]
            )
            results["test_n_samples"] = len(aligned)

            # Quarterly strain metrics
            strain_test = compute_quarterly_strain(
                df_test_aligned, aligned["y_actual"], aligned["y_pred"], high_threshold
            )
            results["strain_test"] = strain_test
        else:
            results["test_mae"] = np.nan
            results["test_rmse"] = np.nan
            results["test_monthly_peak_mae"] = np.nan
            results["test_n_samples"] = 0
            results["strain_test"] = pd.DataFrame()
    else:
        results["test_mae"] = np.nan
        results["test_rmse"] = np.nan
        results["test_monthly_peak_mae"] = np.nan
        results["test_n_samples"] = 0
        results["strain_test"] = pd.DataFrame()

    return results


def main(argv=None) -> None:
    """CLI entry point for baseline evaluation."""
    parser = argparse.ArgumentParser(
        description="Evaluate baseline models on validation and test sets"
    )
    parser.add_argument(
        "--data-path",
        default="data/processed/model_dataset_hourly_2019_2025.parquet",
        help="Path to model dataset parquet file",
    )
    parser.add_argument(
        "--output-dir",
        default="reports",
        help="Directory for output CSV files",
    )

    args = parser.parse_args(argv)

    # Load data
    print("Loading dataset...")
    df = pd.read_parquet(args.data_path)
    print(f"  Loaded {len(df):,} rows")

    # Split into train/val/test
    df_train = df[df["split"] == "train"].copy()
    df_val = df[df["split"] == "val"].copy()
    df_test = df[df["split"] == "test"].copy()

    print(f"\nData splits:")
    print(f"  Train: {len(df_train):,} rows")
    print(f"  Val: {len(df_val):,} rows")
    print(f"  Test: {len(df_test):,} rows")

    # Compute high-demand threshold from training data only
    train_y = df_train["y"].dropna()
    if len(train_y) == 0:
        raise ValueError("No valid y values in training data")

    high_threshold = np.percentile(train_y, 95)
    print(f"\nHigh-demand threshold (95th percentile of train): {high_threshold:,.0f} MW")

    # Warn if test is small
    if len(df_test) < 8000:
        print(
            f"\n⚠️  Warning: Test set has only {len(df_test):,} rows "
            f"(2025 may be partial data)"
        )

    # Initialize models
    models = [
        SeasonalNaive168(),
        LinearWeatherRidge(alpha=1.0, include_trend=True),
        LagRidge(alpha=1.0),
    ]

    # Evaluate each model
    print("\nEvaluating models...\n")
    all_results = []

    for model in models:
        print(f"Evaluating {model.name}...")
        results = evaluate_model(model, df_train, df_val, df_test, high_threshold)
        all_results.append(results)

    # Create metrics summary DataFrame
    metrics_rows = []
    for r in all_results:
        metrics_rows.append(
            {
                "model": r["model_name"],
                "val_mae": r.get("val_mae", np.nan),
                "val_rmse": r.get("val_rmse", np.nan),
                "val_monthly_peak_mae": r.get("val_monthly_peak_mae", np.nan),
                "val_n_samples": r.get("val_n_samples", 0),
                "test_mae": r.get("test_mae", np.nan),
                "test_rmse": r.get("test_rmse", np.nan),
                "test_monthly_peak_mae": r.get("test_monthly_peak_mae", np.nan),
                "test_n_samples": r.get("test_n_samples", 0),
            }
        )

    metrics_df = pd.DataFrame(metrics_rows)

    # Save metrics
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics_path = output_dir / "baseline_metrics.csv"
    metrics_df.to_csv(metrics_path, index=False)
    print(f"\nSaved metrics to {metrics_path}")

    # Save quarterly strain metrics
    # Collect all strain DataFrames
    strain_val_list = []
    strain_test_list = []

    for r in all_results:
        model_name = r["model_name"]
        if "strain_val" in r and len(r["strain_val"]) > 0:
            strain_val = r["strain_val"].copy()
            strain_val["model"] = model_name
            strain_val_list.append(strain_val)

        if "strain_test" in r and len(r["strain_test"]) > 0:
            strain_test = r["strain_test"].copy()
            strain_test["model"] = model_name
            strain_test_list.append(strain_test)

    if strain_val_list:
        strain_val_all = pd.concat(strain_val_list, ignore_index=True)
        strain_val_path = output_dir / "strain_quarterly_val.csv"
        strain_val_all.to_csv(strain_val_path, index=False)
        print(f"Saved val strain metrics to {strain_val_path}")

    if strain_test_list:
        strain_test_all = pd.concat(strain_test_list, ignore_index=True)
        strain_test_path = output_dir / "strain_quarterly_test.csv"
        strain_test_all.to_csv(strain_test_path, index=False)
        print(f"Saved test strain metrics to {strain_test_path}")

    # Print summary
    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)

    # Print metrics table
    print("\nMetrics:")
    print(metrics_df.to_string(index=False))

    # Rank by test RMSE (if test exists)
    if "test_rmse" in metrics_df.columns:
        test_rmse_valid = metrics_df["test_rmse"].notna()
        if test_rmse_valid.any():
            print("\n" + "=" * 60)
            print("MODEL RANKING (by Test RMSE)")
            print("=" * 60)
            ranked = metrics_df[test_rmse_valid].sort_values("test_rmse")
            for idx, row in ranked.iterrows():
                print(
                    f"{row['model']:25s}  Test RMSE: {row['test_rmse']:,.0f} MW"
                )

    print("\n" + "=" * 60)
    print(f"High-demand threshold: {high_threshold:,.0f} MW")
    print("=" * 60)


if __name__ == "__main__":
    main()

