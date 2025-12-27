"""
Train and evaluate neural network. Save model artifacts, predictions, metrics.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow
import joblib
from sklearn.metrics import mean_absolute_error, mean_squared_error

from src.models.neural_net import get_feature_columns, make_xy, train_mlp, predict


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

    # Get timestamp column or index
    if "timestamp" in df.columns:
        timestamp_col = df["timestamp"]
    else:
        timestamp_col = df.index

    # Create working DataFrame
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


def parse_hidden_layers(s: str) -> tuple:
    """
    Parse hidden layer sizes from string like "64,32".

    Args:
        s: Comma-separated string of layer sizes

    Returns:
        Tuple of integers
    """
    return tuple(int(x.strip()) for x in s.split(","))


def main(argv=None) -> None:
    """CLI entry point for neural network evaluation."""
    parser = argparse.ArgumentParser(
        description="Train and evaluate MLPRegressor on demand forecasting"
    )
    parser.add_argument(
        "--data-path",
        default="data/processed/model_dataset_hourly_2019_2025.parquet",
        help="Path to model dataset parquet file",
    )
    parser.add_argument(
        "--hidden",
        default="64,32",
        help="Hidden layer sizes as comma-separated integers (e.g., '64,32')",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=1e-4,
        help="L2 regularization strength",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        dest="learning_rate_init",
        help="Initial learning rate",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        dest="random_state",
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--output-dir",
        default="reports",
        help="Directory for output CSV files",
    )
    parser.add_argument(
        "--artifacts-dir",
        default="artifacts/mlp",
        help="Directory for model artifacts",
    )

    args = parser.parse_args(argv)

    # Parse hidden layers
    hidden_layers = parse_hidden_layers(args.hidden)

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

    # Get feature columns
    feature_cols = get_feature_columns(df)
    print(f"\nFeatures: {len(feature_cols)} numeric features selected")
    print(f"  First 15 features: {feature_cols[:15]}")

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

    # Train model
    print("\nTraining MLP...")
    params = {
        "hidden_layer_sizes": hidden_layers,
        "alpha": args.alpha,
        "learning_rate_init": args.learning_rate_init,
        "random_state": args.random_state,
    }

    trained = train_mlp(df_train, feature_cols, params)
    print("  Training complete")

    # Save artifacts
    artifacts_dir = Path(args.artifacts_dir)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    model_path = artifacts_dir / "mlp_model.joblib"
    scaler_path = artifacts_dir / "scaler.joblib"
    features_path = artifacts_dir / "feature_cols.json"

    joblib.dump(trained["model"], model_path)
    joblib.dump(trained["scaler"], scaler_path)
    with open(features_path, "w") as f:
        json.dump(feature_cols, f, indent=2)

    print(f"\nSaved artifacts to {artifacts_dir}:")
    print(f"  - mlp_model.joblib")
    print(f"  - scaler.joblib")
    print(f"  - feature_cols.json")

    # Evaluate on validation set
    print("\nEvaluating on validation set...")
    y_hat_val = predict(trained, df_val)
    y_actual_val = df_val["y"]

    # Align and drop NaNs
    aligned_val = pd.DataFrame(
        {"y_actual": y_actual_val, "y_pred": y_hat_val}
    ).dropna()

    val_metrics = {}
    if len(aligned_val) > 0:
        val_metrics["val_mae"] = mean_absolute_error(
            aligned_val["y_actual"], aligned_val["y_pred"]
        )
        val_metrics["val_rmse"] = np.sqrt(
            mean_squared_error(aligned_val["y_actual"], aligned_val["y_pred"])
        )
        val_metrics["val_n_samples"] = len(aligned_val)

        # Monthly peak MAE
        df_val_aligned = df_val.loc[aligned_val.index].copy()
        val_metrics["val_monthly_peak_mae"] = compute_monthly_peak_mae(
            df_val_aligned, aligned_val["y_actual"], aligned_val["y_pred"]
        )

        # Quarterly strain
        strain_val = compute_quarterly_strain(
            df_val_aligned, aligned_val["y_actual"], aligned_val["y_pred"], high_threshold
        )
    else:
        val_metrics["val_mae"] = np.nan
        val_metrics["val_rmse"] = np.nan
        val_metrics["val_monthly_peak_mae"] = np.nan
        val_metrics["val_n_samples"] = 0
        strain_val = pd.DataFrame()

    # Evaluate on test set
    print("Evaluating on test set...")
    y_hat_test = predict(trained, df_test)
    y_actual_test = df_test["y"]

    # Align and drop NaNs
    aligned_test = pd.DataFrame(
        {"y_actual": y_actual_test, "y_pred": y_hat_test}
    ).dropna()

    test_metrics = {}
    if len(aligned_test) > 0:
        test_metrics["test_mae"] = mean_absolute_error(
            aligned_test["y_actual"], aligned_test["y_pred"]
        )
        test_metrics["test_rmse"] = np.sqrt(
            mean_squared_error(aligned_test["y_actual"], aligned_test["y_pred"])
        )
        test_metrics["test_n_samples"] = len(aligned_test)

        # Monthly peak MAE
        df_test_aligned = df_test.loc[aligned_test.index].copy()
        test_metrics["test_monthly_peak_mae"] = compute_monthly_peak_mae(
            df_test_aligned, aligned_test["y_actual"], aligned_test["y_pred"]
        )

        # Quarterly strain
        strain_test = compute_quarterly_strain(
            df_test_aligned, aligned_test["y_actual"], aligned_test["y_pred"], high_threshold
        )
    else:
        test_metrics["test_mae"] = np.nan
        test_metrics["test_rmse"] = np.nan
        test_metrics["test_monthly_peak_mae"] = np.nan
        test_metrics["test_n_samples"] = 0
        strain_test = pd.DataFrame()

    # Save metrics
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics_df = pd.DataFrame(
        [
            {
                "model": "MLPRegressor",
                **val_metrics,
                **test_metrics,
            }
        ]
    )
    metrics_path = output_dir / "nn_metrics.csv"
    metrics_df.to_csv(metrics_path, index=False)
    print(f"\nSaved metrics to {metrics_path}")

    # Save quarterly strain metrics
    if len(strain_val) > 0:
        strain_val_path = output_dir / "strain_quarterly_nn_val.csv"
        strain_val.to_csv(strain_val_path, index=False)
        print(f"Saved val strain metrics to {strain_val_path}")

    if len(strain_test) > 0:
        strain_test_path = output_dir / "strain_quarterly_nn_test.csv"
        strain_test.to_csv(strain_test_path, index=False)
        print(f"Saved test strain metrics to {strain_test_path}")

    # Save predictions
    processed_dir = Path("data/processed")
    processed_dir.mkdir(parents=True, exist_ok=True)

    if len(aligned_val) > 0:
        preds_val = pd.DataFrame(
            {
                "timestamp": df_val_aligned["timestamp"],
                "y": aligned_val["y_actual"],
                "y_hat": aligned_val["y_pred"],
                "split": "val",
            }
        )
        preds_val_path = processed_dir / "preds_mlp_val.parquet"
        preds_val.to_parquet(preds_val_path, engine="pyarrow", index=False)
        print(f"Saved val predictions to {preds_val_path}")

    if len(aligned_test) > 0:
        preds_test = pd.DataFrame(
            {
                "timestamp": df_test_aligned["timestamp"],
                "y": aligned_test["y_actual"],
                "y_hat": aligned_test["y_pred"],
                "split": "test",
            }
        )
        preds_test_path = processed_dir / "preds_mlp_test.parquet"
        preds_test.to_parquet(preds_test_path, engine="pyarrow", index=False)
        print(f"Saved test predictions to {preds_test_path}")

    # Print summary
    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)
    print("\nMetrics:")
    print(metrics_df.to_string(index=False))
    print("\n" + "=" * 60)
    print(f"High-demand threshold: {high_threshold:,.0f} MW")
    print("=" * 60)


if __name__ == "__main__":
    main()

