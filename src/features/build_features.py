"""
Merge demand + weather, add features (HDD/CDD, time, lags), create train/val/test splits.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow

# Constants
BASE_TEMP_C = 18.0


def load_and_merge(
    demand_path: str = "data/processed/ieso_hourly_2019_2025.parquet",
    weather_path: str = "data/processed/eccc_hourly_toronto_pearson_2019_2025.parquet",
) -> pd.DataFrame:
    """
    Load demand and weather parquets, left join on timestamp.
    Demand is the spine. Weather columns get added.
    """
    # Load data
    print("Loading demand data...")
    demand = pd.read_parquet(demand_path)
    print(f"  Loaded {len(demand):,} rows")

    print("Loading weather data...")
    weather = pd.read_parquet(weather_path)
    print(f"  Loaded {len(weather):,} rows")

    # Ensure timestamp is index for both
    if "timestamp" in demand.columns:
        demand = demand.set_index("timestamp")
    if "timestamp" in weather.columns:
        weather = weather.set_index("timestamp")

    # Left join: demand is the spine
    # Only merge weather columns we need (avoid conflicts with demand columns)
    weather_cols_to_merge = [
        "temp_c",
        "dewpoint_c",
        "rel_hum_pct",
        "wind_spd_kmh",
        "precip_mm",
        "stn_press_kpa",
        "is_imputed_weather",
    ]
    weather_to_merge = weather[[col for col in weather_cols_to_merge if col in weather.columns]]
    
    print("Merging datasets...")
    df = demand.merge(
        weather_to_merge,
        left_index=True,
        right_index=True,
        how="left",
    )

    # Validate
    # Check timestamp uniqueness
    if df.index.duplicated().any():
        raise ValueError("Duplicate timestamps found after merge")

    # Check ontario_demand_mw has no missing values
    if df["ontario_demand_mw"].isna().any():
        missing_count = df["ontario_demand_mw"].isna().sum()
        raise ValueError(
            f"ontario_demand_mw has {missing_count} missing values after merge"
        )

    # Check temp_c exists (may be NaN, that's okay)
    if "temp_c" not in df.columns:
        raise ValueError("temp_c column not found after merge")

    # Sort by timestamp
    df = df.sort_index()

    print(f"  Merged dataset: {len(df):,} rows")
    return df


def fill_weather(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fill missing weather: interpolate, then forward fill, then backward fill.
    Adds weather_was_missing flag.
    """
    weather_cols = [
        "temp_c",
        "dewpoint_c",
        "rel_hum_pct",
        "wind_spd_kmh",
        "precip_mm",
        "stn_press_kpa",
    ]

    # Track which rows had missing weather before filling
    weather_missing_before = df["temp_c"].isna().copy()
    if "is_imputed_weather" in df.columns:
        weather_imputed = df["is_imputed_weather"].fillna(False)
    else:
        weather_imputed = pd.Series(False, index=df.index)

    # Fill weather columns (exclude precip_mm as it's handled separately)
    for col in weather_cols:
        if col in df.columns and col != "precip_mm":
            # Time-based interpolation
            df[col] = df[col].interpolate(method="time")
            # Forward fill
            df[col] = df[col].ffill()
            # Backward fill
            df[col] = df[col].bfill()

    # Create weather_was_missing flag
    df["weather_was_missing"] = weather_missing_before | weather_imputed

    return df


def add_hdd_cdd(df: pd.DataFrame, base_temp: float = BASE_TEMP_C) -> pd.DataFrame:
    """
    Add Heating Degree Days and Cooling Degree Days features.

    Args:
        df: DataFrame with temp_c column
        base_temp: Base temperature in Celsius (default 18.0)

    Returns:
        DataFrame with hdd_18 and cdd_18 columns added
    """
    df["hdd_18"] = (base_temp - df["temp_c"]).clip(lower=0)
    df["cdd_18"] = (df["temp_c"] - base_temp).clip(lower=0)
    return df


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add time-derived features including cyclical encodings.

    Args:
        df: DataFrame with timestamp index

    Returns:
        DataFrame with time features added
    """
    # Basic time features
    df["hour"] = df.index.hour
    df["day_of_week"] = df.index.dayofweek  # 0=Monday
    df["month"] = df.index.month
    df["year"] = df.index.year
    df["quarter"] = df.index.quarter
    df["is_weekend"] = df["day_of_week"].isin([5, 6])  # Saturday, Sunday

    # Day of year
    doy = df.index.dayofyear

    # Cyclical encodings
    # Hour: 2*pi*hour/24
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)

    # Day of year: 2*pi*doy/365.25
    df["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    df["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)

    return df


def add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add lag features and rolling statistics for demand.

    Args:
        df: DataFrame with ontario_demand_mw column

    Returns:
        DataFrame with lag and rolling features added
    """
    # Lag features (shifted so no leakage)
    df["demand_lag_1h"] = df["ontario_demand_mw"].shift(1)
    df["demand_lag_24h"] = df["ontario_demand_mw"].shift(24)
    df["demand_lag_168h"] = df["ontario_demand_mw"].shift(168)

    # Rolling statistics (shifted so no leakage)
    # shift(1) then rolling: uses previous values only
    df["demand_rollmean_24h"] = (
        df["ontario_demand_mw"].shift(1).rolling(window=24, min_periods=1).mean()
    )
    df["demand_rollmean_168h"] = (
        df["ontario_demand_mw"].shift(1).rolling(window=168, min_periods=1).mean()
    )

    return df


def add_split_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add train/val/test split labels based on timestamp.

    Args:
        df: DataFrame with timestamp index

    Returns:
        DataFrame with split column added
    """
    # Define split boundaries
    train_end = pd.Timestamp("2023-12-31 23:00:00")
    val_end = pd.Timestamp("2024-12-31 23:00:00")

    # Assign splits
    df["split"] = "test"  # default
    df.loc[df.index <= train_end, "split"] = "train"
    df.loc[
        (df.index > train_end) & (df.index <= val_end), "split"
    ] = "val"

    return df


def build_model_dataset(
    demand_path: str = "data/processed/ieso_hourly_2019_2025.parquet",
    weather_path: str = "data/processed/eccc_hourly_toronto_pearson_2019_2025.parquet",
) -> pd.DataFrame:
    """
    Full pipeline: load, merge, fill weather, add all features, create splits.
    Returns DataFrame ready for modeling.
    """
    # Load and merge
    df = load_and_merge(demand_path, weather_path)

    # Handle precip_mm explicitly (fill with 0.0 before other weather filling)
    if "precip_mm" in df.columns:
        print("Filling missing precip_mm with 0.0...")
        df["precip_mm"] = df["precip_mm"].fillna(0.0)

    # Fill weather
    print("Filling missing weather values...")
    df = fill_weather(df)

    # Add HDD/CDD
    print("Adding HDD/CDD features...")
    df = add_hdd_cdd(df)

    # Add time features
    print("Adding time features...")
    df = add_time_features(df)

    # Add lag features
    print("Adding lag features...")
    df = add_lag_features(df)

    # Add split labels
    print("Adding split labels...")
    df = add_split_labels(df)

    # Create target column
    df["y"] = df["ontario_demand_mw"]

    # Select final columns in a clean order
    # Keep timestamp as a column (reset index)
    # Ensure index is named 'timestamp' if it isn't already
    if df.index.name is None:
        df.index.name = "timestamp"
    df = df.reset_index()

    # Define column order
    feature_cols = [
        "timestamp",
        "y",
        "market_demand_mw",
        # Weather
        "temp_c",
        "dewpoint_c",
        "rel_hum_pct",
        "wind_spd_kmh",
        "precip_mm",
        "stn_press_kpa",
        # HDD/CDD
        "hdd_18",
        "cdd_18",
        # Time features
        "hour",
        "day_of_week",
        "month",
        "year",
        "quarter",
        "is_weekend",
        "hour_sin",
        "hour_cos",
        "doy_sin",
        "doy_cos",
        # Lag features
        "demand_lag_1h",
        "demand_lag_24h",
        "demand_lag_168h",
        "demand_rollmean_24h",
        "demand_rollmean_168h",
        # Flags
        "is_imputed",
        "weather_was_missing",
        "split",
    ]

    # Select only columns that exist
    available_cols = [col for col in feature_cols if col in df.columns]
    df = df[available_cols]

    return df


def main(argv=None) -> None:
    """CLI entry point for feature engineering."""
    parser = argparse.ArgumentParser(
        description="Build model-ready dataset with engineered features"
    )
    parser.add_argument(
        "--demand-path",
        default="data/processed/ieso_hourly_2019_2025.parquet",
        help="Path to demand parquet file",
    )
    parser.add_argument(
        "--weather-path",
        default="data/processed/eccc_hourly_toronto_pearson_2019_2025.parquet",
        help="Path to weather parquet file",
    )
    parser.add_argument(
        "--output",
        default="data/processed/model_dataset_hourly_2019_2025.parquet",
        help="Output parquet file path",
    )

    args = parser.parse_args(argv)

    # Build dataset
    print("Building model dataset...\n")
    df = build_model_dataset(args.demand_path, args.weather_path)

    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, engine="pyarrow", index=False)
    print(f"\nSaved to {output_path}\n")

    # Print summary
    print("Summary:")
    print(f"  Total rows: {len(df):,}")
    print(f"  Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")

    # Count per split
    print("\n  Rows per split:")
    for split in ["train", "val", "test"]:
        count = (df["split"] == split).sum()
        pct = 100 * count / len(df)
        print(f"    {split}: {count:,} ({pct:.1f}%)")

    # Compute per-column NaN rates
    nan_rates = df.isna().mean().sort_values(ascending=False)
    
    # Print top 10 columns by NaN rate
    print("\n  Top 10 columns by NaN rate:")
    top_nan_cols = nan_rates.head(10)
    for col, rate in top_nan_cols.items():
        if rate > 0:
            print(f"    {col}: {rate:.1%} ({df[col].isna().sum():,} NaNs)")
    
    # Check for 100% NaN columns (after precip fix)
    full_nan_cols = nan_rates[nan_rates == 1.0].index.tolist()
    if full_nan_cols:
        print(f"\n  WARNING: Columns with 100% NaN: {full_nan_cols}")
    
    # Count rows with NaNs (including all columns)
    nan_rows_all = df.isna().any(axis=1).sum()
    print(f"\n  Rows with any NaNs (including all columns): {nan_rows_all:,} ({100 * nan_rows_all / len(df):.1f}%)")
    
    # Count rows with NaNs excluding precip_mm
    df_excluding_precip = df.drop(columns=["precip_mm"], errors="ignore")
    nan_rows_excluding_precip = df_excluding_precip.isna().any(axis=1).sum()
    print(f"  Rows with any NaNs (excluding precip_mm): {nan_rows_excluding_precip:,} ({100 * nan_rows_excluding_precip / len(df):.1f}%)")

    # Target statistics
    print(f"\n  Target (y) statistics:")
    print(f"    Min: {df['y'].min():,.0f} MW")
    print(f"    Max: {df['y'].max():,.0f} MW")
    print(f"    Mean: {df['y'].mean():,.0f} MW")

    # Imputation flags
    if "is_imputed" in df.columns:
        imputed_pct = 100 * df["is_imputed"].sum() / len(df)
        print(f"\n  Rows with is_imputed=True: {df['is_imputed'].sum():,} ({imputed_pct:.1f}%)")

    if "weather_was_missing" in df.columns:
        weather_missing_pct = 100 * df["weather_was_missing"].sum() / len(df)
        print(
            f"  Rows with weather_was_missing=True: {df['weather_was_missing'].sum():,} ({weather_missing_pct:.1f}%)"
        )


if __name__ == "__main__":
    main()

