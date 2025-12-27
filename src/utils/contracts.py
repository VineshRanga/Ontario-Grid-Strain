"""
Data Contract Definitions

Schema contracts and validation utilities for data quality checks.
"""

from typing import Dict, List

import pandas as pd

# Expected schemas (required columns)
IESO_PROCESSED_REQUIRED_COLS = [
    "timestamp",
    "ontario_demand_mw",
    "market_demand_mw",
    "source_file",
    "is_imputed",
]

ECCC_PROCESSED_REQUIRED_COLS = [
    "timestamp",
    "temp_c",
    "dewpoint_c",
    "rel_hum_pct",
    "wind_spd_kmh",
    "precip_mm",
    "stn_press_kpa",
    "source_file",
    "is_imputed_weather",
]

MODEL_DATASET_REQUIRED_COLS = [
    "timestamp",
    "y",
    "split",
    "hdd_18",
    "cdd_18",
    "demand_lag_1h",
    "demand_lag_24h",
    "demand_lag_168h",
    "demand_rollmean_24h",
    "demand_rollmean_168h",
]


def require_columns(df: pd.DataFrame, required: List[str], name: str) -> None:
    """
    Validate that DataFrame contains all required columns.

    Args:
        df: DataFrame to validate
        required: List of required column names
        name: Name of dataset (for error messages)

    Raises:
        ValueError: If any required columns are missing
    """
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(
            f"{name} missing required columns: {missing}. "
            f"Available columns: {list(df.columns)}"
        )


def require_unique_timestamp(df: pd.DataFrame, name: str) -> None:
    """
    Validate that timestamp column has no duplicates.

    Args:
        df: DataFrame with timestamp column
        name: Name of dataset (for error messages)

    Raises:
        ValueError: If duplicate timestamps are found
    """
    if "timestamp" not in df.columns:
        raise ValueError(f"{name} missing 'timestamp' column")

    duplicates = df["timestamp"].duplicated(keep=False)
    if duplicates.any():
        dup_count = duplicates.sum()
        dup_examples = df.loc[duplicates, "timestamp"].unique()[:5]
        raise ValueError(
            f"{name} has {dup_count} duplicate timestamps. "
            f"Examples: {list(dup_examples)}"
        )


def require_monotonic_hourly(df: pd.DataFrame, name: str) -> None:
    """
    Validate that timestamps are sorted and hourly continuous.

    Args:
        df: DataFrame with timestamp column
        name: Name of dataset (for error messages)

    Raises:
        ValueError: If timestamps are not sorted or have gaps
    """
    if "timestamp" not in df.columns:
        raise ValueError(f"{name} missing 'timestamp' column")

    # Ensure timestamp is datetime
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    # Check sorted
    if not df["timestamp"].is_monotonic_increasing:
        raise ValueError(f"{name} timestamps are not sorted")

    # Check hourly continuity
    if len(df) < 2:
        return  # Single row or empty, no gaps to check

    # Compute time differences
    diffs = df["timestamp"].diff().dropna()
    expected_hour = pd.Timedelta(hours=1)

    # Allow small tolerance for floating point issues
    tolerance = pd.Timedelta(seconds=1)
    gaps = diffs[(diffs - expected_hour).abs() > tolerance]

    if len(gaps) > 0:
        gap_count = len(gaps)
        gap_examples = gaps.head(5)
        raise ValueError(
            f"{name} has {gap_count} timestamp gaps (not hourly continuous). "
            f"Example gaps: {list(gap_examples.values)}"
        )


def summarize_coverage(df: pd.DataFrame, name: str) -> Dict:
    """
    Summarize temporal coverage of dataset.

    Args:
        df: DataFrame with timestamp column
        name: Name of dataset (for output)

    Returns:
        Dictionary with:
        - start_timestamp
        - end_timestamp
        - row_count
        - coverage_by_year: dict mapping year to row count
    """
    if "timestamp" not in df.columns:
        raise ValueError(f"{name} missing 'timestamp' column")

    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    summary = {
        "start_timestamp": df["timestamp"].min(),
        "end_timestamp": df["timestamp"].max(),
        "row_count": len(df),
        "coverage_by_year": {},
    }

    # Count rows by year
    df["year"] = df["timestamp"].dt.year
    year_counts = df["year"].value_counts().sort_index()
    summary["coverage_by_year"] = year_counts.to_dict()

    return summary

