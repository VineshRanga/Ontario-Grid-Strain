"""
Preflight Checks

Validates inputs and outputs before running the pipeline.
"""

import argparse
import sys
from pathlib import Path

import pandas as pd
import pyarrow

from src.utils.contracts import (
    ECCC_PROCESSED_REQUIRED_COLS,
    IESO_PROCESSED_REQUIRED_COLS,
    MODEL_DATASET_REQUIRED_COLS,
    require_columns,
    require_monotonic_hourly,
    require_unique_timestamp,
    summarize_coverage,
)


def check_raw_ieso_files() -> None:
    """
    Check that all required IESO raw CSV files exist.

    Raises:
        SystemExit: If any files are missing
    """
    raw_dir = Path("data/raw/ieso")
    missing_files = []

    for year in range(2019, 2026):
        file_path = raw_dir / f"PUB_Demand_{year}.csv"
        if not file_path.exists():
            missing_files.append(str(file_path))

    if missing_files:
        print("ERROR: Missing IESO raw files:")
        for f in missing_files:
            print(f"  - {f}")
        print("\nPlease ensure all IESO PUB demand CSVs are in data/raw/ieso/")
        raise SystemExit(1)


def check_ieso_csv_headers() -> None:
    """
    Check that IESO CSV files have required headers.

    Raises:
        SystemExit: If headers are invalid
    """
    raw_dir = Path("data/raw/ieso")
    required_headers = ["Date", "Hour", "Market Demand", "Ontario Demand"]

    for year in range(2019, 2026):
        file_path = raw_dir / f"PUB_Demand_{year}.csv"
        if not file_path.exists():
            continue

        # Read first few lines to find header
        with open(file_path, "r") as f:
            lines = []
            for i, line in enumerate(f):
                if i >= 10:  # Read up to 10 lines
                    break
                lines.append(line)

        # Find header line (first line that doesn't start with "\\")
        header_line = None
        for line in lines:
            if not line.strip().startswith("\\"):
                header_line = line.strip()
                break

        if header_line is None:
            print(f"ERROR: Could not find header in {file_path}")
            raise SystemExit(1)

        # Check required headers
        headers = [h.strip() for h in header_line.split(",")]
        missing = [h for h in required_headers if h not in headers]

        if missing:
            print(f"ERROR: {file_path} missing required headers: {missing}")
            print(f"  Found headers: {headers}")
            raise SystemExit(1)


def check_raw_eccc_files() -> bool:
    """
    Check if ECCC raw files exist (warning only, not error).

    Returns:
        True if files exist, False otherwise
    """
    eccc_dir = Path("data/raw/eccc/toronto_pearson")
    if not eccc_dir.exists():
        return False

    # Check if any CSV files exist
    csv_files = list(eccc_dir.glob("**/*.csv"))
    if len(csv_files) == 0:
        print("WARNING: ECCC raw directory exists but contains no CSV files")
        print("  This is OK if you haven't run the ECCC downloader yet")
        return False

    return True


def validate_processed_file(
    file_path: Path, required_cols: list, name: str
) -> None:
    """
    Validate a processed parquet file using contracts.

    Args:
        file_path: Path to parquet file
        required_cols: List of required columns
        name: Name of dataset (for messages)
    """
    if not file_path.exists():
        return  # Skip if file doesn't exist

    print(f"\nValidating {name}...")
    df = pd.read_parquet(file_path)

    # Validate columns
    require_columns(df, required_cols, name)

    # Validate timestamps
    require_unique_timestamp(df, name)
    require_monotonic_hourly(df, name)

    # Summarize coverage
    summary = summarize_coverage(df, name)
    print(f"  Start: {summary['start_timestamp']}")
    print(f"  End: {summary['end_timestamp']}")
    print(f"  Rows: {summary['row_count']:,}")
    print(f"  Coverage by year:")
    for year, count in sorted(summary["coverage_by_year"].items()):
        print(f"    {year}: {count:,} rows")

    # Special checks for IESO
    if "ieso" in name.lower() and "ontario_demand_mw" in df.columns:
        # Check 2025 coverage
        if 2025 in summary["coverage_by_year"]:
            count_2025 = summary["coverage_by_year"][2025]
            if count_2025 < 8760:
                print(
                    f"  WARNING: 2025 has only {count_2025:,} rows "
                    f"(expected ~8760 for full year; partial year is OK)"
                )

        # Check for NaNs in ontario_demand_mw
        nan_count = df["ontario_demand_mw"].isna().sum()
        if nan_count > 0:
            print(f"  ERROR: {nan_count} NaN values found in ontario_demand_mw")
            raise ValueError(f"{name} contains NaN values in ontario_demand_mw")


def main() -> None:
    """CLI entry point for preflight checks."""
    parser = argparse.ArgumentParser(
        description="Preflight checks for pipeline inputs and outputs"
    )
    args = parser.parse_args()

    print("=" * 60)
    print("PREFLIGHT CHECKS")
    print("=" * 60)

    errors = []
    warnings = []

    # A) Check raw IESO files exist
    print("\n[1/5] Checking raw IESO files...")
    try:
        check_raw_ieso_files()
        print("  ✓ All IESO raw files present")
    except SystemExit:
        raise
    except Exception as e:
        errors.append(f"IESO file check failed: {e}")

    # B) Quick raw CSV sanity (IESO headers)
    print("\n[2/5] Checking IESO CSV headers...")
    try:
        check_ieso_csv_headers()
        print("  ✓ IESO CSV headers valid")
    except SystemExit:
        raise
    except Exception as e:
        errors.append(f"IESO header check failed: {e}")

    # C) Check raw ECCC files (warning only)
    print("\n[3/5] Checking raw ECCC files...")
    eccc_exists = check_raw_eccc_files()
    if eccc_exists:
        print("  ✓ ECCC raw files present")
    else:
        warnings.append("ECCC raw files not found (OK if downloader not run)")

    # D) Validate processed artifacts if they exist
    print("\n[4/5] Validating processed artifacts (if present)...")

    ieso_processed = Path("data/processed/ieso_hourly_2019_2025.parquet")
    if ieso_processed.exists():
        try:
            validate_processed_file(
                ieso_processed, IESO_PROCESSED_REQUIRED_COLS, "IESO processed"
            )
        except Exception as e:
            errors.append(f"IESO processed validation failed: {e}")
    else:
        print("  IESO processed file not found (will be created by ingest)")

    eccc_processed = Path(
        "data/processed/eccc_hourly_toronto_pearson_2019_2025.parquet"
    )
    if eccc_processed.exists():
        try:
            validate_processed_file(
                eccc_processed, ECCC_PROCESSED_REQUIRED_COLS, "ECCC processed"
            )
        except Exception as e:
            errors.append(f"ECCC processed validation failed: {e}")
    else:
        print("  ECCC processed file not found (will be created by ingest)")

    model_dataset = Path("data/processed/model_dataset_hourly_2019_2025.parquet")
    if model_dataset.exists():
        try:
            validate_processed_file(
                model_dataset, MODEL_DATASET_REQUIRED_COLS, "Model dataset"
            )
        except Exception as e:
            errors.append(f"Model dataset validation failed: {e}")
    else:
        print("  Model dataset not found (will be created by features)")

    # E) Summary
    print("\n[5/5] Summary...")

    if errors:
        print("\n" + "=" * 60)
        print("PREFLIGHT FAILED")
        print("=" * 60)
        print("\nErrors:")
        for error in errors:
            print(f"  ✗ {error}")
        if warnings:
            print("\nWarnings:")
            for warning in warnings:
                print(f"  ⚠ {warning}")
        raise SystemExit(1)

    if warnings:
        print("\n" + "=" * 60)
        print("PREFLIGHT OK (with warnings)")
        print("=" * 60)
        for warning in warnings:
            print(f"  ⚠ {warning}")
    else:
        print("\n" + "=" * 60)
        print("PREFLIGHT OK")
        print("=" * 60)


if __name__ == "__main__":
    main()

