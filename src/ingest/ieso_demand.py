"""
Load IESO demand CSVs, clean them, fill missing hours, save as parquet.
"""

import argparse
from pathlib import Path

import pandas as pd
import pyarrow


def read_pub_demand_csv(path: str) -> pd.DataFrame:
    """
    Read one IESO CSV, parse Date/Hour columns, create timestamp index.
    Returns DataFrame with demand columns and source_file.
    """
    path_obj = Path(path)
    source_file = path_obj.name

    # Read CSV, skipping metadata lines starting with "\\"
    df = pd.read_csv(path, comment="\\")

    # Validate required columns
    required_cols = ["Date", "Hour", "Market Demand", "Ontario Demand"]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(
            f"Missing required columns in {path}: {missing_cols}. "
            f"Available columns: {list(df.columns)}"
        )

    # Parse columns
    df["Date"] = pd.to_datetime(df["Date"])
    df["Hour"] = df["Hour"].astype(int)

    # Validate Hour range
    if not ((df["Hour"] >= 1) & (df["Hour"] <= 24)).all():
        raise ValueError(
            f"Invalid Hour values in {path}. Expected 1-24, found: {df['Hour'].unique()}"
        )

    # Convert demand columns to float
    df["market_demand_mw"] = pd.to_numeric(df["Market Demand"], errors="coerce")
    df["ontario_demand_mw"] = pd.to_numeric(df["Ontario Demand"], errors="coerce")

    # Create timestamp: Date + (Hour - 1) hours
    df["timestamp"] = df["Date"] + pd.to_timedelta(df["Hour"] - 1, unit="h")

    # Select and rename columns
    result = df[["timestamp", "ontario_demand_mw", "market_demand_mw"]].copy()
    result["source_file"] = source_file

    # Set timestamp as index
    result = result.set_index("timestamp")

    return result


def build_ieso_hourly(
    input_dir: str = "data/raw/ieso", years: range = range(2019, 2026)
) -> pd.DataFrame:
    """
    Load all IESO CSVs, combine them, sort by timestamp.
    """
    input_path = Path(input_dir)
    missing_years = []
    dfs = []

    for year in years:
        file_path = input_path / f"PUB_Demand_{year}.csv"
        if not file_path.exists():
            missing_years.append(year)
        else:
            df = read_pub_demand_csv(str(file_path))
            dfs.append(df)

    if missing_years:
        raise FileNotFoundError(
            f"Missing IESO demand files for years: {missing_years}. "
            f"Expected files: PUB_Demand_{missing_years[0]}.csv, etc."
        )

    if not dfs:
        raise ValueError("No data files found to process")

    # Concatenate and sort
    combined = pd.concat(dfs, ignore_index=False)
    combined = combined.sort_index()

    return combined


def main(argv=None) -> None:
    """CLI entry point for IESO demand ingestion."""
    parser = argparse.ArgumentParser(
        description="Process IESO PUB demand CSVs into hourly parquet file"
    )
    parser.add_argument(
        "--input-dir",
        default="data/raw/ieso",
        help="Directory containing PUB_Demand_*.csv files",
    )
    parser.add_argument(
        "--output",
        default="data/processed/ieso_hourly_2019_2025.parquet",
        help="Output parquet file path",
    )
    parser.add_argument(
        "--start-year",
        type=int,
        default=2019,
        help="First year to process",
    )
    parser.add_argument(
        "--end-year",
        type=int,
        default=2025,
        help="Last year to process (inclusive)",
    )

    args = parser.parse_args(argv)

    # Build combined DataFrame
    print("Loading IESO demand data...")
    years = range(args.start_year, args.end_year + 1)
    df = build_ieso_hourly(input_dir=args.input_dir, years=years)

    # Track missing values before reindexing
    demand_cols = ["ontario_demand_mw", "market_demand_mw"]
    # Store original timestamps that had missing values
    original_missing_timestamps = set(
        df.index[df[demand_cols].isna().any(axis=1)]
    )

    # Check for duplicates
    duplicates = df.index.duplicated(keep=False)
    if duplicates.any():
        dup_count = duplicates.sum()
        dup_examples = df.index[duplicates].unique()[:5]
        raise ValueError(
            f"Found {dup_count} duplicate timestamps. "
            f"Examples: {list(dup_examples)}"
        )

    # Reindex to complete hourly range
    if df.empty:
        raise ValueError("No data to process")

    min_ts = df.index.min()
    max_ts = df.index.max()
    complete_index = pd.date_range(start=min_ts, end=max_ts, freq="h")

    # Track which rows are new (created by reindexing)
    original_timestamps = set(df.index)

    # Reindex
    df = df.reindex(complete_index)

    # Mark newly created rows
    new_rows = df["source_file"].isna()
    df.loc[new_rows, "source_file"] = "IMPUTED"

    # Create is_imputed column
    df["is_imputed"] = False

    # Track missing values after reindexing (before imputation)
    missing_after_reindex = df[demand_cols].isna().any(axis=1)

    # Impute missing values
    # First: time-based interpolation
    for col in demand_cols:
        df[col] = df[col].interpolate(method="time")

    # Then: forward fill
    df[demand_cols] = df[demand_cols].ffill()

    # Then: backward fill
    df[demand_cols] = df[demand_cols].bfill()

    # Update is_imputed flag
    # Mark as imputed if:
    # 1. Row was created by reindexing (new_rows)
    # 2. Value was missing after reindexing (missing_after_reindex)
    # 3. Original timestamp had missing values (from before reindexing)
    rows_with_original_missing = df.index.isin(original_missing_timestamps)
    df.loc[new_rows | missing_after_reindex | rows_with_original_missing, "is_imputed"] = True

    # Ensure timestamp is a column (not the index) before saving
    if df.index.name == "timestamp":
        # Index is named "timestamp", reset_index will create "timestamp" column
        df = df.reset_index()
    elif "timestamp" not in df.columns and df.index.dtype.kind == "M":
        # Index is datetime but not named "timestamp", reset and rename
        df = df.reset_index().rename(columns={"index": "timestamp"})
    # Ensure index is RangeIndex (drop=True removes old index)
    if not isinstance(df.index, pd.RangeIndex):
        df = df.reset_index(drop=True)
    
    # Validate before saving
    assert "timestamp" in df.columns, "timestamp must be a column before saving"
    assert pd.api.types.is_datetime64_any_dtype(df["timestamp"]), "timestamp must be datetime64[ns]"
    assert df["timestamp"].is_unique, "timestamp must be unique"
    
    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, engine="pyarrow")
    print(f"Saved to {output_path}\n")

    # Print summary
    print("Summary:")
    loaded_years = sorted(set(range(args.start_year, args.end_year + 1)))
    print(f"  Files loaded: {len(loaded_years)} years ({args.start_year}-{args.end_year})")
    print(f"  Start timestamp: {df['timestamp'].min()}")
    print(f"  End timestamp: {df['timestamp'].max()}")
    print(f"  Total rows: {len(df):,}")

    # Count missing hours before fill
    original_rows = df[df["source_file"] != "IMPUTED"]
    if not original_rows.empty:
        expected_hours = len(
            pd.date_range(
                start=original_rows["timestamp"].min(),
                end=original_rows["timestamp"].max(),
                freq="h",
            )
        )
        missing_hours = expected_hours - len(original_rows)
        print(f"  Missing hours detected (pre-fill): {missing_hours:,}")

    imputed_count = df["is_imputed"].sum()
    print(f"  Imputed rows: {imputed_count:,} ({100 * imputed_count / len(df):.1f}%)")

    # Check for partial years
    print("\n  Year coverage:")
    for year in range(args.start_year, args.end_year + 1):
        year_data = df[df["timestamp"].dt.year == year]
        if year_data.empty:
            print(f"    {year}: No data")
            continue

        # Check if max timestamp is Dec 31 23:00
        max_ts_year = year_data["timestamp"].max()
        expected_max = pd.Timestamp(f"{year}-12-31 23:00:00")

        if max_ts_year < expected_max:
            print(
                f"    {year}: PARTIAL - Max timestamp is {max_ts_year} "
                f"(expected {expected_max})"
            )
        else:
            # Check if it's a complete year
            expected_hours = 8760 if year % 4 != 0 else 8784  # Account for leap years
            actual_hours = len(year_data)
            if actual_hours < expected_hours:
                print(
                    f"    {year}: PARTIAL - {actual_hours:,} hours "
                    f"(expected {expected_hours:,})"
                )
            else:
                print(f"    {year}: Complete - {actual_hours:,} hours")


if __name__ == "__main__":
    main()

