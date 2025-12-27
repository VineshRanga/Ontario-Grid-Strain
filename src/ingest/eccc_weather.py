"""
Download ECCC weather CSVs for Toronto Pearson, parse them, merge into one parquet.
Handles messy column names and missing values.
"""

import argparse
import re
import time
from pathlib import Path
from typing import Dict, Optional, Tuple
from urllib.parse import urlencode

import numpy as np
import pandas as pd
import pyarrow
import requests


def build_bulk_url(station_id: int, year: int, month: int) -> str:
    """
    Build ECCC bulk data download URL for a specific station, year, and month.

    Args:
        station_id: ECCC station ID (51459 for Toronto Pearson)
        year: Year (e.g., 2019)
        month: Month (1-12)

    Returns:
        Full URL with query parameters
    """
    base_url = "https://climate.weather.gc.ca/climate_data/bulk_data_e.html"
    params = {
        "format": "csv",
        "stationID": station_id,
        "Year": year,
        "Month": f"{month:02d}",
        "Day": "01",
        "timeframe": "1",  # hourly
        "time": "LST",
    }
    return f"{base_url}?{urlencode(params)}"


def download_month_csv(
    url: str,
    out_path: Path,
    timeout: int = 60,
    retries: int = 3,
    force: bool = False,
) -> None:
    """
    Download a monthly CSV from ECCC with caching and retry logic.

    Args:
        url: Full URL to download
        out_path: Path where CSV will be saved
        timeout: Request timeout in seconds
        retries: Number of retry attempts
        force: If True, re-download even if file exists

    Raises:
        RuntimeError: If download fails after all retries
    """
    # Check cache
    if not force and out_path.exists() and out_path.stat().st_size > 0:
        return

    # Ensure parent directory exists
    out_path.parent.mkdir(parents=True, exist_ok=True)

    headers = {"User-Agent": "Mozilla/5.0 (compatible; OntarioGridStrain/1.0)"}

    last_error = None
    for attempt in range(retries):
        try:
            response = requests.get(url, headers=headers, timeout=timeout)
            response.raise_for_status()

            # Write file
            out_path.write_bytes(response.content)
            return

        except requests.exceptions.RequestException as e:
            last_error = e
            if attempt < retries - 1:
                # Exponential backoff: wait 2^attempt seconds
                wait_time = 2 ** attempt
                time.sleep(wait_time)
            else:
                # Final attempt failed
                status_code = getattr(e.response, "status_code", None) if hasattr(e, "response") else None
                raise RuntimeError(
                    f"Failed to download {url} after {retries} attempts. "
                    f"Status code: {status_code}, Error: {str(e)}"
                ) from e

    # Should not reach here, but just in case
    if last_error:
        raise RuntimeError(f"Failed to download {url}: {str(last_error)}") from last_error


def download_range(
    out_dir: str = "data/raw/eccc/toronto_pearson",
    station_id: int = 51459,
    start_year: int = 2019,
    end_year: int = 2025,
    force: bool = False,
) -> list[Path]:
    """
    Download ECCC monthly CSVs for a range of years.

    Args:
        out_dir: Directory to save CSV files
        station_id: ECCC station ID
        start_year: First year to download (inclusive)
        end_year: Last year to download (inclusive)
        force: If True, re-download existing files

    Returns:
        List of paths to downloaded CSV files
    """
    out_path = Path(out_dir)
    downloaded_paths = []

    for year in range(start_year, end_year + 1):
        for month in range(1, 13):
            url = build_bulk_url(station_id, year, month)
            file_path = out_path / str(year) / f"{month:02d}.csv"

            print(f"Downloading {year}-{month:02d}...", end=" ", flush=True)
            try:
                download_month_csv(url, file_path, force=force)
                downloaded_paths.append(file_path)
                print("✓")
            except Exception as e:
                print(f"✗ Error: {e}")
                raise

    return downloaded_paths


def _norm_col(s: str) -> str:
    """
    Strong column normalization helper.
    
    Rules:
    - lowercase
    - strip whitespace
    - replace unicode degree symbol and variants: "°" and "º" -> "deg"
    - remove stray "Â" characters
    - remove punctuation (parentheses, dots, slashes, percent) by replacing with spaces
    - collapse multiple spaces to single space
    
    Args:
        s: Column name string
        
    Returns:
        Normalized string
    """
    # Lowercase and strip
    norm = s.lower().strip()
    
    # Replace degree symbols
    norm = norm.replace("°", "deg")
    norm = norm.replace("º", "deg")
    
    # Remove stray "Â" characters
    norm = norm.replace("â", "")
    norm = norm.replace("Â", "")
    
    # Replace punctuation with spaces
    # Remove: parentheses, dots, slashes, percent signs, commas
    norm = re.sub(r'[()./,%]', ' ', norm)
    
    # Collapse multiple spaces to single space
    norm = re.sub(r'\s+', ' ', norm)
    
    return norm.strip()


def _find_col(cols: list[str], required_tokens: list[str]) -> Optional[str]:
    """
    Find column using token-based matching.
    
    For each column name, compute normalized version.
    If all required_tokens appear as substrings in normalized version, return that column.
    
    Args:
        cols: List of column names to search
        required_tokens: List of tokens that must all appear in normalized column name
        
    Returns:
        Column name if found, None otherwise
    """
    for col in cols:
        norm = _norm_col(col)
        # Check if all required tokens appear in normalized string
        if all(token.lower() in norm for token in required_tokens):
            return col
    return None


def _to_float(series: pd.Series) -> pd.Series:
    """
    Robustly convert a pandas Series to float using regex extraction.
    
    Args:
        series: Input series (may contain strings, numbers, or mixed types)
        
    Returns:
        Series of floats with NaN for unparseable values
    """
    # Keep as string, but preserve missingness
    s = series.astype("string")
    
    # Strip whitespace and quotes
    s = s.str.strip().str.strip('"').str.strip("'")
    
    # Normalize unicode minus signs
    s = s.str.replace("\u2212", "-", regex=False).str.replace("−", "-", regex=False)
    
    # Remove NBSP and zero-width spaces (common hidden chars)
    s = s.str.replace("\u00A0", "", regex=False).str.replace("\u200b", "", regex=False)
    
    # Convert decimal comma to dot
    s = s.str.replace(",", ".", regex=False)
    
    # Treat common missing tokens as <NA>
    s = s.replace({"": pd.NA, "NA": pd.NA, "N/A": pd.NA, "nan": pd.NA, "None": pd.NA})
    
    # Extract first numeric token (handles cases like '4.5', ' 4.5 ', '4.5*', etc.)
    extracted = s.str.extract(r"([-+]?\d*\.?\d+)", expand=False)
    
    return pd.to_numeric(extracted, errors="coerce")


def read_eccc_hourly_csv(path: str, raw_dir: str = "data/raw/eccc/toronto_pearson", debug: bool = False) -> pd.DataFrame:
    """
    Parse an ECCC hourly CSV file into standardized format.

    Args:
        path: Path to CSV file
        raw_dir: Raw directory path for computing relative source_file

    Returns:
        DataFrame with standardized columns:
        - timestamp (index, datetime64[ns])
        - temp_c, dewpoint_c, rel_hum_pct, wind_spd_kmh, precip_mm, stn_press_kpa (float)
        - source_file (str, relative path from raw_dir)
        - is_imputed_weather (bool, initially False)

    Raises:
        ValueError: If datetime column cannot be found
    """
    path_obj = Path(path)
    raw_dir_path = Path(raw_dir)
    
    # Set source_file as relative path from raw_dir (e.g., "2022/08.csv")
    try:
        source_file = str(path_obj.relative_to(raw_dir_path))
    except ValueError:
        # Fallback to year/month if relative path fails
        # Try to extract from path structure
        parts = path_obj.parts
        if len(parts) >= 2:
            source_file = f"{parts[-2]}/{parts[-1]}"
        else:
            source_file = path_obj.name

    # Try reading with different encodings (utf-8-sig first for BOM handling)
    df = None
    for encoding in ["utf-8-sig", "utf-8", "latin-1"]:
        try:
            df = pd.read_csv(path, encoding=encoding, low_memory=False)
            break
        except UnicodeDecodeError:
            continue

    if df is None:
        raise ValueError(f"Could not read {path} with any supported encoding")

    if df.empty:
        # Return empty DataFrame with correct schema
        return pd.DataFrame(
            columns=[
                "temp_c",
                "dewpoint_c",
                "rel_hum_pct",
                "wind_spd_kmh",
                "precip_mm",
                "stn_press_kpa",
                "source_file",
                "is_imputed_weather",
            ]
        ).set_index(pd.DatetimeIndex([]))

    # Find datetime column using token-based matching
    datetime_col = _find_col(list(df.columns), ["date", "time"])
    if datetime_col is None:
        raise ValueError(
            f"Could not find datetime column in {path}. "
            f"Available columns: {list(df.columns)}"
        )

    # Parse timestamps
    ts = pd.to_datetime(df[datetime_col], errors="coerce")
    
    # Create mask for valid timestamps (ONLY filter on timestamp NaT)
    mask_valid = ts.notna()
    
    # Filter both df and ts using mask
    df = df.loc[mask_valid].copy()
    ts = ts.loc[mask_valid]
    
    # IMPORTANT: Reset both indices to ensure alignment
    df.reset_index(drop=True, inplace=True)
    ts = ts.reset_index(drop=True)

    if df.empty or len(ts) == 0:
        # Return empty DataFrame with correct schema
        return pd.DataFrame(
            columns=[
                "temp_c",
                "dewpoint_c",
                "rel_hum_pct",
                "wind_spd_kmh",
                "precip_mm",
                "stn_press_kpa",
                "source_file",
                "is_imputed_weather",
            ]
        ).set_index(pd.DatetimeIndex([]))

    # Column mapping using token-based matching
    column_mapping = {}
    
    # Find columns using token-based matching
    temp_col = _find_col(list(df.columns), ["temp", "degc"])
    if temp_col and "temp_c" not in column_mapping.values():
        column_mapping[temp_col] = "temp_c"
    
    dewpoint_col = _find_col(list(df.columns), ["dew", "point", "temp", "degc"])
    if dewpoint_col and "dewpoint_c" not in column_mapping.values():
        column_mapping[dewpoint_col] = "dewpoint_c"
    
    rel_hum_col = _find_col(list(df.columns), ["rel", "hum"])
    if rel_hum_col and "rel_hum_pct" not in column_mapping.values():
        column_mapping[rel_hum_col] = "rel_hum_pct"
    
    wind_col = _find_col(list(df.columns), ["wind", "spd"])
    if wind_col and "wind_spd_kmh" not in column_mapping.values():
        column_mapping[wind_col] = "wind_spd_kmh"
    
    precip_col = _find_col(list(df.columns), ["precip", "amount"])
    if precip_col and "precip_mm" not in column_mapping.values():
        column_mapping[precip_col] = "precip_mm"
    
    press_col = _find_col(list(df.columns), ["stn", "press"])
    if press_col and "stn_press_kpa" not in column_mapping.values():
        column_mapping[press_col] = "stn_press_kpa"

    # Extract numeric columns from filtered df (after reset)
    # Convert to numpy arrays to avoid index alignment issues
    data_dict = {}
    
    # Map columns using robust numeric conversion
    for orig_col, std_col in column_mapping.items():
        series = df[orig_col].copy()
        
        # Enhanced debug output for temp column (first file only)
        if debug and std_col == "temp_c" and temp_col:
            print(f"\n[DEBUG] Converting {temp_col} -> {std_col}:")
            raw_strs = series.head(10).astype("string").tolist()
            print(f"  First 10 raw strings:")
            for i, val in enumerate(raw_strs):
                print(f"    [{i}] {repr(val)}")
            
            # Apply conversion steps manually for debug
            s_debug = series.astype("string")
            s_debug = s_debug.str.strip().str.strip('"').str.strip("'")
            s_debug = s_debug.str.replace("\u2212", "-", regex=False).str.replace("−", "-", regex=False)
            s_debug = s_debug.str.replace("\u00A0", "", regex=False).str.replace("\u200b", "", regex=False)
            s_debug = s_debug.str.replace(",", ".", regex=False)
            s_debug = s_debug.replace({"": pd.NA, "NA": pd.NA, "N/A": pd.NA, "nan": pd.NA, "None": pd.NA})
            extracted_debug = s_debug.str.extract(r"([-+]?\d*\.?\d+)", expand=False)
            extracted_strs = extracted_debug.head(10).tolist()
            print(f"  First 10 extracted numeric tokens:")
            for i, val in enumerate(extracted_strs):
                print(f"    [{i}] {repr(val)}")
            
            # Final conversion
            converted = pd.to_numeric(extracted_debug, errors="coerce")
            final_floats = converted.head(10).tolist()
            print(f"  First 10 floats after pd.to_numeric:")
            for i, val in enumerate(final_floats):
                print(f"    [{i}] {repr(val)}")
        
        # Convert to float and reset index
        numeric_series = _to_float(series)
        numeric_series = numeric_series.reset_index(drop=True)
        
        # Convert to numpy array with explicit dtype and na_value
        data_dict[std_col] = numeric_series.to_numpy(dtype=float, na_value=np.nan)

    # Add required columns (fill with NaN arrays if missing)
    required_cols = [
        "temp_c",
        "dewpoint_c",
        "rel_hum_pct",
        "wind_spd_kmh",
        "precip_mm",
        "stn_press_kpa",
    ]
    n_rows = len(ts)
    for col in required_cols:
        if col not in data_dict:
            data_dict[col] = np.full(n_rows, np.nan, dtype=float)

    # Add metadata columns
    data_dict["source_file"] = [source_file] * n_rows
    data_dict["is_imputed_weather"] = [False] * n_rows

    # Build DataFrame using arrays to avoid alignment issues
    result = pd.DataFrame(data_dict)
    
    # Set timestamp as index
    result.index = ts
    result.index.name = "timestamp"

    # Ensure columns are in correct order
    result = result[
        [
            "temp_c",
            "dewpoint_c",
            "rel_hum_pct",
            "wind_spd_kmh",
            "precip_mm",
            "stn_press_kpa",
            "source_file",
            "is_imputed_weather",
        ]
    ]
    
    # Debug output after parsing first file
    if debug:
        print(f"\n[DEBUG] Post-parse (file: {path}):")
        print(f"  Detected dt_col: {datetime_col}")
        print(f"  Detected temp_col: {temp_col}")
        print(f"  Rows parsed: {len(result):,}")
        
        # Show dtype and values after building DataFrame
        if len(result) > 0:
            print(f"  temp_c dtype: {result['temp_c'].dtype}")
            print(f"  temp_c head(5): {result['temp_c'].head(5).tolist()}")
            temp_nonnull = result["temp_c"].notna().sum()
            print(f"  temp_c non-null count: {temp_nonnull:,}")
            
            print(f"  First 5 rows (timestamp, temp_c):")
            for idx in result.index[:5]:
                temp_val = result.loc[idx, "temp_c"]
                print(f"    {idx}: temp_c={temp_val}")

    return result


def build_weather_hourly(
    raw_dir: str = "data/raw/eccc/toronto_pearson",
    start_year: int = 2019,
    end_year: int = 2025,
    debug_columns: bool = False,
) -> Tuple[pd.DataFrame, Dict]:
    """
    Build complete hourly weather DataFrame from all monthly CSVs.

    Args:
        raw_dir: Directory containing monthly CSV files
        start_year: First year to process (inclusive)
        end_year: Last year to process (inclusive)

    Returns:
        DataFrame with complete hourly index and imputed missing values

    Raises:
        ValueError: If duplicate timestamps are found
        FileNotFoundError: If no CSV files are found
    """
    raw_path = Path(raw_dir)

    # Find all CSV files for years in range
    csv_files = []
    for year in range(start_year, end_year + 1):
        year_dir = raw_path / str(year)
        if year_dir.exists():
            csv_files.extend(year_dir.glob("*.csv"))

    if not csv_files:
        raise FileNotFoundError(
            f"No CSV files found in {raw_dir} for years {start_year}-{end_year}"
        )

    # Debug mode: print columns from first CSV
    if debug_columns and csv_files:
        first_csv = sorted(csv_files)[0]
        print(f"\n[DEBUG] First CSV file: {first_csv}")
        try:
            # Try to read just to see columns
            for encoding in ["utf-8-sig", "utf-8", "latin-1"]:
                try:
                    sample_df = pd.read_csv(first_csv, encoding=encoding, low_memory=False, nrows=0)
                    print(f"  Columns (as read): {[repr(col) for col in sample_df.columns]}")
                    print(f"  Normalized columns: {[_norm_col(col) for col in sample_df.columns]}")
                    break
                except UnicodeDecodeError:
                    continue
        except Exception as e:
            print(f"  Error reading for debug: {e}")

    # Load all CSVs
    dfs = []
    first_file_path = None
    first_file_temp_col = None
    
    for i, csv_file in enumerate(sorted(csv_files)):
        try:
            # Pass debug flag only for first file
            is_first_file = (i == 0) and debug_columns
            df = read_eccc_hourly_csv(str(csv_file), raw_dir=raw_dir, debug=is_first_file)
            if not df.empty:
                dfs.append(df)
                if is_first_file:
                    first_file_path = str(csv_file)
                    # Re-detect temp_col for debug message
                    try:
                        sample_df = pd.read_csv(first_file_path, encoding="utf-8-sig", low_memory=False, nrows=0)
                        for enc in ["utf-8", "latin-1"]:
                            try:
                                sample_df = pd.read_csv(first_file_path, encoding=enc, low_memory=False, nrows=0)
                                break
                            except UnicodeDecodeError:
                                continue
                        first_file_temp_col = _find_col(list(sample_df.columns), ["temp", "degc"])
                    except:
                        pass
                    
        except Exception as e:
            print(f"Warning: Failed to parse {csv_file}: {e}")
            continue

    if not dfs:
        raise ValueError("No valid data found in any CSV files")

    # Concatenate
    combined = pd.concat(dfs, ignore_index=False)
    combined = combined.sort_index()

    # Check for duplicates
    duplicates = combined.index.duplicated(keep=False)
    if duplicates.any():
        dup_count = duplicates.sum()
        dup_examples = combined.index[duplicates].unique()[:5]
        raise ValueError(
            f"Found {dup_count} duplicate timestamps. "
            f"Examples: {list(dup_examples)}"
        )

    # Check that we actually parsed temperature data
    weather_cols = [
        "temp_c",
        "dewpoint_c",
        "rel_hum_pct",
        "wind_spd_kmh",
        "precip_mm",
        "stn_press_kpa",
    ]
    
    # Check that we actually parsed temperature data BEFORE reindexing
    total_rows_raw = len(combined)
    observed_temp_nonnull_pre_reindex = combined["temp_c"].notna().sum()
    
    if observed_temp_nonnull_pre_reindex == 0:
        error_msg = (
            f"Parsing failed: no temperature data found in any CSV files. "
            f"Total raw rows: {total_rows_raw:,}. "
        )
        if first_file_path:
            error_msg += f"First file: {first_file_path}. "
        if first_file_temp_col:
            error_msg += f"Detected temp_col: {first_file_temp_col}."
        raise RuntimeError(error_msg)
    
    # Track which timestamps had missing temp BEFORE any filling (before reindexing)
    temp_missing_pre_fill_timestamps = set(
        combined.index[combined["temp_c"].isna()]
    )
    
    # Store for summary
    total_rows_pre_reindex = len(combined)
    observed_temp_pct_pre_reindex = 100 * observed_temp_nonnull_pre_reindex / total_rows_pre_reindex if total_rows_pre_reindex > 0 else 0

    # Reindex to complete hourly range
    if combined.empty:
        raise ValueError("No data to process")

    min_ts = combined.index.min()
    max_ts = combined.index.max()
    complete_index = pd.date_range(start=min_ts, end=max_ts, freq="h")

    # Track which rows are new (created by reindexing)
    original_timestamps = set(combined.index)

    # Reindex
    combined = combined.reindex(complete_index)

    # Mark newly created rows (rows created by reindexing)
    new_rows = combined["source_file"].isna()
    combined.loc[new_rows, "source_file"] = "IMPUTED"
    
    # Set is_imputed_weather flag BEFORE filling
    # Mark as imputed if:
    # 1. Row was created by reindexing (new_rows)
    # 2. Original timestamp had missing temp before reindexing
    combined["is_imputed_weather"] = False
    rows_with_original_missing_temp = combined.index.isin(temp_missing_pre_fill_timestamps)
    combined.loc[new_rows | rows_with_original_missing_temp, "is_imputed_weather"] = True

    # NOW impute missing values (after setting flags)
    # First: time-based interpolation
    for col in weather_cols:
        combined[col] = combined[col].interpolate(method="time")

    # Then: forward fill (using modern pandas methods)
    combined[weather_cols] = combined[weather_cols].ffill()

    # Then: backward fill
    combined[weather_cols] = combined[weather_cols].bfill()

    # Return DataFrame and summary stats
    summary_stats = {
        "observed_temp_nonnull": observed_temp_nonnull_pre_reindex,
        "observed_temp_pct": observed_temp_pct_pre_reindex,
        "total_rows_pre_reindex": total_rows_pre_reindex,
    }
    
    return combined, summary_stats


def main(argv=None) -> None:
    """CLI entry point for ECCC weather ingestion."""
    parser = argparse.ArgumentParser(
        description="Download and process ECCC weather data for Toronto Pearson"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download existing files",
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Skip download step, only process existing files",
    )
    parser.add_argument(
        "--raw-dir",
        default="data/raw/eccc/toronto_pearson",
        help="Directory for raw CSV files",
    )
    parser.add_argument(
        "--output",
        default="data/processed/eccc_hourly_toronto_pearson_2019_2025.parquet",
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
        help="Last year to process",
    )
    parser.add_argument(
        "--station-id",
        type=int,
        default=51459,
        help="ECCC station ID",
    )
    parser.add_argument(
        "--debug-columns",
        action="store_true",
        help="Print column debugging information",
    )

    args = parser.parse_args(argv)

    # Download if not skipped
    if not args.skip_download:
        print(f"Downloading ECCC weather data for {args.start_year}-{args.end_year}...")
        download_range(
            out_dir=args.raw_dir,
            station_id=args.station_id,
            start_year=args.start_year,
            end_year=args.end_year,
            force=args.force,
        )
        print("Download complete.\n")

    # Process
    print("Processing weather data...")
    # Find CSV files for summary
    raw_path = Path(args.raw_dir)
    csv_files = []
    for year in range(args.start_year, args.end_year + 1):
        year_dir = raw_path / str(year)
        if year_dir.exists():
            csv_files.extend(year_dir.glob("*.csv"))
    
    df, summary_stats = build_weather_hourly(
        raw_dir=args.raw_dir,
        start_year=args.start_year,
        end_year=args.end_year,
        debug_columns=args.debug_columns,
    )

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
    print(f"  Raw files found: {len(csv_files)}")
    print(f"  Unique source files: {df['source_file'].nunique()}")
    print(f"  Start timestamp: {df['timestamp'].min()}")
    print(f"  End timestamp: {df['timestamp'].max()}")
    print(f"  Total rows: {len(df):,}")
    print(f"  Observed temp (non-null, pre-fill): {summary_stats['observed_temp_nonnull']:,} ({summary_stats['observed_temp_pct']:.1f}%)")

    imputed_count = df["is_imputed_weather"].sum()
    print(f"  Imputed rows: {imputed_count:,} ({100 * imputed_count / len(df):.1f}%)")

    # Check coverage by year
    print("\n  Coverage by year:")
    for year in range(args.start_year, args.end_year + 1):
        year_data = df[df["timestamp"].dt.year == year]
        if year_data.empty:
            print(f"    {year}: No data")
            continue

        expected_hours = 8760 if year % 4 != 0 else 8784  # Account for leap years
        actual_hours = len(year_data)
        coverage_pct = 100 * actual_hours / expected_hours

        if coverage_pct < 100:
            print(
                f"    {year}: {actual_hours:,} hours ({coverage_pct:.1f}%) - PARTIAL"
            )
        else:
            print(f"    {year}: {actual_hours:,} hours ({coverage_pct:.1f}%) - Complete")


if __name__ == "__main__":
    main()

