"""
Generate AI load scenarios. Yearly MW series with YoY growth, hourly adders with different ramp shapes.
"""

import json
from pathlib import Path
from typing import List, Dict

import numpy as np
import pandas as pd


def yearly_mw_series(
    start_year: int,
    end_year: int,
    start_mw: float,
    yoy_growth: float,
) -> pd.Series:
    """
    Generate yearly MW series with YoY growth.

    Args:
        start_year: First year of scenario
        end_year: Last year of scenario (inclusive)
        start_mw: MW at start of start_year
        yoy_growth: Year-over-year growth rate (e.g., 0.07 for 7%)

    Returns:
        Series with year as index (int dtype) and MW as values
    """
    years = list(range(start_year, end_year + 1))
    vals = []
    for i, y in enumerate(years):
        vals.append(start_mw * ((1.0 + yoy_growth) ** i))
    
    return pd.Series(vals, index=pd.Index(years, dtype=int, name="year"), name="mw")


def hourly_ai_adder(
    index: pd.DatetimeIndex,
    yearly_mw: pd.Series,
    ramp_shape: str = "flat",
    allow_missing_years: bool = False,
) -> pd.Series:
    """
    Generate hourly AI load adder for given timestamps.

    Args:
        index: DatetimeIndex of hourly timestamps
        yearly_mw: Series with year as index and MW as values
        ramp_shape: One of "flat", "linear_within_year", "step_quarterly"
        allow_missing_years: If True, fill missing years with 0.0; if False, raise ValueError

    Returns:
        Series of MW adders aligned to index

    Raises:
        ValueError: If years in index are not found in yearly_mw and allow_missing_years=False
    """
    # Coerce yearly_mw index to int at the start
    yearly_mw = yearly_mw.copy()
    yearly_mw.index = yearly_mw.index.astype(int)
    
    # Extract years from index as int
    yrs = index.year.astype(int)
    
    # Check for missing years
    missing_years = sorted(set(yrs) - set(yearly_mw.index))
    if missing_years and not allow_missing_years:
        raise ValueError(
            f"Years {missing_years} in index not found in yearly_mw index {sorted(yearly_mw.index.tolist())}. "
            f"Check scenario start_year/end_year configuration."
        )

    if ramp_shape == "flat":
        # Use direct lookups to avoid Series.map issues
        out = np.zeros(len(index), dtype=float)
        for i, y in enumerate(yrs):
            if y in yearly_mw.index:
                out[i] = float(yearly_mw.loc[y])
            else:
                out[i] = 0.0
        return pd.Series(out, index=index, name="ai_adder_mw")

    elif ramp_shape == "linear_within_year":
        # Linear ramp from year MW at Jan 1 to next-year MW by Dec 31
        result = pd.Series(0.0, index=index, name="ai_adder_mw")
        
        for year_int in yearly_mw.index:
            year = int(year_int)  # Ensure int
            year_mask = yrs == year
            year_timestamps = index[year_mask]

            if len(year_timestamps) == 0:
                continue

            # Get start and end MW using direct lookup
            start_mw = float(yearly_mw.loc[year])

            # Get next year MW (or hold flat if missing)
            next_year = year + 1
            if next_year in yearly_mw.index:
                end_mw = float(yearly_mw.loc[next_year])
            else:
                end_mw = start_mw  # Hold flat

            # Linear interpolation within year
            year_start = pd.Timestamp(f"{year}-01-01 00:00:00")
            year_end = pd.Timestamp(f"{year}-12-31 23:00:00")
            total_hours = (year_end - year_start).total_seconds() / 3600

            for ts in year_timestamps:
                hours_from_start = (ts - year_start).total_seconds() / 3600
                fraction = hours_from_start / total_hours if total_hours > 0 else 0.0
                result.loc[ts] = start_mw + fraction * (end_mw - start_mw)
        
        return result

    elif ramp_shape == "step_quarterly":
        # Step changes at Q1/Q2/Q3/Q4 boundaries
        result = pd.Series(0.0, index=index, name="ai_adder_mw")
        
        for year_int in yearly_mw.index:
            year = int(year_int)  # Ensure int
            year_mask = yrs == year
            year_timestamps = index[year_mask]

            if len(year_timestamps) == 0:
                continue

            # Get start MW using direct lookup
            start_mw = float(yearly_mw.loc[year])

            # Get next year MW (or hold flat)
            next_year = year + 1
            if next_year in yearly_mw.index:
                end_mw = float(yearly_mw.loc[next_year])
            else:
                end_mw = start_mw

            # Define quarter boundaries
            q1_end = pd.Timestamp(f"{year}-03-31 23:00:00")
            q2_end = pd.Timestamp(f"{year}-06-30 23:00:00")
            q3_end = pd.Timestamp(f"{year}-09-30 23:00:00")
            q4_end = pd.Timestamp(f"{year}-12-31 23:00:00")

            # Assign MW by quarter
            for ts in year_timestamps:
                if ts <= q1_end:
                    # Q1: start_mw
                    result.loc[ts] = start_mw
                elif ts <= q2_end:
                    # Q2: 1/3 of the way
                    result.loc[ts] = start_mw + (end_mw - start_mw) * (1 / 3)
                elif ts <= q3_end:
                    # Q3: 2/3 of the way
                    result.loc[ts] = start_mw + (end_mw - start_mw) * (2 / 3)
                else:
                    # Q4: end_mw
                    result.loc[ts] = end_mw
        
        return result

    else:
        raise ValueError(
            f"Unknown ramp_shape: {ramp_shape}. "
            f"Must be one of: flat, linear_within_year, step_quarterly"
        )


def get_default_scenarios() -> List[Dict]:
    """
    Get default AI load scenarios.

    Returns:
        List of scenario dictionaries
    """
    return [
        {
            "name": "low",
            "start_year": 2026,
            "end_year": 2027,
            "start_mw": 200.0,
            "yoy_growth": 0.05,
        },
        {
            "name": "base",
            "start_year": 2026,
            "end_year": 2027,
            "start_mw": 350.0,
            "yoy_growth": 0.07,
        },
        {
            "name": "high",
            "start_year": 2026,
            "end_year": 2027,
            "start_mw": 800.0,
            "yoy_growth": 0.15,
        },
    ]


def save_scenarios_config(scenarios: List[Dict], path: str) -> None:
    """
    Save scenarios configuration to JSON file.

    Args:
        scenarios: List of scenario dictionaries
        path: Path to save JSON file
    """
    path_obj = Path(path)
    path_obj.parent.mkdir(parents=True, exist_ok=True)

    config = {
        "scenarios": scenarios,
        "metadata": {
            "description": "AI/data center load scenarios",
            "units": "MW",
        },
    }

    with open(path_obj, "w") as f:
        json.dump(config, f, indent=2)


def load_scenarios_config(path: str) -> List[Dict]:
    """
    Load scenarios configuration from JSON file.

    Args:
        path: Path to JSON file

    Returns:
        List of scenario dictionaries
    """
    path_obj = Path(path)
    if not path_obj.exists():
        raise FileNotFoundError(f"Scenarios config not found: {path}")

    with open(path_obj, "r") as f:
        config = json.load(f)

    if "scenarios" not in config:
        raise ValueError("Config file must contain 'scenarios' key")

    return config["scenarios"]

