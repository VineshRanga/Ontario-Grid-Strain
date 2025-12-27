"""
Smoke Tests

Lightweight unittest suite for data quality checks.
"""

import unittest
from pathlib import Path

import pandas as pd
import pyarrow

from src.utils.contracts import (
    ECCC_PROCESSED_REQUIRED_COLS,
    IESO_PROCESSED_REQUIRED_COLS,
    MODEL_DATASET_REQUIRED_COLS,
    require_columns,
    require_unique_timestamp,
)


class TestRawIESOFiles(unittest.TestCase):
    """Test that raw IESO files exist."""

    def test_all_ieso_files_exist(self):
        """Assert all 2019-2025 raw IESO files exist."""
        raw_dir = Path("data/raw/ieso")
        for year in range(2019, 2026):
            file_path = raw_dir / f"PUB_Demand_{year}.csv"
            self.assertTrue(
                file_path.exists(),
                f"Missing IESO raw file: {file_path}",
            )


class TestIESOCSVHeaders(unittest.TestCase):
    """Test IESO CSV header sanity."""

    def test_ieso_csv_headers(self):
        """For each raw file, confirm required header fields appear."""
        raw_dir = Path("data/raw/ieso")
        required_headers = ["Date", "Hour", "Market Demand", "Ontario Demand"]

        for year in range(2019, 2026):
            file_path = raw_dir / f"PUB_Demand_{year}.csv"
            if not file_path.exists():
                self.skipTest(f"File not found: {file_path}")

            # Read first few lines to find header
            with open(file_path, "r") as f:
                lines = []
                for i, line in enumerate(f):
                    if i >= 10:
                        break
                    lines.append(line)

            # Find header line
            header_line = None
            for line in lines:
                if not line.strip().startswith("\\"):
                    header_line = line.strip()
                    break

            self.assertIsNotNone(
                header_line, f"Could not find header in {file_path}"
            )

            # Check required headers
            headers = [h.strip() for h in header_line.split(",")]
            for required in required_headers:
                self.assertIn(
                    required,
                    headers,
                    f"{file_path} missing required header: {required}",
                )


class TestProcessedIESO(unittest.TestCase):
    """Test processed IESO parquet if it exists."""

    @unittest.skipUnless(
        Path("data/processed/ieso_hourly_2019_2025.parquet").exists(),
        "IESO processed file not found",
    )
    def test_ieso_schema(self):
        """Validate schema and no NaNs in ontario_demand_mw."""
        file_path = Path("data/processed/ieso_hourly_2019_2025.parquet")
        df = pd.read_parquet(file_path)

        # Validate columns
        require_columns(df, IESO_PROCESSED_REQUIRED_COLS, "IESO processed")

        # Validate unique timestamps
        require_unique_timestamp(df, "IESO processed")

        # Check for NaNs in ontario_demand_mw
        nan_count = df["ontario_demand_mw"].isna().sum()
        self.assertEqual(
            nan_count,
            0,
            f"Found {nan_count} NaN values in ontario_demand_mw",
        )


class TestModelDataset(unittest.TestCase):
    """Test model dataset if it exists."""

    @unittest.skipUnless(
        Path("data/processed/model_dataset_hourly_2019_2025.parquet").exists(),
        "Model dataset not found",
    )
    def test_model_dataset_schema(self):
        """Validate schema."""
        file_path = Path("data/processed/model_dataset_hourly_2019_2025.parquet")
        df = pd.read_parquet(file_path)

        # Validate columns
        require_columns(df, MODEL_DATASET_REQUIRED_COLS, "Model dataset")

        # Validate unique timestamps
        require_unique_timestamp(df, "Model dataset")

    @unittest.skipUnless(
        Path("data/processed/model_dataset_hourly_2019_2025.parquet").exists(),
        "Model dataset not found",
    )
    def test_split_labels(self):
        """Validate split labels are correct."""
        file_path = Path("data/processed/model_dataset_hourly_2019_2025.parquet")
        df = pd.read_parquet(file_path)

        # Check split column exists
        self.assertIn("split", df.columns, "Missing 'split' column")

        # Check split values are valid
        valid_splits = {"train", "val", "test"}
        actual_splits = set(df["split"].unique())
        invalid_splits = actual_splits - valid_splits
        self.assertEqual(
            len(invalid_splits),
            0,
            f"Invalid split values: {invalid_splits}",
        )

        # Check temporal split assignments
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df["year"] = df["timestamp"].dt.year

        # Train should contain 2019-2023
        train_years = set(df[df["split"] == "train"]["year"].unique())
        expected_train_years = {2019, 2020, 2021, 2022, 2023}
        train_intersection = train_years & expected_train_years
        self.assertGreater(
            len(train_intersection),
            0,
            "Train split should contain years 2019-2023",
        )

        # Val should contain 2024
        val_years = set(df[df["split"] == "val"]["year"].unique())
        self.assertIn(
            2024,
            val_years,
            "Val split should contain year 2024",
        )

        # Test should contain 2025 (if present in data)
        if 2025 in df["year"].unique():
            test_years = set(df[df["split"] == "test"]["year"].unique())
            self.assertIn(
                2025,
                test_years,
                "Test split should contain year 2025",
            )


def main() -> None:
    """Run smoke tests."""
    unittest.main(verbosity=2)


if __name__ == "__main__":
    main()

