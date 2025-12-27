"""
Simple baseline models for comparison.
SeasonalNaive168, LinearWeatherRidge, LagRidge.
"""

from typing import List, Optional

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge


def make_years_since_2019(ts: pd.Series) -> np.ndarray:
    """
    Convert timestamp series to years since 2019-01-01.

    Args:
        ts: Series of timestamps

    Returns:
        Array of years since 2019-01-01
    """
    base_date = pd.Timestamp("2019-01-01")
    return (ts - base_date).dt.total_seconds() / (365.25 * 24 * 3600)


class SeasonalNaive168:
    """
    Predicts using same hour last week (168h lag). No training needed.
    """

    def __init__(self):
        self.name = "SeasonalNaive168"

    def fit(self, df: pd.DataFrame) -> "SeasonalNaive168":
        """No-op. This model doesn't learn anything."""
        return self

    def predict(self, df: pd.DataFrame) -> pd.Series:
        """
        Returns demand_lag_168h as predictions.
        """
        if "demand_lag_168h" not in df.columns:
            raise ValueError("demand_lag_168h column required for SeasonalNaive168")

        y_hat = df["demand_lag_168h"].copy()
        y_hat.name = "y_hat"
        return y_hat


class LinearWeatherRidge:
    """
    Ridge regression with weather (HDD/CDD) and time features (hour, day of week, etc).
    """

    def __init__(self, alpha: float = 1.0, include_trend: bool = True):
        """
        alpha: regularization strength
        include_trend: add "years since 2019" feature
        """
        self.alpha = alpha
        self.include_trend = include_trend
        self.name = "LinearWeatherRidge"
        self.model: Optional[Ridge] = None
        self.feature_names_: List[str] = []
        self.coef_: Optional[np.ndarray] = None
        self.intercept_: Optional[float] = None

    def fit(self, df: pd.DataFrame) -> "LinearWeatherRidge":
        """
        Train Ridge regression. Needs base features (hdd_18, cdd_18, hour_sin, etc) and timestamp if include_trend=True.
        """
        # Define base feature set (must exist in df)
        base_features = [
            "hdd_18",
            "cdd_18",
            "hour_sin",
            "hour_cos",
            "doy_sin",
            "doy_cos",
            "is_weekend",
        ]

        # Check for missing base features
        missing_features = [f for f in base_features if f not in df.columns]
        if missing_features:
            raise ValueError(f"Missing required features: {missing_features}")

        # Build X from base features
        X = df[base_features].copy()

        # Add trend if requested (compute from timestamp, don't require in df)
        if self.include_trend:
            if "timestamp" not in df.columns:
                raise ValueError("timestamp column required when include_trend=True")
            X["years_since_2019"] = make_years_since_2019(df["timestamp"])
            self.feature_names_ = base_features + ["years_since_2019"]
        else:
            self.feature_names_ = base_features.copy()

        # Prepare y
        y = df["y"].copy()

        # Drop rows with any NaN in X or y
        mask = ~(X.isna().any(axis=1) | y.isna())
        X = X[mask]
        y = y[mask]

        if len(X) == 0:
            raise ValueError("No valid training samples after dropping NaNs")

        # Fit model (use .values to avoid index alignment issues)
        self.model = Ridge(alpha=self.alpha)
        self.model.fit(X.values, y.values)

        # Store coefficients
        self.coef_ = self.model.coef_
        self.intercept_ = self.model.intercept_

        return self

    def predict(self, df: pd.DataFrame) -> pd.Series:
        """
        Predict demand. Needs same features as fit(), plus timestamp if include_trend=True.
        """
        if self.model is None:
            raise ValueError("Model must be fitted before prediction")

        # Define base features (same as in fit)
        base_features = [
            "hdd_18",
            "cdd_18",
            "hour_sin",
            "hour_cos",
            "doy_sin",
            "doy_cos",
            "is_weekend",
        ]

        # Check for missing base features
        missing_features = [f for f in base_features if f not in df.columns]
        if missing_features:
            raise ValueError(f"Missing required features: {missing_features}")

        # Build X from base features
        X = df[base_features].copy()

        # Add trend if included (compute from timestamp, don't require in df)
        if self.include_trend:
            if "timestamp" not in df.columns:
                raise ValueError("timestamp column required when include_trend=True")
            X["years_since_2019"] = make_years_since_2019(df["timestamp"])

        # Predict (use .values to avoid index alignment issues)
        y_hat = self.model.predict(X.values)
        y_hat = pd.Series(y_hat, index=df.index, name="y_hat")

        return y_hat


class LagRidge:
    """
    Ridge regression using lag features (previous hour, yesterday, last week, rolling means).
    """

    def __init__(self, alpha: float = 1.0):
        """alpha: regularization strength"""
        self.alpha = alpha
        self.name = "LagRidge"
        self.model: Optional[Ridge] = None
        self.feature_names: List[str] = []
        self.coef_: Optional[np.ndarray] = None
        self.intercept_: Optional[float] = None

    def fit(self, df: pd.DataFrame) -> "LagRidge":
        """
        Train Ridge regression on lag features (demand_lag_1h, demand_lag_24h, etc).
        """
        # Define feature set
        lag_features = [
            "demand_lag_1h",
            "demand_lag_24h",
            "demand_lag_168h",
            "demand_rollmean_24h",
            "demand_rollmean_168h",
        ]

        # Check which features exist
        available_features = [f for f in lag_features if f in df.columns]
        if not available_features:
            raise ValueError("No lag features found in DataFrame")

        self.feature_names = available_features

        # Prepare X and y
        X = df[self.feature_names].copy()
        y = df["y"].copy()

        # Drop rows with any NaN in X or y
        mask = ~(X.isna().any(axis=1) | y.isna())
        X = X[mask]
        y = y[mask]

        if len(X) == 0:
            raise ValueError("No valid training samples after dropping NaNs")

        # Fit model
        self.model = Ridge(alpha=self.alpha)
        self.model.fit(X, y)

        # Store coefficients
        self.coef_ = self.model.coef_
        self.intercept_ = self.model.intercept_

        return self

    def predict(self, df: pd.DataFrame) -> pd.Series:
        """
        Predict demand using lag features.
        """
        if self.model is None:
            raise ValueError("Model must be fitted before prediction")

        # Prepare features
        X = df[self.feature_names].copy()

        # Predict
        y_hat = self.model.predict(X)
        y_hat = pd.Series(y_hat, index=df.index, name="y_hat")

        return y_hat

