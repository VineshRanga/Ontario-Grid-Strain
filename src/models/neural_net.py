"""
Neural network model using scikit-learn's MLPRegressor.
Handles feature selection, scaling, training, and prediction.
"""

import json
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler


def get_feature_columns(df: pd.DataFrame) -> List[str]:
    """
    Pick numeric columns, exclude metadata (timestamp, y, split, flags).
    Returns sorted list.
    """
    # Columns to exclude
    exclude_cols = {
        "timestamp",
        "y",
        "split",
        "source_file",
        "is_imputed",
        "is_imputed_weather",
        "weather_was_missing",
    }

    # Get numeric columns only
    feature_cols = []
    for col in df.columns:
        if col not in exclude_cols:
            # Check if numeric (integer, float, boolean)
            if df[col].dtype.kind in "ifb":
                feature_cols.append(col)

    # Return sorted for reproducibility
    return sorted(feature_cols)


def make_xy(df: pd.DataFrame, feature_cols: List[str]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract X (features) and y (target) as numpy arrays. Drops rows with any NaNs.
    """
    if "y" not in df.columns:
        raise ValueError("y column required in DataFrame")

    # Check all feature columns exist
    missing_cols = [col for col in feature_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing feature columns: {missing_cols}")

    # Select features and target
    X = df[feature_cols].copy()
    y = df["y"].copy()

    # Drop rows where y is NaN or any feature is NaN
    mask = ~(y.isna() | X.isna().any(axis=1))
    X = X[mask].values
    y = y[mask].values

    return X, y


def make_mlp(
    hidden_layer_sizes: Tuple[int, ...] = (64, 32),
    alpha: float = 1e-4,
    learning_rate_init: float = 1e-3,
    random_state: int = 42,
) -> MLPRegressor:
    """
    Create MLPRegressor with early stopping. Default: 2 hidden layers (64, 32), ReLU, Adam solver.
    """
    return MLPRegressor(
        hidden_layer_sizes=hidden_layer_sizes,
        activation="relu",
        solver="adam",
        alpha=alpha,
        learning_rate_init=learning_rate_init,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=10,
        max_iter=200,
        batch_size=256,
        random_state=random_state,
        verbose=False,
    )


def train_mlp(
    df_train: pd.DataFrame, feature_cols: List[str], params: Dict
) -> Dict:
    """
    Train MLPRegressor. Scales features first, then fits model.
    Returns dict with model, scaler, and feature_cols.
            - "feature_cols": feature_cols
    """
    # Build X, y
    X_train, y_train = make_xy(df_train, feature_cols)

    if len(X_train) == 0:
        raise ValueError("No valid training samples after dropping NaNs")

    # Fit scaler on training data only
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    # Create and fit MLP
    mlp = make_mlp(
        hidden_layer_sizes=params.get("hidden_layer_sizes", (64, 32)),
        alpha=params.get("alpha", 1e-4),
        learning_rate_init=params.get("learning_rate_init", 1e-3),
        random_state=params.get("random_state", 42),
    )

    mlp.fit(X_train_scaled, y_train)

    return {
        "model": mlp,
        "scaler": scaler,
        "feature_cols": feature_cols,
    }


def predict(trained: Dict, df: pd.DataFrame) -> pd.Series:
    """
    Predict using trained model. Scales features first, then predicts.
    Returns Series aligned to df index. NaN for rows with missing features.
    """
    model = trained["model"]
    scaler = trained["scaler"]
    feature_cols = trained["feature_cols"]

    # Check all feature columns exist
    missing_cols = [col for col in feature_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing feature columns: {missing_cols}")

    # Initialize predictions as NaN (preserves alignment)
    y_hat = pd.Series(np.nan, index=df.index, name="y_hat")

    # Select features
    X = df[feature_cols].copy()

    # Find rows without NaNs
    valid_mask = ~X.isna().any(axis=1)
    X_valid = X[valid_mask]

    if len(X_valid) > 0:
        # Scale and predict
        X_scaled = scaler.transform(X_valid.values)
        y_hat_valid = model.predict(X_scaled)

        # Assign predictions to valid rows
        y_hat.loc[valid_mask] = y_hat_valid

    return y_hat

