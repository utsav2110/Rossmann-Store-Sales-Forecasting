"""
Aggregate (global) sales preprocessing pipeline.
Produces exactly 22 features matching the trained aggregate models.

FEATURE ORDER (must match training):
    DayOfWeek, Promo, SchoolHoliday,
    Year, Month, Day, WeekOfYear, IsWeekend,
    Month_sin, Month_cos, DayOfWeek_sin, DayOfWeek_cos,
    Sales_lag_1, Sales_lag_7, Sales_lag_14, Sales_lag_21, Sales_lag_28,
    Rolling_mean_7, Rolling_mean_14, Rolling_mean_30, Rolling_std_7,
    Trend
    = 22 columns

UNIFIED input mode:
  — Pick a prediction date (≤ 2015-07-31) + 3 features for that date
  — Last 60 days before prediction_date are auto-fetched from train.csv
  — Returns both sequence and future features for dual-input models
"""

import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from utils.feature_engineering import (
    add_date_features,
    add_cyclical_features,
    add_lag_features,
)
from utils.scaler import scale_features
from utils.sequence import build_sequence, SEQUENCE_LENGTH

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")

# ── 22 features — must match your training notebook exactly ──
AGGREGATE_FEATURE_COLS = [
    "DayOfWeek", "Promo", "SchoolHoliday",
    "Year", "Month", "Day", "WeekOfYear", "IsWeekend",
    "Month_sin", "Month_cos", "DayOfWeek_sin", "DayOfWeek_cos",
    "Sales_lag_1", "Sales_lag_7", "Sales_lag_14", "Sales_lag_21", "Sales_lag_28",
    "Rolling_mean_7", "Rolling_mean_14", "Rolling_mean_30", "Rolling_std_7",
    "Trend",
]
assert len(AGGREGATE_FEATURE_COLS) == 22, "Feature list must be exactly 22 columns"


# ─────────────────────────────────────────────────────────────
# DATA HELPERS
# ─────────────────────────────────────────────────────────────

def _load_train() -> pd.DataFrame:
    path = os.path.join(DATA_DIR, "train.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"train.csv not found at {path}\n"
            "Place the Rossmann train.csv inside the data/ folder."
        )
    df = pd.read_csv(
        path,
        parse_dates=["Date"],
        dtype={"StateHoliday": str},
        low_memory=False,
    )
    return df[df["Open"] == 1].sort_values("Date").reset_index(drop=True)


def _aggregate_daily(df: pd.DataFrame) -> pd.DataFrame:
    """Sum all stores per day → global daily series."""
    agg = (
        df.groupby("Date")
        .agg(
            Sales        = ("Sales",        "sum"),
            DayOfWeek    = ("DayOfWeek",    "first"),
            Promo        = ("Promo",        "max"),
            SchoolHoliday= ("SchoolHoliday","max"),
        )
        .reset_index()
    )
    return agg.sort_values("Date").reset_index(drop=True)


def _engineer(df: pd.DataFrame) -> pd.DataFrame:
    """Apply all feature engineering steps to an aggregated daily DataFrame."""
    df = add_date_features(df)
    df = add_cyclical_features(df)
    df = add_lag_features(df)
    return df


# ─────────────────────────────────────────────────────────────
# MODE 1 — AUTO (recommended)
# ─────────────────────────────────────────────────────────────

def prepare_aggregate_auto(
    prediction_date: datetime,
    feature_scaler,
    model_name: str,
    seq_len: int = SEQUENCE_LENGTH,
) -> np.ndarray:
    """
    Load the last `seq_len` days before `prediction_date` from train.csv,
    engineer features, scale, build sequence.

    Returns shaped tensor ready for model.predict().
    """
    df_raw = _load_train()
    df_agg = _aggregate_daily(df_raw)

    cutoff  = pd.Timestamp(prediction_date)
    df_hist = df_agg[df_agg["Date"] < cutoff].copy()

    if len(df_hist) < seq_len + 30:   # need extra rows for lag NaN warm-up
        raise ValueError(
            f"Only {len(df_hist)} days of data before {prediction_date.date()}. "
            f"Need at least {seq_len + 30}."
        )

    df_hist = _engineer(df_hist)

    # Drop NaN rows caused by lag computation at the start of series
    df_hist = df_hist.dropna(subset=AGGREGATE_FEATURE_COLS).reset_index(drop=True)

    features_2d     = df_hist[AGGREGATE_FEATURE_COLS].values.astype(np.float32)
    features_scaled = scale_features(feature_scaler, features_2d)

    return build_sequence(features_scaled, model_name, seq_len)


# ─────────────────────────────────────────────────────────────
# MODE 2 — MANUAL
# ─────────────────────────────────────────────────────────────

def prepare_aggregate_manual(
    day_of_week:    int,
    promo:          int,
    school_holiday: int,
    feature_scaler,
    model_name: str,
    seq_len: int = SEQUENCE_LENGTH,
) -> np.ndarray:
    """
    Append a synthetic 'next-day' row to the end of train.csv history,
    engineer features, scale, build sequence.

    The synthetic row's Sales is filled with the 7-day rolling mean as a
    placeholder (it won't affect the output — only the lag features from
    the 60 rows before it matter).
    """
    df_raw = _load_train()
    df_agg = _aggregate_daily(df_raw)

    last_date = df_agg["Date"].max()
    next_date = last_date + timedelta(days=1)

    synthetic = pd.DataFrame([{
        "Date":          next_date,
        "Sales":         df_agg["Sales"].iloc[-7:].mean(),
        "DayOfWeek":     day_of_week,
        "Promo":         promo,
        "SchoolHoliday": school_holiday,
    }])
    df_full = pd.concat([df_agg, synthetic], ignore_index=True)
    df_full  = _engineer(df_full)
    df_full  = df_full.ffill().fillna(0)

    features_2d     = df_full[AGGREGATE_FEATURE_COLS].values.astype(np.float32)
    features_scaled = scale_features(feature_scaler, features_2d)

    return build_sequence(features_scaled, model_name, seq_len)


# ─────────────────────────────────────────────────────────────
# UNIFIED MODE — single date input + 3 features for that date
# Returns both sequence and future features for dual-input models
# ─────────────────────────────────────────────────────────────

def prepare_aggregate_unified(
    prediction_date: datetime,
    day_of_week:     int,
    promo:           int,
    school_holiday:  int,
    feature_scaler,
    model_name: str,
    seq_len: int = SEQUENCE_LENGTH,
):
    """
    Unified prediction mode: takes date + 3 features.
    
    Loads the last `seq_len` days BEFORE `prediction_date` from train.csv,
    builds the historical sequence, and returns both the sequence and the
    future features (3 values) for the prediction_date.
    
    Returns
    -------
    tuple: (sequence_input, future_features)
        - sequence_input: np.ndarray shaped for model (1, seq_len, 22) or (1, seq_len*22)
        - future_features: np.ndarray shaped (1, 3) with [DayOfWeek, Promo, SchoolHoliday]
    """
    df_raw = _load_train()
    df_agg = _aggregate_daily(df_raw)

    cutoff  = pd.Timestamp(prediction_date)
    df_hist = df_agg[df_agg["Date"] < cutoff].copy()

    if len(df_hist) < seq_len + 30:   # need extra rows for lag NaN warm-up
        raise ValueError(
            f"Only {len(df_hist)} days of data before {prediction_date.date()}. "
            f"Need at least {seq_len + 30}."
        )

    df_hist = _engineer(df_hist)

    # Drop NaN rows caused by lag computation at the start of series
    df_hist = df_hist.dropna(subset=AGGREGATE_FEATURE_COLS).reset_index(drop=True)

    features_2d     = df_hist[AGGREGATE_FEATURE_COLS].values.astype(np.float32)
    features_scaled = scale_features(feature_scaler, features_2d)

    # Build the sequence for the historical data
    sequence_input = build_sequence(features_scaled, model_name, seq_len)

    # Build future features and SCALE them to match training data
    # Create a synthetic full-feature row with user inputs + zeros for engineered features
    # This ensures the 3 features are scaled using the same scaler as training
    synthetic_full = np.zeros((1, 22), dtype=np.float32)
    synthetic_full[0, 0] = day_of_week       # DayOfWeek (feature 0)
    synthetic_full[0, 1] = promo             # Promo (feature 1)
    synthetic_full[0, 2] = school_holiday    # SchoolHoliday (feature 2)
    # Other features stay 0 (Year, Month, Day, cyclical, lag, rolling stats)

    # Scale the synthetic row using the same scaler as training
    synthetic_scaled = scale_features(feature_scaler, synthetic_full)
    
    # Extract just the first 3 scaled features (DayOfWeek, Promo, SchoolHoliday)
    future_features = synthetic_scaled[:, :3]

    return sequence_input, future_features
