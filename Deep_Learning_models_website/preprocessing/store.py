"""
Store-wise sales preprocessing pipeline.
Produces exactly 30 features for the store-specific models.

FEATURE ORDER (must match your store model training notebook):
    DayOfWeek, Promo, SchoolHoliday,
    StoreType, Assortment, CompetitionDistance,
    Promo2, CompetitionOpenDays, Promo2RunningDays, IsPromoMonth,
    Year, Month, Day, WeekOfYear, IsWeekend,
    Month_sin, Month_cos, DayOfWeek_sin, DayOfWeek_cos,
    Sales_lag_1, Sales_lag_7, Sales_lag_14, Sales_lag_21, Sales_lag_28,
    Rolling_mean_7, Rolling_mean_14, Rolling_mean_30, Rolling_std_7,
    Trend
    = 29 columns

These models live in saved_models/store/ and use their own
feature_scaler.pkl (30-feature) and target_scaler.pkl.
They are COMPLETELY SEPARATE from the aggregate models.
"""

import os
import numpy as np
import pandas as pd
from datetime import datetime

from utils.feature_engineering import (
    add_date_features,
    add_cyclical_features,
    add_lag_features,
    add_competition_features,
    add_promo2_features,
    encode_store_categoricals,
)
from utils.scaler import scale_features
from utils.sequence import build_sequence, SEQUENCE_LENGTH

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")

# ── 32 features — must match your store training notebook exactly ──
STORE_FEATURE_COLS = [
    "DayOfWeek", "Promo", "SchoolHoliday", "StateHoliday",
    "StoreType", "Assortment", "CompetitionDistance",
    "Promo2", "CompetitionOpenDays", "Promo2RunningDays", "IsPromoMonth",
    "NoCompetitionInfo", "NoPromo2",
    "Year", "Month", "Day", "WeekOfYear", "IsWeekend",
    "Month_sin", "Month_cos", "DayOfWeek_sin", "DayOfWeek_cos",
    "Sales_lag_1", "Sales_lag_7", "Sales_lag_14", "Sales_lag_21", "Sales_lag_28",
    "Rolling_mean_7", "Rolling_mean_14", "Rolling_mean_30", "Rolling_std_7",
    "Trend",
]
assert len(STORE_FEATURE_COLS) == 32, "Store feature list must be exactly 32 columns"


# ─────────────────────────────────────────────────────────────
# DATA HELPERS
# ─────────────────────────────────────────────────────────────

def _load_store_metadata() -> pd.DataFrame:
    path = os.path.join(DATA_DIR, "store.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"store.csv not found at {path}\n"
            "Place the Rossmann store.csv inside the data/ folder."
        )
    return pd.read_csv(path)


def _load_store_history(store_id: int) -> pd.DataFrame:
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
    df = df[(df["Store"] == store_id) & (df["Open"] == 1)].copy()
    if df.empty:
        raise ValueError(f"No open-day records found for Store {store_id} in train.csv")
    return df.sort_values("Date").reset_index(drop=True)


def get_all_store_ids() -> list:
    """Sorted list of store IDs from store.csv."""
    return sorted(_load_store_metadata()["Store"].unique().tolist())


# ─────────────────────────────────────────────────────────────
# MAIN PIPELINE
# ─────────────────────────────────────────────────────────────

def prepare_store_features(
    store_id:        int,
    prediction_date: datetime,
    promo:           int,
    school_holiday:  int,
    feature_scaler,
    model_name:      str,
    seq_len:         int = SEQUENCE_LENGTH,
) -> np.ndarray:
    """
    Full store-wise feature pipeline for one store.

    Steps
    ─────
    1. Load store metadata (StoreType, Assortment, Competition, Promo2 …)
    2. Load store's own Sales history from train.csv
    3. Append a synthetic next-day row for the prediction date
    4. Merge store metadata columns into every row
    5. Apply date features, cyclical encoding, store-specific features
    6. Compute lag / rolling features from THIS STORE's Sales only
    7. Scale with the store feature_scaler (30 features)
    8. Build and return the sequence tensor

    Returns
    -------
    np.ndarray shaped for model.predict():
        (1, 1800)     for FNN  (1800 = 60 × 30)
        (1, 60, 30)   for GRU / RNN / N-HiTS
    """

    # ── 1. Metadata ──────────────────────────────────────────
    meta_df   = _load_store_metadata()
    store_row = meta_df[meta_df["Store"] == store_id]
    if store_row.empty:
        raise ValueError(f"Store {store_id} not found in store.csv")
    store_meta = store_row.iloc[0].to_dict()

    # ── 2. Store history ─────────────────────────────────────
    df_hist = _load_store_history(store_id)

    # ── 3. Synthetic next-day row ────────────────────────────
    pred_ts           = pd.Timestamp(prediction_date)
    placeholder_sales = df_hist["Sales"].iloc[-7:].mean()

    synthetic = pd.DataFrame([{
        "Date":          pred_ts,
        "Sales":         placeholder_sales,
        "DayOfWeek":     pred_ts.dayofweek + 1,   # Rossmann uses 1-7
        "Promo":         promo,
        "SchoolHoliday": school_holiday,
        "Store":         store_id,
        "Open":          1,
        "StateHoliday":  "0",
        "Customers":     0,
    }])
    df_full = pd.concat([df_hist, synthetic], ignore_index=True)

    # ── 4. Merge store metadata ───────────────────────────────
    for col, val in store_meta.items():
        if col != "Store":
            df_full[col] = val

    # ── 5. Feature engineering ───────────────────────────────
    df_full = add_date_features(df_full)
    df_full = add_cyclical_features(df_full)
    df_full = encode_store_categoricals(df_full)
    df_full = add_competition_features(df_full)
    df_full = add_promo2_features(df_full)

    # ── 6. Lag features (per-store Sales only) ───────────────
    df_full = add_lag_features(df_full, sales_col="Sales")

    # Fill early NaNs (lag warm-up rows at start of series)
    df_full = df_full.ffill().fillna(0)

    # ── 7. Validate columns ──────────────────────────────────
    missing = [c for c in STORE_FEATURE_COLS if c not in df_full.columns]
    if missing:
        raise KeyError(
            f"These feature columns are missing after engineering: {missing}\n"
            "Check that store.csv has the expected columns and STORE_FEATURE_COLS "
            "matches your training notebook."
        )

    features_2d     = df_full[STORE_FEATURE_COLS].values.astype(np.float32)
    features_scaled = scale_features(feature_scaler, features_2d)

    # ── 8. Sequence ──────────────────────────────────────────
    return build_sequence(features_scaled, model_name, seq_len)


# ─────────────────────────────────────────────────────────────
# UNIFIED PIPELINE (for Streamlit UI)
# ─────────────────────────────────────────────────────────────

def prepare_store_unified(
    store_id:        int,
    prediction_date: str,  # Format: "YYYY-MM-DD"
    feature_scaler,
    model_name:      str,
    seq_len:         int = SEQUENCE_LENGTH,
) -> np.ndarray:
    """
    Simplified store-wise pipeline for UI predictions.
    
    Takes only:
    - store_id: Which store to predict for
    - prediction_date: Target prediction date (e.g., "2015-07-31")
    - feature_scaler: Fitted MinMaxScaler for 30 store features
    - model_name: "GRU", "N-HiTS", "FNN" (determines output shape)
    
    Returns sequence ONLY (no future features):
    - (1, 1800) for FNN (flattened 60×30)
    - (1, 60, 30) for GRU/RNN/N-HiTS
    
    Uses ONLY past 60 days before prediction_date (NO synthetic future row).
    This matches how store models were trained in Colab.
    """
    
    # ── 1. Metadata ──────────────────────────────────────────
    meta_df   = _load_store_metadata()
    store_row = meta_df[meta_df["Store"] == store_id]
    if store_row.empty:
        raise ValueError(f"Store {store_id} not found in store.csv")
    store_meta = store_row.iloc[0].to_dict()

    # ── 2. Store history ─────────────────────────────────────
    df_hist = _load_store_history(store_id)

    # ── 3. Filter to <= prediction_date (NO synthetic future row) ──
    pred_ts = pd.Timestamp(prediction_date)
    df_full = df_hist[df_hist["Date"] <= pred_ts].copy()
    
    if df_full.empty:
        raise ValueError(
            f"No historical data found for Store {store_id} "
            f"on or before {prediction_date}"
        )

    # ── 4. Merge store metadata ───────────────────────────────
    for col, val in store_meta.items():
        if col != "Store":
            df_full[col] = val

    # ── 5. Feature engineering ───────────────────────────────
    df_full = add_date_features(df_full)
    df_full = add_cyclical_features(df_full)
    df_full = encode_store_categoricals(df_full)
    df_full = add_competition_features(df_full)
    df_full = add_promo2_features(df_full)

    # ── 6. Lag features (per-store Sales only) ───────────────
    df_full = add_lag_features(df_full, sales_col="Sales")

    # Fill early NaNs (lag warm-up rows at start of series)
    df_full = df_full.ffill().fillna(0)

    # ── 7. Validate columns ──────────────────────────────────
    missing = [c for c in STORE_FEATURE_COLS if c not in df_full.columns]
    if missing:
        raise KeyError(
            f"These feature columns are missing after engineering: {missing}\n"
            "Check that store.csv has the expected columns and STORE_FEATURE_COLS "
            "matches your training notebook."
        )

    features_2d     = df_full[STORE_FEATURE_COLS].values.astype(np.float32)
    features_scaled = scale_features(feature_scaler, features_2d)

    # ── 8. Sequence (uses past 60 days only) ─────────────────
    return build_sequence(features_scaled, model_name, seq_len)
