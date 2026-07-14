"""
Feature engineering utilities shared by both pipelines.
"""

import numpy as np
import pandas as pd


# ─────────────────────────────────────────────────────────────
# DATE FEATURES
# ─────────────────────────────────────────────────────────────

def add_date_features(df: pd.DataFrame, date_col: str = "Date") -> pd.DataFrame:
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df["Year"]       = df[date_col].dt.year
    df["Month"]      = df[date_col].dt.month
    df["Day"]        = df[date_col].dt.day
    df["WeekOfYear"] = df[date_col].dt.isocalendar().week.astype(int)
    df["IsWeekend"]  = (df[date_col].dt.dayofweek >= 5).astype(int)
    return df


# ─────────────────────────────────────────────────────────────
# CYCLICAL ENCODING
# ─────────────────────────────────────────────────────────────

def add_cyclical_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Month_sin"]     = np.sin(2 * np.pi * df["Month"] / 12)
    df["Month_cos"]     = np.cos(2 * np.pi * df["Month"] / 12)
    df["DayOfWeek_sin"] = np.sin(2 * np.pi * df["DayOfWeek"] / 7)
    df["DayOfWeek_cos"] = np.cos(2 * np.pi * df["DayOfWeek"] / 7)
    return df


# ─────────────────────────────────────────────────────────────
# LAG + ROLLING FEATURES  (must be sorted by Date before calling)
# ─────────────────────────────────────────────────────────────

def add_lag_features(df: pd.DataFrame, sales_col: str = "Sales") -> pd.DataFrame:
    df = df.copy().sort_values("Date").reset_index(drop=True)
    for lag in [1, 7, 14, 21, 28]:
        df[f"Sales_lag_{lag}"] = df[sales_col].shift(lag)
    df["Rolling_mean_7"]  = df[sales_col].shift(1).rolling(7).mean()
    df["Rolling_mean_14"] = df[sales_col].shift(1).rolling(14).mean()
    df["Rolling_mean_30"] = df[sales_col].shift(1).rolling(30).mean()
    df["Rolling_std_7"]   = df[sales_col].shift(1).rolling(7).std()
    df["Trend"]           = np.arange(len(df))
    return df


# ─────────────────────────────────────────────────────────────
# STORE-SPECIFIC FEATURES
# ─────────────────────────────────────────────────────────────

def add_competition_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"])
    has = df["CompetitionOpenSinceYear"].notna() & df["CompetitionOpenSinceMonth"].notna()
    comp_dates = pd.Series(pd.NaT, index=df.index)
    if has.any():
        comp_dates[has] = pd.to_datetime(dict(
            year  = df.loc[has, "CompetitionOpenSinceYear"].astype(int),
            month = df.loc[has, "CompetitionOpenSinceMonth"].astype(int),
            day   = 1,
        ))
    df["CompetitionOpenDays"] = (
        (df["Date"] - comp_dates).dt.days.fillna(0).clip(lower=0)
    )
    df["NoCompetitionInfo"] = df["CompetitionOpenSinceYear"].isna().astype(int)
    return df


def add_promo2_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"])
    has = df["Promo2SinceYear"].notna() & df["Promo2SinceWeek"].notna() & (df["Promo2"] == 1)
    start_dates = pd.Series(pd.NaT, index=df.index)
    if has.any():
        start_dates[has] = pd.to_datetime(
            df.loc[has, "Promo2SinceYear"].astype(int).astype(str)
            + "-W"
            + df.loc[has, "Promo2SinceWeek"].astype(int).astype(str).str.zfill(2)
            + "-1",
            format="%Y-W%W-%w", errors="coerce",
        )
    df["Promo2RunningDays"] = (
        (df["Date"] - start_dates).dt.days.fillna(0).clip(lower=0)
    )
    df["NoPromo2"] = df["Promo2SinceYear"].isna().astype(int)
    MONTH_MAP = {
        "Jan":1,"Feb":2,"Mar":3,"Apr":4,"May":5,"Jun":6,
        "Jul":7,"Aug":8,"Sept":9,"Oct":10,"Nov":11,"Dec":12,
    }
    def _is_promo(row):
        if row.get("Promo2", 0) != 1 or pd.isna(row.get("PromoInterval")):
            return 0
        months = [MONTH_MAP.get(m.strip(), 0) for m in str(row["PromoInterval"]).split(",")]
        return int(row["Date"].month in months)
    df["IsPromoMonth"] = df.apply(_is_promo, axis=1)
    return df


def encode_store_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # Encode StateHoliday: '0' → 0, 'a'/'b'/'c' → 1/2/3
    df["StateHoliday"] = df["StateHoliday"].astype(str).map(
        {"0": 0, "a": 1, "b": 2, "c": 3}
    ).fillna(0).astype(int)
    df["StoreType"]  = df["StoreType"].astype(str).str.lower().map(
        {"a": 0, "b": 1, "c": 2, "d": 3}
    ).fillna(0).astype(int)
    df["Assortment"] = df["Assortment"].astype(str).str.lower().map(
        {"a": 0, "b": 1, "c": 2}
    ).fillna(0).astype(int)
    return df
