"""
utils.py — Data loading, preprocessing, feature engineering, and forecasting helpers.
Strictly follows logic from aggregated_store.ipynb and particular_store.ipynb.
All @st.cache_data decorators live in app.py to keep utils import-safe.
"""

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller

# ─────────────────────────────────────────────────────────────────────────────
# URLs
# ─────────────────────────────────────────────────────────────────────────────
TRAIN_URL = (
    "https://raw.githubusercontent.com/sarthaksoni25/"
    "Rossmann-Store-Sales-Prediction/refs/heads/master/dataset/train.csv"
)
STORE_URL = (
    "https://raw.githubusercontent.com/sarthaksoni25/"
    "Rossmann-Store-Sales-Prediction/master/dataset/store.csv"
)


# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADERS
# ─────────────────────────────────────────────────────────────────────────────

def load_raw_data() -> pd.DataFrame:
    return pd.read_csv(TRAIN_URL, low_memory=False)


def load_store_data() -> pd.DataFrame:
    df_train = pd.read_csv(TRAIN_URL, low_memory=False)
    df_store = pd.read_csv(STORE_URL, low_memory=False)
    return df_train.merge(df_store, on="Store", how="left")


# ─────────────────────────────────────────────────────────────────────────────
# AGGREGATED PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

def build_aggregated_df(raw_df: pd.DataFrame) -> pd.DataFrame:
    df = raw_df.copy()
    df.drop(columns=["Store", "Id"], inplace=True, errors="ignore")
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date")
    df_open = df[df["Open"] == 1].copy()
    df_open.drop(columns=["Open"], inplace=True, errors="ignore")
    df_open.drop(columns=["StateHoliday", "DayOfWeek"], inplace=True, errors="ignore")

    df_agg = df_open.groupby("Date").agg(
        Sales=("Sales", "sum"),
        Customers=("Customers", "sum"),
        Promo=("Promo", "mean"),
        SchoolHoliday=("SchoolHoliday", "mean"),
    ).reset_index()

    df_agg["Promo"] = (df_agg["Promo"] > 0.5).astype(int)
    df_agg["SchoolHoliday"] = (df_agg["SchoolHoliday"] > 0.5).astype(int)
    df_agg["DayOfWeek"] = df_agg["Date"].dt.dayofweek
    df_agg["Month"] = df_agg["Date"].dt.month
    df_agg["Year"] = df_agg["Date"].dt.year
    df_agg["IsWeekend"] = df_agg["DayOfWeek"].isin([5, 6]).astype(int)

    df_agg = df_agg.sort_values("Date")
    for lag in [1, 7, 14, 28]:
        df_agg[f"lag_{lag}"] = df_agg["Sales"].shift(lag)

    df_agg["rolling_mean_7"] = df_agg["Sales"].rolling(7).mean()
    df_agg["rolling_std_7"] = df_agg["Sales"].rolling(7).std()
    df_agg["dow_sin"] = np.sin(2 * np.pi * df_agg["DayOfWeek"] / 7)
    df_agg["dow_cos"] = np.cos(2 * np.pi * df_agg["DayOfWeek"] / 7)

    df_agg.dropna(inplace=True)
    df_agg = df_agg.replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)
    df_agg = df_agg.sort_values("Date")
    df_agg.set_index("Date", inplace=True)
    df_agg = df_agg.asfreq("D")
    return df_agg


_EXOG_COLS_AGG = [
    "Promo", "SchoolHoliday", "IsWeekend",
    "lag_1", "lag_7", "lag_14", "lag_28",
    "rolling_mean_7", "rolling_std_7",
    "dow_sin", "dow_cos",
]


def train_test_split_agg(df_agg: pd.DataFrame, ratio: float = 0.8):
    train_size = int(len(df_agg) * ratio)
    train = df_agg.iloc[:train_size]
    test = df_agg.iloc[train_size:]
    return (
        train, test,
        train["Sales"], test["Sales"],
        train[_EXOG_COLS_AGG], test[_EXOG_COLS_AGG],
    )


# ─────────────────────────────────────────────────────────────────────────────
# METRICS
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_model(y_true, y_pred, k: int):
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)
    rmse = float(np.sqrt(mean_squared_error(yt, yp)))
    mae = float(mean_absolute_error(yt, yp))
    mape = float(np.mean(np.abs((yt - yp) / (yt + 1e-8))) * 100)
    r2 = float(r2_score(yt, yp))
    n = len(yt)
    adj_r2 = float(1 - (1 - r2) * (n - 1) / (n - k - 1)) if n > k + 1 else float("nan")
    return rmse, mae, mape, r2, adj_r2


def _coverage(y_true, lower, upper):
    return float(((np.asarray(y_true) >= np.asarray(lower)) &
                  (np.asarray(y_true) <= np.asarray(upper))).mean())


def _crps(y_true, mean, std):
    z = (y_true - mean) / (std + 1e-8)
    vals = std * (z * (2 * norm.cdf(z) - 1) + 2 * norm.pdf(z) - 1 / np.sqrt(np.pi))
    return float(np.mean(vals))


def metrics_dict(name, y_true, y_pred, k, conf_int=None):
    rmse, mae, mape, r2, adj_r2 = evaluate_model(y_true, y_pred, k)
    d = {
        "Model": name,
        "RMSE": round(rmse, 2), "MAE": round(mae, 2),
        "MAPE (%)": round(mape, 2), "R²": round(r2, 4), "Adj-R²": round(adj_r2, 4),
        "Coverage (95%)": None, "CRPS": None,
    }
    if conf_int is not None:
        lo = np.asarray(conf_int.iloc[:, 0], dtype=float)
        hi = np.asarray(conf_int.iloc[:, 1], dtype=float)
        std = (hi - lo) / (2 * 1.96)
        d["Coverage (95%)"] = round(_coverage(np.asarray(y_true), lo, hi), 4)
        d["CRPS"] = round(_crps(np.asarray(y_true), np.asarray(y_pred), std), 2)
    return d


# ─────────────────────────────────────────────────────────────────────────────
# AGGREGATED MODEL FITTING
# ─────────────────────────────────────────────────────────────────────────────

def fit_sarimax_agg(y_train, exog_train):
    model = SARIMAX(y_train, exog=exog_train, order=(1, 1, 1), seasonal_order=(1, 1, 1, 7))
    return model.fit(disp=False)


def fit_prophet_agg(df_agg: pd.DataFrame, train_size: int):
    from prophet import Prophet
    df_prophet = df_agg.reset_index()[["Date", "Sales", "Promo", "SchoolHoliday"]].copy()
    df_prophet.columns = ["ds", "y", "Promo", "SchoolHoliday"]
    train_p = df_prophet.iloc[:train_size].copy()
    test_p = df_prophet.iloc[train_size:].copy()

    model_p = Prophet(
        yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=False,
        changepoint_prior_scale=0.1, seasonality_mode="multiplicative",
    )
    model_p.add_regressor("Promo")
    model_p.add_regressor("SchoolHoliday")
    model_p.fit(train_p)

    future = model_p.make_future_dataframe(periods=len(test_p))
    future = future.merge(df_prophet[["ds", "Promo", "SchoolHoliday"]], on="ds", how="left")
    future["Promo"] = future["Promo"].fillna(0)
    future["SchoolHoliday"] = future["SchoolHoliday"].fillna(0)
    forecast = model_p.predict(future)
    return model_p, forecast, train_p, test_p, df_prophet


def fit_tbats_agg(y_train):
    from tbats import TBATS
    # estimator = TBATS(
    #     seasonal_periods=[7], use_box_cox=False,
    #     use_trend=True, use_damped_trend=True, show_warnings=False,
    # )
    # return estimator.fit(y_train)
    y_train = y_train.astype(float)
    y_train = y_train.dropna().astype(float)
    estimator = TBATS(
        seasonal_periods=None,   # ✅ float instead of int
        use_box_cox=False,
        use_trend=True,
        use_damped_trend=True,
        show_warnings=False,
    )

    return estimator.fit(y_train)


# ─────────────────────────────────────────────────────────────────────────────
# AGGREGATED FUTURE FORECASTING
# ─────────────────────────────────────────────────────────────────────────────

def _build_future_exog_agg(df_agg, steps, user_overrides=None):
    sales_hist = df_agg["Sales"].tolist()
    last_date = df_agg.index[-1]
    future_rows, future_dates = [], []

    for i in range(steps):
        next_date = last_date + pd.Timedelta(days=i + 1)
        dow = next_date.dayofweek
        row = {
            "IsWeekend": int(dow >= 5),
            "dow_sin": float(np.sin(2 * np.pi * dow / 7)),
            "dow_cos": float(np.cos(2 * np.pi * dow / 7)),
            "Promo": 0, "SchoolHoliday": 0,
        }
        if user_overrides:
            for key in ("Promo", "SchoolHoliday"):
                if key in user_overrides:
                    v = user_overrides[key]
                    row[key] = int(v[i]) if hasattr(v, "__getitem__") else int(v)

        n = len(sales_hist)
        row["lag_1"] = sales_hist[-1] if n >= 1 else 0.0
        row["lag_7"] = sales_hist[-7] if n >= 7 else sales_hist[0]
        row["lag_14"] = sales_hist[-14] if n >= 14 else sales_hist[0]
        row["lag_28"] = sales_hist[-28] if n >= 28 else sales_hist[0]
        row["rolling_mean_7"] = float(np.mean(sales_hist[-7:]))
        row["rolling_std_7"] = float(np.std(sales_hist[-7:])) if n >= 2 else 0.0

        future_rows.append(row)
        future_dates.append(next_date)
        sales_hist.append(row["lag_1"])

    return pd.DataFrame(future_rows, index=pd.DatetimeIndex(future_dates))[_EXOG_COLS_AGG]


def sarimax_future_forecast(fitted_model, df_agg, steps=7, user_overrides=None):
    future_exog = _build_future_exog_agg(df_agg, steps=steps, user_overrides=user_overrides)
    fc = fitted_model.get_forecast(steps=steps, exog=future_exog)
    return fc.predicted_mean, fc.conf_int()


def prophet_future_forecast(model_p, df_prophet, last_date, steps=7):
    future = model_p.make_future_dataframe(periods=len(df_prophet) + steps)
    future = future.merge(df_prophet[["ds", "Promo", "SchoolHoliday"]], on="ds", how="left")
    future["Promo"] = future["Promo"].fillna(0)
    future["SchoolHoliday"] = future["SchoolHoliday"].fillna(0)
    forecast = model_p.predict(future)
    return forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].tail(steps).reset_index(drop=True)


def tbats_future_forecast(tbats_model, last_date, steps=7):
    preds = tbats_model.forecast(steps=steps)
    dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=steps, freq="D")
    return pd.Series(preds, index=dates)


# ─────────────────────────────────────────────────────────────────────────────
# STORE-LEVEL PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

def _fe_store(df):
    df = df.copy()
    df["DayOfWeek"] = df["Date"].dt.dayofweek
    df["IsWeekend"] = (df["DayOfWeek"] >= 5).astype(int)
    df["StateHoliday"] = df["StateHoliday"].astype(str).map({"0":0,"a":1,"b":1,"c":1}).fillna(0)
    df["SchoolHoliday"] = df["SchoolHoliday"].fillna(0)
    if "StoreType" in df.columns:
        df["StoreType"] = df["StoreType"].astype("category").cat.codes
    if "Assortment" in df.columns:
        df["Assortment"] = df["Assortment"].astype("category").cat.codes
    if "CompetitionDistance" in df.columns:
        df["CompetitionDistance"] = df["CompetitionDistance"].fillna(df["CompetitionDistance"].median())
    if "Promo2" in df.columns:
        df["Promo2"] = df["Promo2"].fillna(0)
    return df


def prepare_store_df(raw_merged, store_id):
    df = raw_merged.copy()
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values(["Store", "Date"])
    df = df[df["Open"] == 1].copy()
    df = _fe_store(df)

    store_df = df[df["Store"] == store_id].copy()
    store_df.set_index("Date", inplace=True)
    full_index = pd.date_range(store_df.index.min(), store_df.index.max(), freq="D")
    store_df = store_df.reindex(full_index)
    store_df["Sales"] = store_df["Sales"].fillna(0)

    for col in ["Promo","SchoolHoliday","StateHoliday","StoreType","Assortment","CompetitionDistance","Promo2"]:
        if col in store_df.columns:
            store_df[col] = store_df[col].ffill()

    store_df["DayOfWeek"] = store_df.index.dayofweek
    store_df["IsWeekend"] = (store_df["DayOfWeek"] >= 5).astype(int)
    store_df["lag_1"] = store_df["Sales"].shift(1)
    store_df["lag_7"] = store_df["Sales"].shift(7)
    store_df["lag_14"] = store_df["Sales"].shift(14)
    store_df = store_df.fillna(0)
    return store_df


_EXOG_COLS_STORE = ["Promo", "SchoolHoliday", "IsWeekend", "lag_1", "lag_7", "lag_14"]


def train_test_split_store(store_df, ratio=0.8):
    split = int(len(store_df) * ratio)
    train = store_df.iloc[:split]
    test = store_df.iloc[split:]
    return (
        train, test,
        train["Sales"], test["Sales"],
        train[_EXOG_COLS_STORE], test[_EXOG_COLS_STORE],
    )


def check_stationarity(series, alpha=0.05):
    result = adfuller(series.dropna())
    return result[1] < alpha, result[0], result[1]


def make_stationary_d(series, max_diff=2):
    d, temp = 0, series.copy()
    while d <= max_diff:
        is_stat, _, _ = check_stationarity(temp)
        if is_stat:
            return d
        temp = temp.diff().dropna()
        d += 1
    return d


def fit_sarimax_store(y_train, X_train):
    model = SARIMAX(y_train, exog=X_train, order=(5, 0, 3), seasonal_order=(1, 1, 1, 7))
    return model.fit(disp=False)


def _build_future_exog_store(store_df, steps, user_promo, user_school_holiday):
    sales_hist = store_df["Sales"].tolist()
    future_dates = pd.date_range(start=store_df.index[-1] + pd.Timedelta(days=1), periods=steps, freq="D")
    rows = []
    for i in range(steps):
        dow = future_dates[i].dayofweek
        p = int(user_promo[i]) if hasattr(user_promo, "__getitem__") else int(user_promo)
        sh = int(user_school_holiday[i]) if hasattr(user_school_holiday, "__getitem__") else int(user_school_holiday)
        n = len(sales_hist)
        l1 = sales_hist[-1] if n >= 1 else 0.0
        rows.append({
            "Promo": p, "SchoolHoliday": sh, "IsWeekend": int(dow >= 5),
            "lag_1": l1,
            "lag_7": sales_hist[-7] if n >= 7 else l1,
            "lag_14": sales_hist[-14] if n >= 14 else l1,
        })
        sales_hist.append(l1)
    return pd.DataFrame(rows, index=future_dates)[_EXOG_COLS_STORE]


def sarimax_store_forecast(fitted_model, store_df, steps=7, user_promo=0, user_school_holiday=0):
    future_exog = _build_future_exog_store(store_df, steps, user_promo, user_school_holiday)
    fc = fitted_model.get_forecast(steps=steps, exog=future_exog)
    return fc.predicted_mean.clip(lower=0)


def get_store_ids(df):
    return sorted(df["Store"].dropna().unique().astype(int).tolist())
