"""
app.py — Rossmann Store Sales Forecasting Dashboard
Three tabs: Aggregated EDA | Aggregated Models | Store-Level Forecasting
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from utils import (
    # Data
    load_raw_data, load_store_data,
    build_aggregated_df, train_test_split_agg,
    prepare_store_df, train_test_split_store, get_store_ids,
    # Models – aggregated
    fit_sarimax_agg, fit_prophet_agg, fit_tbats_agg,
    sarimax_future_forecast, prophet_future_forecast, tbats_future_forecast,
    # Models – store
    make_stationary_d, fit_sarimax_store, sarimax_store_forecast,
    # Metrics
    metrics_dict, evaluate_model,
)

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Rossmann Sales Forecasting",
    page_icon="🛒",
    layout="wide",
)

st.markdown("""
<style>
    .main-title { font-size:2.2rem; font-weight:700; }
    .section-header { font-size:1.4rem; font-weight:600; border-bottom:2px solid #3498db; padding-bottom:4px; margin-top:1.5rem; }
    .metric-card { background:#f8f9fa; border-left:4px solid #3498db; padding:12px; border-radius:4px; }
    .pred-table td { font-size:1.05rem; }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">🛒 Rossmann Store Sales Forecasting Dashboard</div>', unsafe_allow_html=True)
# i want to give some space here
st.markdown("<br>", unsafe_allow_html=True)
# ─────────────────────────────────────────────────────────────────────────────
# CACHED DATA LOADERS
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data(show_spinner="Loading raw data…")
def get_raw():
    return load_raw_data()

@st.cache_data(show_spinner="Loading merged store data…")
def get_merged():
    return load_store_data()

@st.cache_data(show_spinner="Building aggregated dataset…")
def get_agg_df(_raw):
    return build_aggregated_df(_raw)

@st.cache_data(show_spinner="Training SARIMAX (aggregated)…")
def get_sarimax_agg(_df_agg):
    train, test, y_train, y_test, exog_train, exog_test = train_test_split_agg(_df_agg)
    fitted = fit_sarimax_agg(y_train, exog_train)
    return fitted, train, test, y_train, y_test, exog_train, exog_test

@st.cache_data(show_spinner="Training Prophet (aggregated)…")
def get_prophet_agg(_df_agg, _train_size):
    return fit_prophet_agg(_df_agg, _train_size)

# @st.cache_data(show_spinner="Training TBATS (aggregated)…")
# def get_tbats_agg(_y_train):
#     return fit_tbats_agg(_y_train)

@st.cache_data(show_spinner=f"Fitting SARIMAX for store…")
def get_store_model(_store_df):
    train, test, y_train, y_test, X_train, X_test = train_test_split_store(_store_df)
    fitted = fit_sarimax_store(y_train, X_train)
    return fitted, train, test, y_train, y_test, X_train, X_test

st.markdown("""
<style>
/* Increase tab font size */
button[data-baseweb="tab"] {
    font-size: 1.1rem;
    font-weight: 600;
    padding: 10px 20px;
}

/* Optional: highlight active tab */
button[data-baseweb="tab"][aria-selected="true"] {
    color: #00BFA6;
    border-bottom: 3px solid #00BFA6;
    font-weight: 700;
    font-size: 1.5rem;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Data Visualization",
    "🏬 All Store (Aggregated) Models",
    "🏪 Store Level Forecasting",
    "💡 Business Insights",
])

# ═════════════════════════════════════════════════════════════════════════════
# TAB 1 — AGGREGATED EDA
# ═════════════════════════════════════════════════════════════════════════════
with tab1:
    # st.markdown('<div class="section-header">📊 Data Visualization</div>', unsafe_allow_html=True)

    raw = get_raw()
    df_agg = get_agg_df(raw)
    df_plot = df_agg.reset_index()   # keep 'Date' as column for Plotly

    # ── Summary statistics ────────────────────────────────────────────────────
    # with st.expander("📋 Dataset Summary", expanded=False):
    #     st.write(f"**Shape:** {df_plot.shape[0]} rows × {df_plot.shape[1]} columns")
    #     st.dataframe(df_plot.describe().T.round(2), width='stretch')

    # ── 1. Total Daily Sales Over Time ───────────────────────────────────────
    st.markdown("#### 1. Total Daily Sales Over Time")
    fig1 = px.line(df_plot, x="Date", y="Sales",
                   labels={"Sales": "Sales", "Date": "Date"},
                   template="plotly_white")
    fig1.update_traces(line_color="#2980b9")
    fig1.update_layout(height=380, hovermode="x unified")
    st.plotly_chart(fig1, width='stretch')

    # ── 2. Sales by Day of Week ───────────────────────────────────────────────
    st.markdown("#### 2. Sales Distribution by Day of Week")
    dow_labels = {0:"Mon", 1:"Tue", 2:"Wed", 3:"Thu", 4:"Fri", 5:"Sat", 6:"Sun"}
    df_plot["DayLabel"] = df_plot["DayOfWeek"].map(dow_labels)
    fig2 = px.box(df_plot, x="DayOfWeek", y="Sales",
                #   title="Sales Distribution by Day of Week",
                  labels={"DayOfWeek": "Day (0=Mon, 6=Sun)", "Sales": "Sales"},
                  template="plotly_white",
                  color="DayOfWeek",
                  color_discrete_sequence=px.colors.qualitative.Set2)
    fig2.update_layout(height=400, showlegend=False)
    st.plotly_chart(fig2, width='stretch')

    col_a, col_b = st.columns(2)

    # ── 3. Average Sales: Promo vs No Promo ──────────────────────────────────
    with col_a:
        st.markdown("#### 3. Average Sales: Promo vs No Promo")
        promo_sales = df_plot.groupby("Promo")["Sales"].mean().reset_index()
        promo_sales["Promo"] = promo_sales["Promo"].map({0: "No Promo", 1: "Promo"})
        fig3 = px.bar(promo_sales, x="Promo", y="Sales",
                      color="Promo",
                      color_discrete_map={"No Promo": "#e74c3c", "Promo": "#27ae60"},
                    #   title="Average Sales: Promo vs No Promo",
                      template="plotly_white")
        fig3.update_layout(height=380, showlegend=False)
        st.plotly_chart(fig3, width='stretch')

    # ── 4. Sales Boxplot: Promo vs No Promo ──────────────────────────────────
    with col_b:
        st.markdown("#### 4. Sales Distribution: Promo vs No Promo")
        df_plot["PromoLabel"] = df_plot["Promo"].map({0: "No Promo", 1: "Promo"})
        fig4 = px.box(df_plot, x="PromoLabel", y="Sales",
                      color="PromoLabel",
                      color_discrete_map={"No Promo": "#e74c3c", "Promo": "#27ae60"},
                    #   title="Sales Distribution: Promo vs No Promo",
                      template="plotly_white")
        fig4.update_layout(height=380, showlegend=False)
        st.plotly_chart(fig4, width='stretch')

    # ── 5. Sales Over Time: Promo vs No Promo ────────────────────────────────
    st.markdown("#### 5. Sales Over Time: Promo vs No Promo")
    fig5 = go.Figure()
    for promo_val, label, color in [(0, "No Promo", "#e74c3c"), (1, "Promo Active", "#27ae60")]:
        subset = df_plot[df_plot["Promo"] == promo_val]
        fig5.add_trace(go.Scatter(x=subset["Date"], y=subset["Sales"],
                                   mode="lines", name=label,
                                   marker=dict(color=color, size=4, opacity=0.6)))
    fig5.update_layout(xaxis_title="Date", yaxis_title="Sales",
                       template="plotly_white", height=380, hovermode="x unified")
    st.plotly_chart(fig5, width='stretch')

    # ── 6. Monthly Sales: Promo vs No Promo ──────────────────────────────────
    st.markdown("#### 6. Monthly Average Sales: Promo vs No Promo")
    monthly_promo = df_plot.groupby(["Month", "Promo"])["Sales"].mean().reset_index()
    monthly_promo["PromoLabel"] = monthly_promo["Promo"].map({0: "No Promo", 1: "Promo"})
    fig6 = px.bar(monthly_promo, x="Month", y="Sales", color="PromoLabel",
                  barmode="group",
                  color_discrete_map={"No Promo": "#e74c3c", "Promo": "#27ae60"},
                #   title="Monthly Average Sales: Promo vs No Promo",
                  template="plotly_white")
    fig6.update_layout(height=400)
    st.plotly_chart(fig6, width='stretch')

    # ── 7. Correlation Heatmap ────────────────────────────────────────────────
    st.markdown("#### 7. Feature Correlation Heatmap")
    corr_cols = ["Sales", "Customers", "Promo", "SchoolHoliday", "DayOfWeek", "Month"]
    corr_matrix = df_plot[corr_cols].corr().round(2)
    fig7 = px.imshow(corr_matrix, text_auto=True, aspect="auto",
                     color_continuous_scale="RdBu_r",
                    #  title="Feature Correlation Heatmap",
                     template="plotly_white")
    fig7.update_layout(height=450)
    st.plotly_chart(fig7, width='stretch')

    col_c, col_d = st.columns(2)

    # ── 8. Weekend vs Weekday Sales ───────────────────────────────────────────
    with col_c:
        st.markdown("#### 8. Weekend vs Weekday Sales")
        df_plot["WeekendLabel"] = df_plot["IsWeekend"].map({0: "Weekday", 1: "Weekend"})
        fig8 = px.box(df_plot, x="WeekendLabel", y="Sales",
                      color="WeekendLabel",
                      color_discrete_map={"Weekday": "#3498db", "Weekend": "#e67e22"},
                    #   title="Weekend vs Weekday Sales",
                      template="plotly_white")
        fig8.update_layout(height=380, showlegend=False)
        st.plotly_chart(fig8, width='stretch')

    # ── 9. Average Monthly Sales ──────────────────────────────────────────────
    with col_d:
        st.markdown("#### 9. Average Monthly Sales")
        monthly_sales = df_plot.groupby("Month")["Sales"].mean().reset_index()
        fig9 = px.bar(monthly_sales, x="Month", y="Sales",
                    #   title="Average Monthly Sales",
                      template="plotly_white",
                      color="Sales", color_continuous_scale="Blues")
        fig9.update_layout(height=380)
        st.plotly_chart(fig9, width='stretch')

    # ── 10. Monthly Sales Trend (line) ────────────────────────────────────────
    st.markdown("#### 10. Monthly Sales Trend")
    fig10 = px.line(monthly_sales, x="Month", y="Sales",
                    markers=True,
                    # title="Monthly Sales Trend",
                    template="plotly_white")
    fig10.update_traces(line_color="#8e44ad", marker=dict(size=8))
    fig10.update_layout(height=360)
    st.plotly_chart(fig10, width='stretch')


# ═════════════════════════════════════════════════════════════════════════════
# TAB 2 — AGGREGATED MODELS
# ═════════════════════════════════════════════════════════════════════════════
with tab2:
    # st.markdown('<div class="section-header">🏬 Aggregated Data — Forecasting Models</div>', unsafe_allow_html=True)

    raw2 = get_raw()
    df_agg2 = get_agg_df(raw2)

    with st.spinner("Training models on aggregated data… (first run may take 2–5 min)"):
        sarimax_fitted, train2, test2, y_train2, y_test2, exog_train2, exog_test2 = get_sarimax_agg(df_agg2)
        train_size2 = len(y_train2)
        prophet_model, prophet_forecast_df, train_p2, test_p2, df_prophet2 = get_prophet_agg(df_agg2, train_size2)
        # tbats_fitted = get_tbats_agg(y_train2)

    # ── PART 1: Model Result Graphs ───────────────────────────────────────────
    st.markdown("### 📈 Model Forecast Graphs")

    # SARIMAX test predictions
    sarimax_fc = sarimax_fitted.get_forecast(steps=len(y_test2), exog=exog_test2)
    sarimax_preds = sarimax_fc.predicted_mean
    sarimax_ci = sarimax_fc.conf_int()
    sarimax_preds.index = y_test2.index

    # ── SARIMAX Plot ──────────────────────────────────────────────────────────
    st.markdown("#### SARIMAX Forecast with 95% CI")
    fig_sx = go.Figure()
    fig_sx.add_trace(go.Scatter(x=y_test2.index, y=y_test2.values, name="Actual",
                                 line=dict(color="black", width=1.5)))
    fig_sx.add_trace(go.Scatter(x=sarimax_preds.index, y=sarimax_preds.values,
                                 name="SARIMAX Forecast", line=dict(color="#e74c3c", width=2)))
    fig_sx.add_trace(go.Scatter(
        x=list(sarimax_ci.index) + list(sarimax_ci.index[::-1]),
        y=list(sarimax_ci.iloc[:, 0]) + list(sarimax_ci.iloc[:, 1][::-1]),
        fill="toself", fillcolor="rgba(231,76,60,0.15)", line=dict(color="rgba(0,0,0,0)"),
        name="95% CI"
    ))
    fig_sx.update_layout(xaxis_title="Date", yaxis_title="Sales",
                          template="plotly_white", height=420, hovermode="x unified")
    st.plotly_chart(fig_sx, width='stretch')

    # ── Prophet Plot ──────────────────────────────────────────────────────────
    st.markdown("#### Prophet Forecast with Uncertainty Interval")
    prophet_test_slice = prophet_forecast_df[["ds", "yhat", "yhat_lower", "yhat_upper"]].iloc[-len(test_p2):]

    fig_pr = go.Figure()
    fig_pr.add_trace(go.Scatter(x=test_p2["ds"], y=test_p2["y"].values, name="Actual",
                                 line=dict(color="black", width=1.5)))
    fig_pr.add_trace(go.Scatter(x=prophet_test_slice["ds"], y=prophet_test_slice["yhat"],
                                 name="Prophet Forecast", line=dict(color="#27ae60", width=2)))
    fig_pr.add_trace(go.Scatter(
        x=list(prophet_test_slice["ds"]) + list(prophet_test_slice["ds"][::-1]),
        y=list(prophet_test_slice["yhat_lower"]) + list(prophet_test_slice["yhat_upper"][::-1]),
        fill="toself", fillcolor="rgba(39,174,96,0.15)", line=dict(color="rgba(0,0,0,0)"),
        name="Uncertainty"
    ))
    fig_pr.update_layout( xaxis_title="Date", yaxis_title="Sales",
                          template="plotly_white", height=420, hovermode="x unified")
    st.plotly_chart(fig_pr, width='stretch')

    # TBATS test
    # tbats_test_preds = tbats_fitted.forecast(steps=len(y_test2))
    # tbats_test_preds = pd.Series(tbats_test_preds, index=y_test2.index)

    # ── TBATS Plot ────────────────────────────────────────────────────────────
    # st.markdown("#### TBATS — Test Forecast")
    # fig_tb = go.Figure()
    # fig_tb.add_trace(go.Scatter(x=y_test2.index, y=y_test2.values, name="Actual",
    #                              line=dict(color="black", width=1.5)))
    # fig_tb.add_trace(go.Scatter(x=tbats_test_preds.index, y=tbats_test_preds.values,
    #                              name="TBATS Forecast", line=dict(color="#8e44ad", width=2)))
    # fig_tb.update_layout(title="TBATS — Test Period Forecast",
    #                       xaxis_title="Date", yaxis_title="Sales",
    #                       template="plotly_white", height=420, hovermode="x unified")
    # st.plotly_chart(fig_tb, width='stretch')

    # ── Comparison Plot ───────────────────────────────────────────────────────
    st.markdown("#### All Best Models Comparison on Test Period")
    fig_cmp = go.Figure()
    fig_cmp.add_trace(go.Scatter(x=y_test2.index, y=y_test2.values, name="Actual",
                                  line=dict(color="black", width=2)))
    fig_cmp.add_trace(go.Scatter(x=sarimax_preds.index, y=sarimax_preds.values,
                                  name="SARIMAX", line=dict(color="#e74c3c", width=1.5)))
    fig_cmp.add_trace(go.Scatter(x=pd.to_datetime(prophet_test_slice["ds"]),
                                  y=prophet_test_slice["yhat"],
                                  name="Prophet", line=dict(color="#27ae60", width=1.5)))
    # fig_cmp.add_trace(go.Scatter(x=tbats_test_preds.index, y=tbats_test_preds.values,
    #                               name="TBATS", line=dict(color="#8e44ad", width=1.5)))
    fig_cmp.update_layout(xaxis_title="Date", yaxis_title="Sales",
                           template="plotly_white", height=440, hovermode="x unified")
    st.plotly_chart(fig_cmp, width='stretch')

    # ── Accuracy Metrics ──────────────────────────────────────────────────────
    st.markdown("### 📈 Model Accuracy Metrics")

    m_sx = metrics_dict("SARIMAX", y_test2, sarimax_preds,
                         k=len(sarimax_fitted.params), conf_int=sarimax_ci)

    prophet_preds_arr = prophet_test_slice["yhat"].values
    prophet_lower = prophet_test_slice["yhat_lower"].values
    prophet_upper = prophet_test_slice["yhat_upper"].values
    prophet_ci_df = pd.DataFrame({"lower": prophet_lower, "upper": prophet_upper},
                                  index=range(len(prophet_lower)))
    m_pr = metrics_dict("Prophet", test_p2["y"].values, prophet_preds_arr, k=5,
                         conf_int=prophet_ci_df)

    # m_tb = metrics_dict("TBATS", y_test2, tbats_test_preds, k=5)

    # metrics_df = pd.DataFrame([m_sx, m_pr])
    # st.dataframe(
    #     metrics_df.set_index("Model").style
    #         .format({"MAPE (%)": "{:.2f}", "R²": "{:.4f}", "Adj-R²": "{:.4f}",
    #                  "Coverage (95%)": "{:.4f}", "CRPS": "{:.2f}"})
    #         .background_gradient(cmap="RdYlGn", subset=["R²"])
    #         .background_gradient(cmap="RdYlGn_r", subset=["RMSE", "MAE"]),
    #     width='stretch'
    # )
    metrics_df = pd.DataFrame([m_sx, m_pr]).set_index("Model")

    # Scale RMSE and MAE
    metrics_df["RMSE"] = metrics_df["RMSE"] / 1e6
    metrics_df["MAE"] = metrics_df["MAE"] / 1e6

    cols_to_show = ["RMSE", "MAE", "R²", "Coverage (95%)"]

    st.dataframe(
        metrics_df[cols_to_show].style
            .format({
                "RMSE": "{:.2f}",
                "MAE": "{:.2f}",
                "R²": "{:.4f}",
                "Coverage (95%)": "{:.4f}"
            }),
        width='stretch'
    )

    st.markdown("---")
    # ── PART 2: Future Prediction (Prophet + TBATS, numbers only) ────────────
    # st.markdown("### Next 7 Day Sales Prediction Using Prophet")
    # st.info("Predictions use default assumptions: Promo=0, SchoolHoliday=0 for future dates.")

    last_date = df_agg2.index[-1]

    # Prophet future
    prophet_future_7 = prophet_future_forecast(prophet_model, df_prophet2, last_date, steps=7)
    # TBATS future
    # tbats_future_7 = tbats_future_forecast(tbats_fitted, last_date, steps=7)

    

    # with col_p1:
    st.markdown("#### 🟢 Next 7 Day Sales Prediction Using Prophet")
    prophet_out = prophet_future_7[["ds", "yhat", "yhat_lower", "yhat_upper"]].copy()
    prophet_out.columns = ["Date", "Predicted Sales", "Lower CI", "Upper CI"]
    prophet_out["Day"] = prophet_out["Date"].dt.day_name()
    prophet_out["Predicted Sales"] = prophet_out["Predicted Sales"].clip(lower=0).round(0).astype(int)
    prophet_out["Lower CI"] = prophet_out["Lower CI"].clip(lower=0).round(0).astype(int)
    prophet_out["Upper CI"] = prophet_out["Upper CI"].round(0).astype(int)
    prophet_out["Date"] = prophet_out["Date"].dt.strftime("%Y-%m-%d")
    prophet_out = prophet_out[["Date", "Day", "Predicted Sales", "Lower CI", "Upper CI"]]

    st.dataframe(prophet_out, width='stretch', hide_index=True)

    # with col_p2:
    #     st.markdown("#### 🟣 TBATS — Next 7 Days")
        # tbats_out = pd.DataFrame({
        #     "Date": tbats_future_7.index.strftime("%Y-%m-%d"),
        #     "Predicted Sales": tbats_future_7.clip(lower=0).round(0).astype(int).values
        # })
        # st.dataframe(tbats_out, width='stretch', hide_index=True)

    st.markdown("---")
    # ── PART 3: SARIMAX Custom Prediction ────────────────────────────────────
    st.markdown("### 📊 SARIMAX Custom 7 Day Prediction")
    st.markdown("Enter the exogenous variables for the next 7 days:")

    with st.form("sarimax_custom_form"):
        days = [str((last_date + pd.Timedelta(days=i+1)).date()) for i in range(7)]
        promo_inputs, school_inputs = [], []

        col_h1, col_h2, col_h3 = st.columns([2, 1, 1])
        col_h1.markdown("**Date**")
        col_h2.markdown("**Promo**")
        col_h3.markdown("**School Holiday**")

        for day in days:
            c1, c2, c3 = st.columns([2, 1, 1])
            c1.markdown(f"`{day}`")
            # p = c2.selectbox("", [0, 1], key=f"promo_{day}", label_visibility="collapsed")
            # s = c3.selectbox("", [0, 1], key=f"school_{day}", label_visibility="collapsed")
            p = c2.selectbox("Promo", ["No", "Yes"], key=f"promo_{day}", label_visibility="collapsed")
            s = c3.selectbox("School Holiday", ["No", "Yes"], key=f"school_{day}", label_visibility="collapsed")

            promo_inputs.append(1 if p == "Yes" else 0)
            school_inputs.append(1 if s == "Yes" else 0)
            # promo_inputs.append(p)
            # school_inputs.append(s)

        submitted = st.form_submit_button("🔮 Predict with SARIMAX", type="primary")

    if submitted:
        with st.spinner("Running SARIMAX forecast…"):
            sarimax_preds_custom, sarimax_ci_custom = sarimax_future_forecast(
                sarimax_fitted, df_agg2, steps=7,
                user_overrides={"Promo": promo_inputs, "SchoolHoliday": school_inputs}
            )

        future_dates_7 = [str((last_date + pd.Timedelta(days=i+1)).date()) for i in range(7)]
        sarimax_out = pd.DataFrame({
            "Date": future_dates_7,
            "Promo": promo_inputs,
            "School Holiday": school_inputs,
            "Predicted Sales": sarimax_preds_custom.clip(lower=0).round(0).astype(int).values,
            "Lower CI": sarimax_ci_custom.iloc[:, 0].clip(lower=0).round(0).astype(int).values,
            "Upper CI": sarimax_ci_custom.iloc[:, 1].round(0).astype(int).values,
        })

        st.markdown("#### 🔴 SARIMAX — Next 7 Days (Custom Inputs)")
        st.dataframe(sarimax_out, width='stretch', hide_index=True)


# ═════════════════════════════════════════════════════════════════════════════
# TAB 3 — STORE-LEVEL FORECASTING
# ═════════════════════════════════════════════════════════════════════════════
with tab3:
    # st.markdown('<div class="section-header">🏪 Store Level Forecasting</div>', unsafe_allow_html=True)
    # st.markdown("Train and forecast for an individual Rossmann store using SARIMAX(5,0,3)(1,1,1,7).")

    merged_df = get_merged()
    valid_store_ids = get_store_ids(merged_df[merged_df["Open"] == 1])
    min_id, max_id = int(min(valid_store_ids)), int(max(valid_store_ids))

    # ── Step 1: Store selection ───────────────────────────────────────────────
    st.markdown("### Step 1 — Select Store")

    col_s1, col_s2 = st.columns([1, 2])
    with col_s1:
        store_input = st.number_input(
            f"Enter Store Number ({min_id} – {max_id})",
            min_value=min_id, max_value=max_id, value=733, step=1
        )
    with col_s2:
        st.markdown("")
        st.markdown("")
        if store_input not in valid_store_ids:
            st.error(f"Store {store_input} not found in dataset.")
            st.stop()
        else:
            st.success(f"✅ Store **{store_input}** found in dataset.")

    # ── Step 2: Model Training ────────────────────────────────────────────────
    st.markdown("### Step 2 — Model Training")

    with st.spinner(f"Preparing data for Store {store_input}…"):
        store_df = prepare_store_df(merged_df, store_input)

    st.write(f"**Store {store_input} data:** {len(store_df)} days | "
             f"{store_df.index.min().date()} → {store_df.index.max().date()}")

    with st.spinner(f"Training SARIMAX for Store {store_input}… (may take 1–3 min)"):
        s_fitted, s_train, s_test, s_y_train, s_y_test, s_X_train, s_X_test = get_store_model(store_df)

    st.success("✅ Model trained successfully.")

    # Test-period evaluation
    with st.spinner("Evaluating on test period…"):
        try:
            s_fc = s_fitted.get_forecast(steps=len(s_y_test), exog=s_X_test)
            s_preds_test = s_fc.predicted_mean.clip(lower=0)
            s_preds_test.index = s_y_test.index
            s_ci_test = s_fc.conf_int()
        except Exception:
            s_preds_test = s_fitted.predict(
                start=len(s_y_train), end=len(s_y_train) + len(s_y_test) - 1,
                exog=s_X_test
            )
            s_preds_test = pd.Series(s_preds_test.values, index=s_y_test.index).clip(lower=0)
            s_ci_test = None

    # Metrics
    s_rmse, s_mae, s_mape, s_r2, s_adj_r2 = evaluate_model(s_y_test, s_preds_test, k=len(s_fitted.params))

    st.markdown("#### 📈 Model Accuracy on Test Data")
    c1, c2, c3 = st.columns(3)
    c1.metric("RMSE", f"{s_rmse:,.0f}")
    c2.metric("MAE", f"{s_mae:,.0f}")
    c3.metric("MAPE (%)", f"{s_mape:.2f}%")

    # Test forecast chart
    st.markdown(f"#### Test Data Forecast for Store {store_input}")
    fig_store = go.Figure()
    fig_store.add_trace(go.Scatter(x=s_y_test.index, y=s_y_test.values,
                                    name="Actual", line=dict(color="black", width=1.5)))
    fig_store.add_trace(go.Scatter(x=s_preds_test.index, y=s_preds_test.values,
                                    name="Predicted", line=dict(color="#e74c3c", width=2)))
    if s_ci_test is not None:
        fig_store.add_trace(go.Scatter(
            x=list(s_ci_test.index) + list(s_ci_test.index[::-1]),
            y=list(s_ci_test.iloc[:, 0]) + list(s_ci_test.iloc[:, 1][::-1]),
            fill="toself", fillcolor="rgba(231,76,60,0.15)", line=dict(color="rgba(0,0,0,0)"),
            name="95% CI"
        ))
    fig_store.update_layout(
        # title=f"Store {store_input} — SARIMAX Test Forecast",
        xaxis_title="Date", yaxis_title="Sales",
        template="plotly_white", height=420, hovermode="x unified"
    )
    st.plotly_chart(fig_store, width='stretch')

    # ── Step 3: Future Forecasting ────────────────────────────────────────────
    st.markdown("### Step 3 — Next 7 Day Forecast")
    st.markdown("Provide the exogenous variables for the next 7 forecast days:")

    store_last_date = store_df.index[-1]
    store_days = [str((store_last_date + pd.Timedelta(days=i+1)).date()) for i in range(7)]

    with st.form("store_forecast_form"):
        s_promo_inputs, s_school_inputs = [], []

        sc1, sc2, sc3 = st.columns([2, 1, 1])
        sc1.markdown("**Date**")
        sc2.markdown("**Promo**")
        sc3.markdown("**School Holiday**")

        for day in store_days:
            c1, c2, c3 = st.columns([2, 1, 1])
            c1.markdown(f"`{day}`")
           
            p = c2.selectbox("Promo_2", ["No", "Yes"], key=f"s_promo_{day}", label_visibility="collapsed")
            s = c3.selectbox("School Holiday_2", ["No", "Yes"], key=f"s_school_{day}", label_visibility="collapsed")

            s_promo_inputs.append(1 if p == "Yes" else 0)
            s_school_inputs.append(1 if s == "Yes" else 0)

        store_submitted = st.form_submit_button("🔮 Predict Next 7 Days", type="primary")

    if store_submitted:
        with st.spinner("Running store-level SARIMAX forecast…"):
            store_preds_7 = sarimax_store_forecast(
                s_fitted, store_df, steps=7,
                user_promo=s_promo_inputs,
                user_school_holiday=s_school_inputs
            )

        st.markdown(f"#### 🏪 Store {store_input} — Predicted Sales")
        store_out = pd.DataFrame({
            "Date": store_days,
            "Promo": s_promo_inputs,
            "School Holiday": s_school_inputs,
            "Predicted Sales": store_preds_7.round(0).astype(int).values,
        })
        st.dataframe(store_out, width='stretch', hide_index=True)


# ═════════════════════════════════════════════════════════════════════════════
# TAB 4 — BUSINESS INSIGHTS
# ═════════════════════════════════════════════════════════════════════════════
with tab4:

    @st.cache_data(show_spinner="Computing store-level metrics…")
    def get_store_metrics(_merged_df):
        df = _merged_df.copy()
        df = df[df["Open"] == 1].copy()
        df["Date"] = pd.to_datetime(df["Date"])

        # Per-store total and daily avg
        store_stats = df.groupby("Store").agg(
            TotalSales=("Sales", "sum"),
            AvgDailySales=("Sales", "mean"),
            SalesStd=("Sales", "std"),
            NumDays=("Sales", "count"),
        ).reset_index()

        # Growth rate: compare last 6 months vs previous 6 months
        df_sorted = df.sort_values("Date")
        mid = df_sorted["Date"].median()
        first_half = df_sorted[df_sorted["Date"] < mid].groupby("Store")["Sales"].mean()
        second_half = df_sorted[df_sorted["Date"] >= mid].groupby("Store")["Sales"].mean()
        growth = ((second_half - first_half) / (first_half + 1e-8) * 100).rename("GrowthRate").reset_index()
        store_stats = store_stats.merge(growth, on="Store", how="left")

        # Promo vs non-promo per store
        promo_avg = df.groupby(["Store", "Promo"])["Sales"].mean().unstack(fill_value=0)
        promo_avg.columns = ["AvgSales_NoPromo", "AvgSales_Promo"]
        promo_avg = promo_avg.reset_index()
        promo_avg["PromoLift_pct"] = (
            (promo_avg["AvgSales_Promo"] - promo_avg["AvgSales_NoPromo"]) /
            (promo_avg["AvgSales_NoPromo"] + 1e-8) * 100
        )
        store_stats = store_stats.merge(promo_avg, on="Store", how="left")

        # School holiday impact per store
        holiday_avg = df.groupby(["Store", "SchoolHoliday"])["Sales"].mean().unstack(fill_value=0)
        holiday_avg.columns = ["AvgSales_NoHoliday", "AvgSales_Holiday"] if 1 in df["SchoolHoliday"].unique() else ["AvgSales_NoHoliday"]
        if "AvgSales_Holiday" not in holiday_avg.columns:
            holiday_avg["AvgSales_Holiday"] = 0
        holiday_avg = holiday_avg.reset_index()
        holiday_avg["HolidayDrop_pct"] = (
            (holiday_avg["AvgSales_Holiday"] - holiday_avg["AvgSales_NoHoliday"]) /
            (holiday_avg["AvgSales_NoHoliday"] + 1e-8) * 100
        )
        store_stats = store_stats.merge(holiday_avg[["Store", "AvgSales_NoHoliday", "AvgSales_Holiday", "HolidayDrop_pct"]], on="Store", how="left")

        return store_stats

    @st.cache_data(show_spinner="Loading time series per store…")
    def get_store_timeseries(_merged_df):
        df = _merged_df.copy()
        df = df[df["Open"] == 1].copy()
        df["Date"] = pd.to_datetime(df["Date"])
        return df.groupby(["Store", "Date"])["Sales"].sum().reset_index()

    merged_bi = get_merged()
    store_metrics = get_store_metrics(merged_bi)
    total_revenue = store_metrics["TotalSales"].sum()

    # ── Section A: Multi-Store Comparison ────────────────────────────────────
    st.markdown("## 📊 Multi Store Comparison")
    st.markdown("Compare up to **10 stores** across key performance metrics.")

    all_store_ids = sorted(store_metrics["Store"].unique().tolist())
    default_stores = all_store_ids[:5]
    selected_stores = st.multiselect(
        "Select stores to compare (up to 10):",
        options=all_store_ids,
        default=default_stores,
        max_selections=10,
    )

    if selected_stores:
        sel_metrics = store_metrics[store_metrics["Store"].isin(selected_stores)].copy()
        sel_metrics["Store"] = sel_metrics["Store"].astype(str)

        # KPI Summary Row
        col_m1, col_m2, col_m3 = st.columns(3)
        with col_m1:
            st.metric("Total Sales (Selected)", f"€{sel_metrics['TotalSales'].sum():,.0f}")
        with col_m2:
            st.metric("Avg Daily Sales (mean)", f"€{sel_metrics['AvgDailySales'].mean():,.0f}")
        with col_m3:
            best_growth = sel_metrics.loc[sel_metrics['GrowthRate'].idxmax()]
            st.metric("Best Growth", f"Store {best_growth['Store']}", f"{best_growth['GrowthRate']:.1f}%")

        col_line, col_bar = st.columns(2)

        with col_line:
            st.markdown("#### 📈 Daily Sales Trend — Multi Store")
            store_ts = get_store_timeseries(merged_bi)
            ts_sel = store_ts[store_ts["Store"].isin(selected_stores)].copy()
            ts_sel["Store"] = ts_sel["Store"].astype(str)
            # Resample weekly for clarity
            ts_sel["Week"] = pd.to_datetime(ts_sel["Date"]).dt.to_period("W").apply(lambda r: r.start_time)
            ts_weekly = ts_sel.groupby(["Store", "Week"])["Sales"].mean().reset_index()
            fig_line = px.line(
                ts_weekly, x="Week", y="Sales", color="Store",
                template="plotly_white", labels={"Sales": "Avg Weekly Sales", "Week": "Week"},
            )
            fig_line.update_layout(height=380, hovermode="x unified", legend_title="Store")
            st.plotly_chart(fig_line, width='stretch')

        with col_bar:
            st.markdown("#### 🏆 Store Revenue Ranking")
            sel_sorted = sel_metrics.sort_values("TotalSales", ascending=True)
            fig_rank = px.bar(
                sel_sorted, x="TotalSales", y="Store", orientation="h",
                color="TotalSales", color_continuous_scale="Blues",
                template="plotly_white", labels={"TotalSales": "Total Sales (€)", "Store": "Store ID"},
            )
            fig_rank.update_layout(height=380, coloraxis_showscale=False, yaxis=dict(type="category"))
            st.plotly_chart(fig_rank, width='stretch')

        st.markdown("#### 📋 Comparison Table")
        display_tbl = sel_metrics[["Store", "TotalSales", "AvgDailySales", "GrowthRate", "NumDays"]].copy()
        display_tbl["TotalSales"] = display_tbl["TotalSales"].round(0).astype(int)
        display_tbl["AvgDailySales"] = display_tbl["AvgDailySales"].round(0).astype(int)
        display_tbl["GrowthRate"] = display_tbl["GrowthRate"].round(2)
        display_tbl.columns = ["Store", "Total Sales (€)", "Avg Daily Sales (€)", "Growth Rate (%)", "Trading Days"]
        st.dataframe(display_tbl.set_index("Store"), width='stretch')

    st.markdown("---")

    # ── Section B: Why Did Sales Change? ────────────────────────────────────
    st.markdown("## 💡 Why Did Sales Change?")
    st.markdown("Understand the **drivers** behind revenue performance across all stores.")

    col_promo, col_hol = st.columns(2)

    with col_promo:
        st.markdown("#### 🎯 Promo Impact on Sales")
        overall_promo = store_metrics[["PromoLift_pct"]].dropna()
        avg_promo_lift = overall_promo["PromoLift_pct"].mean()
        median_promo_lift = overall_promo["PromoLift_pct"].median()

        st.metric("Avg Promo Lift across all stores", f"+{avg_promo_lift:.1f}%")

        # fig_promo_hist = px.histogram(
        #     store_metrics.dropna(subset=["PromoLift_pct"]),
        #     x="PromoLift_pct", nbins=40,
        #     template="plotly_white",
        #     color_discrete_sequence=["#27ae60"],
        #     labels={"PromoLift_pct": "Promo Lift (%)"},
        # )
        # fig_promo_hist.add_vline(x=avg_promo_lift, line_dash="dash", line_color="#e74c3c",
        #                           annotation_text=f"Mean: {avg_promo_lift:.1f}%")
        # fig_promo_hist.update_layout(height=340, showlegend=False)
        # st.plotly_chart(fig_promo_hist, width='stretch')

        # Promo vs no-promo bar
        # promo_summary = pd.DataFrame({
        #     "Condition": ["No Promo", "Promo Active"],
        #     "Avg Sales": [
        #         store_metrics["AvgSales_NoPromo"].mean(),
        #         store_metrics["AvgSales_Promo"].mean(),
        #     ]
        # })
        # fig_promo_bar = px.bar(promo_summary, x="Condition", y="Avg Sales",
        #                         color="Condition",
        #                         color_discrete_map={"No Promo": "#e74c3c", "Promo Active": "#27ae60"},
        #                         template="plotly_white")
        # fig_promo_bar.update_layout(height=320, showlegend=False)
        # st.plotly_chart(fig_promo_bar, width='stretch')

    with col_hol:
        st.markdown("#### 🏫 School Holiday Impact")
        holiday_clean = store_metrics.dropna(subset=["HolidayDrop_pct"])
        avg_holiday_effect = holiday_clean["HolidayDrop_pct"].mean()
        median_holiday_effect = holiday_clean["HolidayDrop_pct"].median()

        delta_color = "normal" if avg_holiday_effect >= 0 else "inverse"
        sign = "+" if avg_holiday_effect >= 0 else ""
        st.metric("Avg School Holiday Effect", f"{sign}{avg_holiday_effect:.1f}%")
                #   delta=f"vs non-holiday baseline", delta_color=delta_color

        # fig_hol_hist = px.histogram(
        #     holiday_clean,
        #     x="HolidayDrop_pct", nbins=40,
        #     template="plotly_white",
        #     color_discrete_sequence=["#e67e22"],
        #     labels={"HolidayDrop_pct": "Holiday Effect (%)"},
        # )
        # fig_hol_hist.add_vline(x=avg_holiday_effect, line_dash="dash", line_color="#2980b9",
        #                         annotation_text=f"Mean: {avg_holiday_effect:.1f}%")
        # fig_hol_hist.update_layout(height=340, showlegend=False)
        # st.plotly_chart(fig_hol_hist, width='stretch')

        # Holiday vs no-holiday bar
        # holiday_summary = pd.DataFrame({
        #     "Condition": ["No Holiday", "School Holiday"],
        #     "Avg Sales": [
        #         store_metrics["AvgSales_NoHoliday"].mean(),
        #         store_metrics["AvgSales_Holiday"].mean(),
        #     ]
        # })
        # fig_hol_bar = px.bar(holiday_summary, x="Condition", y="Avg Sales",
        #                       color="Condition",
        #                       color_discrete_map={"No Holiday": "#3498db", "School Holiday": "#e67e22"},
        #                       template="plotly_white")
        # fig_hol_bar.update_layout(height=320, showlegend=False)
        # st.plotly_chart(fig_hol_bar, width='stretch')

    st.markdown("---")

    # ── Section C: Top / Worst Performing Stores ─────────────────────────────
    st.markdown("## 🏅 Top & Worst Performing Stores")

    perf_metric = st.radio(
        "Rank stores by:",
        ["Revenue (Total Sales)", "Avg Daily Sales", "Consistency (Low Variance)"],
        horizontal=True,
    )

    n_show = st.slider("Number of stores to show", min_value=5, max_value=15, value=10)

    if perf_metric == "Revenue (Total Sales)":
        sort_col, label = "TotalSales", "Total Sales (€)"
        ascending_best = False
    elif perf_metric == "Avg Daily Sales":
        sort_col, label = "AvgDailySales", "Avg Daily Sales (€)"
        ascending_best = False
    else:
        sort_col, label = "SalesStd", "Sales Std Dev (€)  [lower = more consistent]"
        ascending_best = True

    top_n = store_metrics.nsmallest(n_show, sort_col) if ascending_best else store_metrics.nlargest(n_show, sort_col)
    worst_n = store_metrics.nlargest(n_show, sort_col) if ascending_best else store_metrics.nsmallest(n_show, sort_col)

    top_n = top_n.sort_values(sort_col, ascending=ascending_best)
    worst_n = worst_n.sort_values(sort_col, ascending=not ascending_best)

    col_top, col_worst = st.columns(2)

    with col_top:
        emoji = "🌟" if not ascending_best else "🎯"
        title = f"{emoji} Top {n_show} Stores"
        st.markdown(f"#### {title}")
        top_n["Store"] = top_n["Store"].astype(str)
        fig_top = px.bar(
            top_n, x=sort_col, y="Store", orientation="h",
            color=sort_col, color_continuous_scale="Greens",
            template="plotly_white", labels={sort_col: label, "Store": "Store ID"},
        )
        fig_top.update_layout(height=420, coloraxis_showscale=False, yaxis=dict(type="category"))
        st.plotly_chart(fig_top, width='stretch')

        top_display = top_n[["Store", "TotalSales", "AvgDailySales", "GrowthRate"]].copy()
        top_display.columns = ["Store", "Total Sales", "Avg Daily", "Growth %"]
        top_display["Total Sales"] = top_display["Total Sales"].round(0).astype(int)
        top_display["Avg Daily"] = top_display["Avg Daily"].round(0).astype(int)
        top_display["Growth %"] = top_display["Growth %"].round(1)
        st.dataframe(top_display.set_index("Store"), width='stretch')

    with col_worst:
        title2 = f"⚠️ Worst {n_show} Stores"
        st.markdown(f"#### {title2}")
        worst_n["Store"] = worst_n["Store"].astype(str)
        fig_worst = px.bar(
            worst_n, x=sort_col, y="Store", orientation="h",
            color=sort_col, color_continuous_scale="Reds",
            template="plotly_white", labels={sort_col: label, "Store": "Store ID"},
        )
        fig_worst.update_layout(height=420, coloraxis_showscale=False, yaxis=dict(type="category"))
        st.plotly_chart(fig_worst, width='stretch')

        worst_display = worst_n[["Store", "TotalSales", "AvgDailySales", "GrowthRate"]].copy()
        worst_display.columns = ["Store", "Total Sales", "Avg Daily", "Growth %"]
        worst_display["Total Sales"] = worst_display["Total Sales"].round(0).astype(int)
        worst_display["Avg Daily"] = worst_display["Avg Daily"].round(0).astype(int)
        worst_display["Growth %"] = worst_display["Growth %"].round(1)
        st.dataframe(worst_display.set_index("Store"), width='stretch')

    st.markdown("---")

    # ── Section: Competition Impact ─────────────────────────────────────────────
    # st.markdown("---")
    st.markdown("## 🏪 Competition Distance Impact on Sales")
    # st.markdown("Analyze how **distance to nearest competitor** affects store performance.")

    # Ensure CompetitionDistance exists
    if "CompetitionDistance" in merged_bi.columns:

        comp_df = merged_bi.copy()
        comp_df = comp_df[comp_df["Open"] == 1]

        # Remove extreme outliers for cleaner graph
        # comp_df = comp_df[comp_df["CompetitionDistance"] < comp_df["CompetitionDistance"].quantile(0.99)]

        # ── 1. Scatter Plot ─────────────────────────────────────────────
        st.markdown("#### 📉 Sales vs Competition Distance")

        fig_scatter = px.scatter(
            comp_df,  # sample for speed
            x="CompetitionDistance",
            y="Sales",
            opacity=0.5,
            trendline="lowess",
            template="plotly_white",
            labels={
                "CompetitionDistance": "Distance to Competitor (meters)",
                "Sales": "Sales"
            }
        )

        fig_scatter.update_layout(height=400)
        st.plotly_chart(fig_scatter, width='stretch')

        # ── 2. Binned Analysis ─────────────────────────────────────────
        st.markdown("#### 📊 Average Sales by Competition Distance Range")

        bins = [0, 500, 1000, 2000, 5000, 10000, 20000]
        labels = ["<500m", "500-1km", "1-2km", "2-5km", "5-10km", "10km+"]

        comp_df["DistanceBin"] = pd.cut(
            comp_df["CompetitionDistance"],
            bins=bins,
            labels=labels
        )

        bin_sales = comp_df.groupby("DistanceBin")["Sales"].mean().reset_index()

        fig_bar = px.bar(
            bin_sales,
            x="DistanceBin",
            y="Sales",
            color="Sales",
            color_continuous_scale="Blues",
            template="plotly_white",
            labels={"DistanceBin": "Distance Range", "Sales": "Avg Sales"}
        )

        fig_bar.update_layout(height=400)
        st.plotly_chart(fig_bar, width='stretch')

        # # ── 3. Correlation Metric ──────────────────────────────────────
        # corr = comp_df["CompetitionDistance"].corr(comp_df["Sales"])

        # st.metric("Correlation (Distance vs Sales)", f"{corr:.3f}")

        # # ── 4. Business Insight Text ───────────────────────────────────
        # if corr > 0:
        #     insight = "Stores farther from competitors tend to have higher sales."
        # else:
        #     insight = "Stores closer to competitors may experience reduced sales due to competition."

        # st.info(f"📌 Insight: {insight}")

    else:
        st.warning("CompetitionDistance feature not available in dataset.")

        # ── Section D: Sales Contribution Analysis ────────────────────────────────
        # st.markdown("## 📊 Sales Contribution Analysis")
        # st.markdown("Which stores drive the **80/20 rule**? Identify your revenue pillars.")

        # store_contrib = store_metrics[["Store", "TotalSales"]].copy()
        # store_contrib = store_contrib.sort_values("TotalSales", ascending=False).reset_index(drop=True)
        # store_contrib["CumSales"] = store_contrib["TotalSales"].cumsum()
        # store_contrib["CumPct"] = store_contrib["CumSales"] / total_revenue * 100
        # store_contrib["ContribPct"] = store_contrib["TotalSales"] / total_revenue * 100
        # store_contrib["StoreRank"] = range(1, len(store_contrib) + 1)

        # # Count stores in top 80%
        # stores_80 = (store_contrib["CumPct"] <= 80).sum() + 1
        # pct_stores_80 = stores_80 / len(store_contrib) * 100

        # col_pie, col_pareto = st.columns(2)

        # with col_pie:
        #     st.markdown("#### 🥧 Revenue Share — Top 20 Stores vs Rest")
        #     top20_sales = store_contrib.head(20)["TotalSales"].sum()
        #     rest_sales = total_revenue - top20_sales
        #     pie_df = pd.DataFrame({
        #         "Group": [f"Top 20 Stores", "All Other Stores"],
        #         "Sales": [top20_sales, rest_sales],
        #     })
        #     fig_pie = px.pie(
        #         pie_df, values="Sales", names="Group",
        #         color_discrete_sequence=["#2980b9", "#bdc3c7"],
        #         template="plotly_white",
        #     )
        #     fig_pie.update_traces(textinfo="percent+label", hole=0.38)
        #     fig_pie.update_layout(height=400, showlegend=True)
        #     st.plotly_chart(fig_pie, width='stretch')

        #     st.info(f"🎯 **Pareto Insight:** The top **{stores_80} stores** ({pct_stores_80:.1f}% of all stores) "
        #             f"generate **80%** of total revenue.")

        # with col_pareto:
        #     st.markdown("#### 📉 Pareto Chart — Cumulative Revenue by Store Rank")
        #     fig_pareto = go.Figure()
        #     fig_pareto.add_trace(go.Bar(
        #         x=store_contrib["StoreRank"],
        #         y=store_contrib["ContribPct"],
        #         name="Individual Contribution %",
        #         marker_color="#3498db",
        #         opacity=0.7,
        #     ))
        #     fig_pareto.add_trace(go.Scatter(
        #         x=store_contrib["StoreRank"],
        #         y=store_contrib["CumPct"],
        #         name="Cumulative %",
        #         mode="lines",
        #         line=dict(color="#e74c3c", width=2.5),
        #         yaxis="y2",
        #     ))
        #     fig_pareto.add_hline(y=80, line_dash="dash", line_color="#f39c12",
        #                           annotation_text="80% threshold", yref="y2")
        #     fig_pareto.update_layout(
        #         height=400,
        #         template="plotly_white",
        #         xaxis_title="Store Rank",
        #         yaxis_title="Individual Contribution (%)",
        #         yaxis2=dict(title="Cumulative %", overlaying="y", side="right", range=[0, 105]),
        #         legend=dict(x=0.5, y=1.1, orientation="h"),
        #         hovermode="x unified",
        #     )
        #     st.plotly_chart(fig_pareto, width='stretch')

        # # Full contribution table
        # with st.expander("📋 Full Store Contribution Table", expanded=False):
        #     contrib_display = store_contrib[["StoreRank", "Store", "TotalSales", "ContribPct", "CumPct"]].copy()
        #     contrib_display.columns = ["Rank", "Store", "Total Sales (€)", "Contribution (%)", "Cumulative (%)"]
        #     contrib_display["Total Sales (€)"] = contrib_display["Total Sales (€)"].round(0).astype(int)
        #     contrib_display["Contribution (%)"] = contrib_display["Contribution (%)"].round(3)
        #     contrib_display["Cumulative (%)"] = contrib_display["Cumulative (%)"].round(2)
        #     st.dataframe(contrib_display.set_index("Rank"), width='stretch')
