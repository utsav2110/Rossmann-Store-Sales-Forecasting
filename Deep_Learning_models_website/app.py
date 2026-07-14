"""
╔══════════════════════════════════════════════════════════════════════╗
║             Rossmann Sales Predictor  —  Streamlit App              ║
║                                                                      ║
║  Tab 1 : Aggregate Prediction  (22 features, aggregate models)      ║
║  Tab 2 : Store-wise Prediction (30 features, store models)          ║
║  Tab 3 : Data Insights                                               ║
╚══════════════════════════════════════════════════════════════════════╝

Run:
    streamlit run app.py
"""

import os, sys, traceback
sys.path.insert(0, os.path.dirname(__file__))

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, date, timedelta
import plotly.graph_objects as go
import plotly.express as px

# Import custom Keras layers early to ensure they're registered
from models.nhits_layers import NHiTSBlock, NHiTS
from models.load_models import available_models, is_quantile_model, store_models_available
from models.predict import predict, format_inr
from utils.scaler import load_scalers


# ══════════════════════════════════════════════════════════════════════
# PAGE CONFIG
# ══════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="Rossmann Sales Predictor",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
/* ── metric card (point prediction) ── */
.metric-card {
    background: linear-gradient(135deg,#667eea,#764ba2);
    border-radius:16px; padding:1.6rem; color:#fff;
    text-align:center; box-shadow:0 8px 32px rgba(102,126,234,.3); margin:.5rem 0;
}
.metric-card .lbl { font-size:.8rem; opacity:.8; text-transform:uppercase; letter-spacing:1px; }
.metric-card .val { font-size:2.2rem; font-weight:700; margin-top:.3rem; }

/* ── quantile cards ── */
.q-row { display:flex; gap:1rem; margin-top:.5rem; }
.q-card { flex:1; border-radius:12px; padding:1.1rem; text-align:center; color:#fff; }
.q-low  { background:linear-gradient(135deg,#f093fb,#f5576c); }
.q-med  { background:linear-gradient(135deg,#4facfe,#00f2fe); }
.q-high { background:linear-gradient(135deg,#43e97b,#38f9d7); }
.q-card .ql { font-size:.72rem; opacity:.9; text-transform:uppercase; letter-spacing:.8px; }
.q-card .qv { font-size:1.45rem; font-weight:700; margin-top:.25rem; }

/* ── info / warning boxes ── */
.info-box  { background:#1e1e2e; border-left:4px solid #667eea; border-radius:8px;
             padding:.75rem 1rem; margin:.4rem 0; font-size:.86rem; color:#cdd6f4; }
.warn-box  { background:#2a1f1f; border-left:4px solid #f5576c; border-radius:8px;
             padding:.75rem 1rem; margin:.4rem 0; font-size:.86rem; color:#f38ba8; }

/* ── section title ── */
.stitle { font-size:1.05rem; font-weight:600; color:#667eea; margin-bottom:.4rem; }

/* ── fancy divider ── */
hr.fancy { border:none; height:2px;
           background:linear-gradient(90deg,#667eea,#764ba2,transparent); margin:1.2rem 0; }

/* ── pipeline badge ── */
.badge-agg   { background:#667eea; color:#fff; border-radius:20px;
               padding:.15rem .75rem; font-size:.78rem; font-weight:600; }
.badge-store { background:#43e97b; color:#1e1e2e; border-radius:20px;
               padding:.15rem .75rem; font-size:.78rem; font-weight:600; }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════
# SCALER LOADING (cached separately for each pipeline)
# ══════════════════════════════════════════════════════════════════════

@st.cache_resource(show_spinner=False)
def get_agg_scalers():
    return load_scalers("aggregate")

@st.cache_resource(show_spinner=False)
def get_store_scalers():
    return load_scalers("store")

# Try loading aggregate scalers (required)
try:
    agg_feature_scaler, agg_target_scaler = get_agg_scalers()
    agg_scalers_ok = True
except Exception as e:
    agg_scalers_ok = False
    agg_scaler_err = str(e)

# Try loading store scalers (optional — only if user trained store models)
try:
    store_feature_scaler, store_target_scaler = get_store_scalers()
    store_scalers_ok = True
except Exception as e:
    store_scalers_ok = False
    store_scaler_err = str(e)


# ══════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("## 🛒 Rossmann Predictor")
    st.caption("Deep Learning Sales Forecasting")
    st.markdown("<hr class='fancy'>", unsafe_allow_html=True)

    # ── System status ───────────────────────────────────────
    st.markdown("### 📦 System Status")

    def _status(ok, label):
        icon = "✅" if ok else "❌"
        st.markdown(f"{icon} {label}")

    _status(agg_scalers_ok,   "Aggregate scalers")
    _status(len(available_models("aggregate")) > 0, "Aggregate models")
    _status(store_scalers_ok, "Store scalers")
    _status(store_models_available(), "Store models")

    st.markdown("<hr class='fancy'>", unsafe_allow_html=True)
    st.markdown("### ℹ️  Architecture")
    st.caption(
        "**Aggregate tab** uses models trained on daily total sales "
        "(22 features, seq_len=60).  \n\n"
        "**Store tab** uses *separate* models trained on per-store sales "
        "(30 features, seq_len=60).  \n\n"
        "Place your store models in `saved_models/store/`."
    )


# ══════════════════════════════════════════════════════════════════════
# HEADER
# ══════════════════════════════════════════════════════════════════════

st.title("🛒 Rossmann Sales Predictor")
st.caption("Two independent pipelines: **Aggregate** total-sales forecast · **Store-wise** per-store forecast")
st.markdown("<hr class='fancy'>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════
# SHARED HELPERS
# ══════════════════════════════════════════════════════════════════════

def render_point_result(val: float, model_name: str, pipeline_label: str):
    st.markdown(f"""
    <div class="metric-card">
        <div class="lbl">{pipeline_label} &nbsp;·&nbsp; {model_name}</div>
        <div class="val">{format_inr(val)}</div>
    </div>""", unsafe_allow_html=True)

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=val,
        number={"prefix": "€ ", "valueformat": ",.0f"},
        gauge={
            "axis": {"range": [0, val * 2.2]},
            "bar":  {"color": "#667eea"},
            "steps": [
                {"range": [0,       val * 0.6], "color": "rgba(245, 87, 108, 0.13)"},
                {"range": [val*0.6, val * 1.6], "color": "rgba(67, 233, 123, 0.13)"},
            ],
            "threshold": {"line": {"color": "#f5576c", "width": 3},
                          "thickness": .75, "value": val},
        },
        title={"text": "Predicted Sales"},
    ))
    fig.update_layout(height=260, margin=dict(t=40, b=0))
    st.plotly_chart(fig, use_container_width=True)


def render_quantile_result(low: float, med: float, high: float, model_name: str, pipeline_label: str):
    st.markdown(f"""
    <p style="color:#aaa;font-size:.82rem;margin-bottom:.4rem">
        {pipeline_label} &nbsp;·&nbsp; {model_name}
    </p>
    <div class="q-row">
        <div class="q-card q-low">
            <div class="ql">⬇ Lower Bound</div>
            <div class="qv">{format_inr(low)}</div>
        </div>
        <div class="q-card q-med">
            <div class="ql">🎯 Expected (Median)</div>
            <div class="qv">{format_inr(med)}</div>
        </div>
        <div class="q-card q-high">
            <div class="ql">⬆ Upper Bound</div>
            <div class="qv">{format_inr(high)}</div>
        </div>
    </div>""", unsafe_allow_html=True)

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=["Lower", "Median (Expected)", "Upper"],
        y=[low, med, high],
        marker_color=["#f5576c", "#4facfe", "#43e97b"],
        text=[format_inr(v) for v in [low, med, high]],
        textposition="outside",
    ))
    fig.update_layout(
        title="Prediction Interval",
        yaxis_title="Sales (€)",
        height=280, margin=dict(t=40, b=0), showlegend=False,
    )
    st.plotly_chart(fig, use_container_width=True)


def render_result(result: dict, model_name: str, pipeline_label: str):
    st.markdown("<hr class='fancy'>", unsafe_allow_html=True)
    st.markdown("## 🎯 Prediction Result")
    if "prediction" in result:
        render_point_result(result["prediction"], model_name, pipeline_label)
    else:
        render_quantile_result(result["lower"], result["median"], result["upper"],
                               model_name, pipeline_label)


def show_traceback():
    with st.expander("🔍 Full error traceback"):
        st.code(traceback.format_exc())


# ══════════════════════════════════════════════════════════════════════
# TABS
# ══════════════════════════════════════════════════════════════════════

tab_agg, tab_store, tab_insights = st.tabs([
    "📊 Aggregate Prediction",
    "🏪 Store-wise Prediction",
    "📈 Data Insights",
])


# ──────────────────────────────────────────────────────────────────────
# TAB 1  —  AGGREGATE PREDICTION (UNIFIED MODE)
# Input  : prediction_date + DayOfWeek + Promo + SchoolHoliday
# Models : saved_models/aggregate/  (22 features, seq_len=60, dual-input models)
# Scaler : saved_models/aggregate/feature_scaler.pkl (22 features)
# ──────────────────────────────────────────────────────────────────────

with tab_agg:
    st.markdown(
        '<span class="badge-agg">AGGREGATE PIPELINE</span> '
        '&nbsp; 22 features · seq_len=60 · global daily sales (dual-input models)',
        unsafe_allow_html=True,
    )
    st.markdown("### 📊 Aggregate Sales Prediction")
    st.caption(
        "Forecast the **total sales across all Rossmann stores** for a selected date. "
        "Provide the date and its conditions (Promo, Holiday status, Day of Week)."
    )

    # Scaler guard
    if not agg_scalers_ok:
        st.error(f"❌ Aggregate scalers not loaded: {agg_scaler_err}")
        st.info("Place `feature_scaler.pkl` and `target_scaler.pkl` inside `saved_models/aggregate/`")
        st.stop()

    # Model selector (aggregate only)
    agg_models = available_models("aggregate")
    col_m, col_info = st.columns([2, 3])
    with col_m:
        agg_model = st.selectbox(
            "🤖 Model (aggregate)",
            agg_models,
            help="These models live in saved_models/aggregate/ and expect 22 features + 3 future features.",
        )
    with col_info:
        if is_quantile_model(agg_model):
            st.info("📊 Quantile model — outputs Lower / Median / Upper bounds.")
        else:
            st.success("🎯 Point model — outputs a single sales estimate.")

    st.markdown("<hr class='fancy'>", unsafe_allow_html=True)

    # ── UNIFIED INPUT: Date + 3 Features ──────────────────────────────
    st.markdown('<p class="stitle">📅 Prediction Date & Conditions</p>', unsafe_allow_html=True)

    # Date input (max = last training date)
    pred_date_agg = st.date_input(
        "Select date to predict (≤ 2015-07-31)",
        value=date(2015, 7, 31),
        min_value=date(2013, 2, 1),
        max_value=date(2015, 7, 31),
        key="agg_unified_date",
        help="Must be ≤ 2015-07-31 (last training date). The app loads the 60 days before this date.",
    )

    # 3 Feature inputs
    st.markdown('<p class="stitle">Day Conditions</p>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    
    with c1:
        dow = st.selectbox(
            "Day of Week",
            [1, 2, 3, 4, 5, 6, 7],
            format_func=lambda x: ["Monday", "Tuesday", "Wednesday", "Thursday",
                                    "Friday", "Saturday", "Sunday"][x-1],
            key="agg_unified_dow",
        )
    
    with c2:
        promo_agg = st.radio(
            "Promo Active?", [0, 1],
            format_func=lambda x: "Yes ✅" if x else "No ❌",
            horizontal=True,
            key="agg_unified_promo",
        )
    
    with c3:
        school_agg = st.radio(
            "School Holiday?", [0, 1],
            format_func=lambda x: "Yes 🏫" if x else "No ❌",
            horizontal=True,
            key="agg_unified_school",
        )

    st.markdown(
        '<div class="info-box">ℹ️ The last 60 days before the selected date are loaded from '
        '<code>data/train.csv</code>. The 3 features above describe the conditions on the '
        'prediction date. Lag features, rolling statistics, and cyclical encodings are '
        'computed automatically.</div>',
        unsafe_allow_html=True,
    )

    if st.button("🚀 Predict Aggregate Sales", type="primary", use_container_width=True, key="btn_agg_unified"):
        with st.spinner("Loading data → engineering features → running inference …"):
            try:
                from preprocessing.aggregate import prepare_aggregate_unified
                seq, future_feats = prepare_aggregate_unified(
                    prediction_date=datetime.combine(pred_date_agg, datetime.min.time()),
                    day_of_week=dow,
                    promo=promo_agg,
                    school_holiday=school_agg,
                    feature_scaler=agg_feature_scaler,
                    model_name=agg_model,
                )
                result = predict("aggregate", agg_model, seq, agg_target_scaler, future_features=future_feats)
                render_result(result, agg_model, "Aggregate — All Stores")
            except Exception:
                st.error("❌ Prediction failed. See traceback below.")
                show_traceback()


# ──────────────────────────────────────────────────────────────────────
# TAB 2  —  STORE-WISE PREDICTION
# Input  : Store ID + Date + Promo + SchoolHoliday
# Models : saved_models/store/  (30 features, seq_len=60)
# Scaler : saved_models/store/feature_scaler.pkl (30 features)
#
# ⚠️  These are SEPARATE models from the aggregate ones.
#    You must train store-level models and place them in saved_models/store/.
# ──────────────────────────────────────────────────────────────────────

with tab_store:
    st.markdown(
        '<span class="badge-store">STORE PIPELINE</span> '
        '&nbsp; 30 features · seq_len=60 · per-store sales',
        unsafe_allow_html=True,
    )
    st.markdown("### 🏪 Store-wise Sales Prediction")
    st.caption(
        "Forecast next-day sales for **one specific store**.  "
        "This pipeline uses store metadata (StoreType, CompetitionDistance, Promo2 …) "
        "and lag features computed from **that store's own history only**."
    )

    # ── Missing store models banner ───────────────────────────────────
    if not store_scalers_ok or not store_models_available():
        st.markdown("""
        <div class="warn-box">
        ⚠️ <strong>Store models not found.</strong><br><br>
        The store-wise pipeline requires <strong>separately trained models</strong>
        (30 input features) placed in <code>saved_models/store/</code>.<br><br>
        These are different from the aggregate models — a 22-feature aggregate
        model <em>cannot</em> be used here because the feature count and scale are different.<br><br>
        <strong>Steps to enable this tab:</strong><br>
        1. Train your store-level models using 30 features (see README for the column list).<br>
        2. Save them as <code>fnn_model.keras</code>, <code>gru_model.keras</code> etc. inside
           <code>saved_models/store/</code>.<br>
        3. Save the corresponding <code>feature_scaler.pkl</code> (fitted on 30 features) and
           <code>target_scaler.pkl</code> in the same folder.<br>
        4. Restart the app.
        </div>
        """, unsafe_allow_html=True)

        # Show the UI anyway so user can see what inputs are required
        st.markdown("---")
        st.markdown("#### 👀 Preview — Input fields (will be active once models are loaded)")

    # Model selector (store only)
    store_models_list = available_models("store")
    col_sm, col_sinfo = st.columns([2, 3])
    with col_sm:
        store_model = st.selectbox(
            "🤖 Model (store-wise)",
            store_models_list,
            disabled=(not store_scalers_ok or not store_models_available()),
            help="These models live in saved_models/store/ and expect 30 features.",
        )
    with col_sinfo:
        if is_quantile_model(store_model):
            st.info("📊 Quantile model — outputs Lower / Median / Upper bounds.")
        else:
            st.success("🎯 Point model — outputs a single sales estimate.")

    st.markdown("<hr class='fancy'>", unsafe_allow_html=True)

    # ── Store & Date inputs ───────────────────────────────────────────
    st.markdown('<p class="stitle">🏪 Store Selection & Date</p>', unsafe_allow_html=True)

    @st.cache_data(show_spinner=False)
    def get_store_ids():
        try:
            from preprocessing.store import get_all_store_ids
            return get_all_store_ids()
        except Exception:
            return list(range(1, 1116))

    store_ids = get_store_ids()

    col_sid, col_sdate = st.columns(2)
    with col_sid:
        store_id = st.selectbox(
            "Store ID",
            store_ids,
            help="Each store has its own sales history. Lag features are computed only from this store's data.",
        )
    with col_sdate:
        pred_date_store = st.date_input(
            "Prediction Date",
            value=date(2015, 7, 31),
            min_value=date(2013, 2, 1),
            max_value=date(2016, 12, 31),
            key="store_date",
        )

    st.markdown(
        '<div class="info-box">'
        'ℹ️ <strong>Store models use only past 60 days.</strong> '
        'Promo, SchoolHoliday, StoreType, Assortment, CompetitionDistance, Promo2 metadata, '
        'CompetitionOpenDays, Promo2RunningDays, and IsPromoMonth are loaded automatically '
        'from <code>data/store.csv</code>. Lag/rolling features are computed from '
        '<strong>this store\'s own history</strong> in <code>data/train.csv</code>.'
        '</div>',
        unsafe_allow_html=True,
    )

    # ── Store metadata preview ────────────────────────────────────────
    @st.cache_data(show_spinner=False)
    def get_store_meta(sid):
        path = os.path.join(os.path.dirname(__file__), "data", "store.csv")
        if os.path.exists(path):
            df = pd.read_csv(path)
            row = df[df["Store"] == sid]
            return row.iloc[0].to_dict() if not row.empty else {}
        return {}

    meta = get_store_meta(store_id)
    if meta:
        with st.expander(f"🔍 Store {store_id} — metadata from store.csv", expanded=False):
            items = [(k, v) for k, v in meta.items() if k != "Store"]
            cols  = st.columns(4)
            for i, (k, v) in enumerate(items):
                cols[i % 4].metric(k, "—" if pd.isna(v) else str(v))

    # ── Predict button ────────────────────────────────────────────────
    predict_disabled = not store_scalers_ok or not store_models_available()

    if st.button(
        "🚀 Predict Store Sales",
        type="primary",
        use_container_width=True,
        disabled=predict_disabled,
        key="btn_store",
    ):
        with st.spinner(f"Loading Store {store_id} history → engineering features → inference …"):
            try:
                from preprocessing.store import prepare_store_unified
                seq = prepare_store_unified(
                    store_id=store_id,
                    prediction_date=pred_date_store.strftime("%Y-%m-%d"),
                    feature_scaler=store_feature_scaler,
                    model_name=store_model,
                )
                result = predict("store", store_model, seq, store_target_scaler)
                render_result(result, store_model, f"Store {store_id}")
            except Exception:
                st.error("❌ Prediction failed. See traceback below.")
                show_traceback()

    # ── Multi-store comparison (bonus) ────────────────────────────────
    if store_scalers_ok and store_models_available():
        st.markdown("<hr class='fancy'>", unsafe_allow_html=True)
        st.markdown("### 🏆 Multi-Store Comparison")
        st.caption("Run the same prediction for multiple stores and rank results side by side.")

        compare_stores = st.multiselect(
            "Select stores to compare (max 10)",
            store_ids,
            default=store_ids[:5] if len(store_ids) >= 5 else store_ids,
            max_selections=10,
        )

        if st.button("📊 Compare Stores", use_container_width=True, key="btn_compare"):
            if not compare_stores:
                st.warning("Select at least one store.")
            else:
                from preprocessing.store import prepare_store_unified
                rows     = []
                progress = st.progress(0, text="Predicting …")
                for i, sid in enumerate(compare_stores):
                    try:
                        seq = prepare_store_unified(
                            store_id=sid,
                            prediction_date=pred_date_store.strftime("%Y-%m-%d"),
                            feature_scaler=store_feature_scaler,
                            model_name=store_model,
                        )
                        res = predict("store", store_model, seq, store_target_scaler)
                        val = res.get("prediction") or res.get("median", 0)
                    except Exception:
                        val = 0
                    rows.append({"Store": f"Store {sid}", "Predicted Sales": float(val)})
                    progress.progress((i + 1) / len(compare_stores), text=f"Done Store {sid}")
                progress.empty()

                df_cmp = pd.DataFrame(rows).sort_values("Predicted Sales", ascending=False)
                fig = px.bar(
                    df_cmp, x="Store", y="Predicted Sales",
                    color="Predicted Sales", color_continuous_scale="Viridis",
                    text=df_cmp["Predicted Sales"].apply(format_inr),
                    title=f"Store Sales Comparison — {pred_date_store}",
                )
                fig.update_traces(textposition="outside")
                fig.update_layout(height=400, coloraxis_showscale=False)
                st.plotly_chart(fig, use_container_width=True)

                st.dataframe(
                    df_cmp.assign(Formatted=lambda d: d["Predicted Sales"].apply(format_inr))
                          .rename(columns={"Formatted": "Predicted Sales (€)"})
                          .drop(columns=["Predicted Sales"]),
                    use_container_width=True, hide_index=True,
                )


# ──────────────────────────────────────────────────────────────────────
# TAB 3  —  DATA INSIGHTS
# ──────────────────────────────────────────────────────────────────────

with tab_insights:
    st.markdown("### 📈 Dataset Insights")
    st.caption("Explore patterns in the Rossmann training dataset.")

    @st.cache_data(show_spinner="Loading train.csv …")
    def load_insights_data():
        path = os.path.join(os.path.dirname(__file__), "data", "train.csv")
        if not os.path.exists(path):
            return None
        df = pd.read_csv(path, parse_dates=["Date"], low_memory=False)
        return df[df["Open"] == 1].copy()

    df_ins = load_insights_data()

    if df_ins is None:
        st.warning("⚠️ Place `train.csv` inside `data/` to see insights.")
    else:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Records",       f"{len(df_ins):,}")
        c2.metric("Stores",        df_ins["Store"].nunique())
        c3.metric("Date Range",    f"{df_ins['Date'].min().date()} → {df_ins['Date'].max().date()}")
        c4.metric("Avg Daily ∑",   format_inr(df_ins.groupby("Date")["Sales"].sum().mean()))

        st.markdown("<hr class='fancy'>", unsafe_allow_html=True)

        # Monthly aggregate sales
        monthly = (
            df_ins.groupby(df_ins["Date"].dt.to_period("M"))["Sales"]
            .sum().reset_index()
        )
        monthly["Date"] = monthly["Date"].astype(str)
        fig_m = px.line(monthly, x="Date", y="Sales",
                        title="📅 Monthly Total Sales (all stores)", markers=True)
        fig_m.update_traces(line_color="#667eea", line_width=2)
        fig_m.update_layout(height=300)
        st.plotly_chart(fig_m, use_container_width=True)

        col_l, col_r = st.columns(2)

        with col_l:
            dow_map = {1:"Mon",2:"Tue",3:"Wed",4:"Thu",5:"Fri",6:"Sat",7:"Sun"}
            df_dow = (
                df_ins.groupby("DayOfWeek")["Sales"].mean()
                .reset_index().assign(Day=lambda d: d["DayOfWeek"].map(dow_map))
            )
            fig_d = px.bar(df_dow, x="Day", y="Sales",
                           title="📆 Avg Sales by Day of Week",
                           color="Sales", color_continuous_scale="Blues")
            fig_d.update_layout(height=300, coloraxis_showscale=False)
            st.plotly_chart(fig_d, use_container_width=True)

        with col_r:
            df_p = (
                df_ins.groupby("Promo")["Sales"].mean()
                .reset_index()
                .assign(Promo=lambda d: d["Promo"].map({0:"No Promo",1:"Promo Active"}))
            )
            fig_p = px.pie(df_p, names="Promo", values="Sales",
                           title="🎟️ Avg Sales: Promo vs No Promo",
                           color_discrete_sequence=["#667eea","#43e97b"])
            fig_p.update_layout(height=300)
            st.plotly_chart(fig_p, use_container_width=True)

        # Top 20 stores
        top20 = (
            df_ins.groupby("Store")["Sales"].mean()
            .nlargest(20).reset_index().rename(columns={"Sales":"Avg Daily Sales"})
        )
        fig_t = px.bar(top20, x="Store", y="Avg Daily Sales",
                       title="🏆 Top 20 Stores by Avg Daily Sales",
                       color="Avg Daily Sales", color_continuous_scale="Viridis")
        fig_t.update_layout(height=320, coloraxis_showscale=False)
        st.plotly_chart(fig_t, use_container_width=True)

        # Per-store distribution
        store_avg = df_ins.groupby("Store")["Sales"].mean().reset_index()
        fig_h = px.histogram(store_avg, x="Sales", nbins=40,
                             title="📊 Distribution of Store Avg Daily Sales",
                             color_discrete_sequence=["#667eea"])
        fig_h.update_layout(height=280)
        st.plotly_chart(fig_h, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════
# FOOTER
# ══════════════════════════════════════════════════════════════════════

st.markdown("<hr class='fancy'>", unsafe_allow_html=True)
st.markdown(
    "<center><small>"
    "Rossmann Sales Predictor &nbsp;·&nbsp; "
    "Aggregate: 22 features &nbsp;|&nbsp; Store-wise: 30 features &nbsp;·&nbsp; "
    "FNN · GRU · RNN · N-HiTS · Quantile variants"
    "</small></center>",
    unsafe_allow_html=True,
)
