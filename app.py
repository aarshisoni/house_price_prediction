"""
app.py
------
Professional Streamlit frontend for the House Price Prediction System.
Loads the pre-trained pipeline from model/house_price_model.pkl.
Does NOT retrain the model.

Run:
    streamlit run app.py
"""

import os
import joblib
import numpy  as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

# ── Page config (MUST be first Streamlit call) ────────────────────────────────
st.set_page_config(
    page_title="House Price Prediction System",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Relative paths (works on any computer) ───────────────────────────────────
BASE_DIR        = os.path.dirname(os.path.abspath(__file__))
DATA_PATH       = os.path.join(BASE_DIR, "housing_price.csv")
MODEL_PATH      = os.path.join(BASE_DIR, "model", "house_price_model.pkl")
EVAL_PATH       = os.path.join(BASE_DIR, "model", "eval_results.csv")
BEST_MODEL_PATH = os.path.join(BASE_DIR, "model", "best_model.txt")

LOCATIONS = ["Downtown", "City_Center", "Suburban", "Rural"]

# ── Global CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── General ── */
[data-testid="stAppViewContainer"] { background: #f8fafc; }
[data-testid="stHeader"]           { background: transparent; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #1e3a5f 0%, #2d5a8e 100%);
    border-right: none;
}
[data-testid="stSidebar"] * { color: #e8f0fe !important; }
[data-testid="stSidebar"] .stSelectbox label,
[data-testid="stSidebar"] .stSlider label,
[data-testid="stSidebar"] .stNumberInput label { color: #c8d8f0 !important; font-size: 13px !important; }
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p { color: #a8c4e0 !important; }

/* ── Metric cards ── */
[data-testid="metric-container"] {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 10px;
    padding: 16px 20px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.06);
}

/* ── Prediction card ── */
.pred-card {
    background: linear-gradient(135deg, #1e3a5f 0%, #2d5a8e 60%, #3b6eb5 100%);
    border-radius: 16px;
    padding: 36px 40px;
    text-align: center;
    color: white;
    box-shadow: 0 8px 24px rgba(30,58,95,0.25);
    margin: 8px 0 16px 0;
}
.pred-card .label {
    font-size: 14px;
    letter-spacing: 2px;
    text-transform: uppercase;
    opacity: 0.8;
    margin-bottom: 10px;
}
.pred-card .price {
    font-size: 3.2rem;
    font-weight: 800;
    letter-spacing: -1px;
    line-height: 1.1;
    margin: 0;
}
.pred-card .details {
    font-size: 13px;
    opacity: 0.75;
    margin-top: 12px;
}

/* ── Section headers ── */
.section-header {
    font-size: 1.25rem;
    font-weight: 700;
    color: #1e3a5f;
    padding-bottom: 6px;
    border-bottom: 2px solid #3b6eb5;
    margin-bottom: 18px;
}

/* ── Info box ── */
.info-box {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-left: 4px solid #3b6eb5;
    border-radius: 8px;
    padding: 16px 20px;
    font-size: 14px;
    color: #374151;
    line-height: 1.7;
}

/* ── Model badge ── */
.model-badge {
    display: inline-block;
    background: #e8f5e9;
    color: #1b5e20;
    border: 1px solid #a5d6a7;
    border-radius: 20px;
    padding: 4px 14px;
    font-size: 13px;
    font-weight: 600;
}

/* ── Footer ── */
.footer {
    text-align: center;
    color: #9ca3af;
    font-size: 12px;
    padding: 24px 0 8px 0;
    border-top: 1px solid #e5e7eb;
    margin-top: 32px;
}

/* ── Tabs ── */
[data-testid="stTab"] { font-weight: 500; }
</style>
""", unsafe_allow_html=True)


# ── Cached loaders ────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        return None
    return joblib.load(MODEL_PATH)

@st.cache_data
def load_data():
    if not os.path.exists(DATA_PATH):
        return None
    return pd.read_csv(DATA_PATH)

@st.cache_data
def load_eval():
    if not os.path.exists(EVAL_PATH):
        return None, ""
    eval_df = pd.read_csv(EVAL_PATH)
    best = ""
    if os.path.exists(BEST_MODEL_PATH):
        with open(BEST_MODEL_PATH) as f:
            best = f.read().strip()
    return eval_df, best


# ── Indian rupee formatter ────────────────────────────────────────────────────
def format_inr(amount: float) -> str:
    amount = int(round(amount))
    s = str(amount)
    if len(s) <= 3:
        return s
    result = s[-3:]
    s = s[:-3]
    while s:
        result = s[-2:] + "," + result
        s = s[:-2]
    return result.lstrip(",")


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 🏠 House Price Predictor")
    st.markdown("---")

    # ── Model info card ──────────────────────────────────────────────────────
    eval_df, best_model_name = load_eval()

    st.markdown("### 📊 Model Information")
    df_side = load_data()

    if eval_df is not None and best_model_name:
        best_row = eval_df[eval_df["Model"] == best_model_name].iloc[0]
        n_records = len(df_side) if df_side is not None else "N/A"
        n_features = (df_side.shape[1] - 1) if df_side is not None else "N/A"

        st.markdown(f"**Active Model**")
        st.markdown(f'<span class="model-badge">✓ {best_model_name}</span>', unsafe_allow_html=True)
        st.markdown("")

        col_a, col_b = st.columns(2)
        col_a.metric("Training Records", f"{n_records}")
        col_b.metric("Features", f"{n_features}")

        col_c, col_d = st.columns(2)
        col_c.metric("R² Score", f"{best_row['R2']:.4f}")
        col_d.metric("MAE", f"₹{best_row['MAE']/1e5:.2f}L")

        st.metric("RMSE", f"₹{best_row['RMSE']/1e5:.2f}L")
    else:
        st.info("Run `python train_model.py` to load model info.")

    st.markdown("---")

    # ── Input form ───────────────────────────────────────────────────────────
    st.markdown("### 🏡 Enter House Details")

    area_sqft = st.slider(
        "Area (sq ft)", min_value=400, max_value=5000, value=1500, step=50,
        help="Total floor area of the house in square feet"
    )
    col1, col2 = st.columns(2)
    with col1:
        bedrooms  = st.selectbox("Bedrooms",  options=[1,2,3,4,5], index=2)
        stories   = st.selectbox("Stories",   options=[1,2,3,4],   index=0)
        age_years = st.number_input("Age (years)", min_value=0, max_value=60, value=10, step=1)
    with col2:
        bathrooms = st.selectbox("Bathrooms", options=[1,2,3,4],   index=1)
        parking   = st.selectbox("Parking",   options=[0,1,2,3],   index=1)

    location = st.selectbox(
        "Location",
        options=LOCATIONS,
        index=0,
        help="Neighbourhood category of the property"
    )

    st.markdown("")
    predict_btn = st.button(
        "🔮  Predict House Price",
        use_container_width=True,
        type="primary",
    )


# ══════════════════════════════════════════════════════════════════════════════
# MAIN PAGE
# ══════════════════════════════════════════════════════════════════════════════

# ── Page header ───────────────────────────────────────────────────────────────
st.markdown(
    "<h1 style='color:#1e3a5f; margin-bottom:4px;'>🏠 House Price Prediction System</h1>",
    unsafe_allow_html=True,
)
st.markdown(
    "<p style='color:#6b7280; font-size:16px; margin-top:0;'>"
    "Predict the estimated price of a house using Machine Learning</p>",
    unsafe_allow_html=True,
)
st.markdown("---")

# ── Two-column layout: Prediction | Dataset Insights ─────────────────────────
main_col, right_col = st.columns([3, 2], gap="large")

with main_col:
    st.markdown('<p class="section-header">🎯 Prediction</p>', unsafe_allow_html=True)

    model = load_model()

    if model is None:
        st.warning(
            "Model not found at `model/house_price_model.pkl`.  \n"
            "Please run `python train_model.py` first."
        )
    else:
        if predict_btn:
            # ── Build input dataframe ─────────────────────────────────────────
            input_df = pd.DataFrame([{
                "area_sqft": int(area_sqft),
                "bedrooms":  int(bedrooms),
                "bathrooms": int(bathrooms),
                "stories":   int(stories),
                "parking":   int(parking),
                "age_years": int(age_years),
                "location":  location,
            }])

            predicted_price = max(100_000.0, float(model.predict(input_df)[0]))

            # ── Prediction card ───────────────────────────────────────────────
            st.markdown(
                f"""
                <div class="pred-card">
                    <div class="label">Estimated House Price</div>
                    <p class="price">&#8377; {format_inr(predicted_price)}</p>
                    <div class="details">
                        {area_sqft:,} sq ft &nbsp;|&nbsp;
                        {bedrooms} Bed &nbsp;|&nbsp;
                        {bathrooms} Bath &nbsp;|&nbsp;
                        {stories} Floor(s) &nbsp;|&nbsp;
                        {location}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.caption("Estimated using the trained machine learning model.")

            # ── Input summary expander ────────────────────────────────────────
            with st.expander("View input summary"):
                summary_df = pd.DataFrame({
                    "Feature": ["Area (sq ft)", "Bedrooms", "Bathrooms",
                                "Stories", "Parking", "Age (years)", "Location"],
                    "Value":   [f"{area_sqft:,} sq ft", bedrooms, bathrooms,
                                stories, parking, f"{age_years} yrs", location],
                })
                st.dataframe(summary_df, hide_index=True, use_container_width=True)

        else:
            # Placeholder before prediction
            st.markdown(
                """
                <div style="
                    background:#f1f5f9;
                    border:2px dashed #cbd5e1;
                    border-radius:16px;
                    padding:48px 32px;
                    text-align:center;
                    color:#94a3b8;
                ">
                    <div style="font-size:48px; margin-bottom:12px;">🏡</div>
                    <div style="font-size:16px; font-weight:600;">Ready to Predict</div>
                    <div style="font-size:13px; margin-top:6px;">
                        Fill in the house details in the sidebar<br>
                        and click <strong>Predict House Price</strong>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

with right_col:
    # ── Dataset insights ──────────────────────────────────────────────────────
    st.markdown('<p class="section-header">📈 Dataset Insights</p>', unsafe_allow_html=True)
    df = load_data()

    if df is None:
        st.warning("Dataset not found. Run `python generate_dataset.py` first.")
    else:
        r1c1, r1c2 = st.columns(2)
        r1c1.metric("Total Houses",  f"{len(df):,}")
        r1c2.metric("Features",      f"{df.shape[1] - 1}")

        r2c1, r2c2 = st.columns(2)
        r2c1.metric("Avg Price",     f"₹{df['price'].mean()/1e5:.1f}L")
        r2c2.metric("Median Price",  f"₹{df['price'].median()/1e5:.1f}L")

        r3c1, r3c2 = st.columns(2)
        r3c1.metric("Min Price",     f"₹{df['price'].min()/1e5:.1f}L")
        r3c2.metric("Max Price",     f"₹{df['price'].max()/1e5:.1f}L")

        st.markdown("")

        # mini location breakdown
        loc_counts = df["location"].value_counts()
        st.markdown("**Houses by Location**")
        for loc, cnt in loc_counts.items():
            pct = cnt / len(df) * 100
            st.markdown(
                f"<div style='display:flex;justify-content:space-between;"
                f"font-size:13px;padding:3px 0;'>"
                f"<span>{loc}</span>"
                f"<span style='color:#6b7280;'>{cnt} &nbsp;({pct:.0f}%)</span></div>",
                unsafe_allow_html=True,
            )
            st.progress(int(pct))

st.markdown("---")

# ══════════════════════════════════════════════════════════════════════════════
# VISUALIZATIONS
# ══════════════════════════════════════════════════════════════════════════════
st.markdown('<p class="section-header">📊 Data Visualization</p>', unsafe_allow_html=True)

df = load_data()
if df is not None:
    tab1, tab2, tab3 = st.tabs([
        "  📉  Price Distribution  ",
        "  📐  Area vs Price  ",
        "  📍  Avg Price by Location  ",
    ])

    plt.rcParams.update({
        "axes.spines.top":    False,
        "axes.spines.right":  False,
        "axes.grid":          True,
        "grid.alpha":         0.35,
        "grid.linestyle":     "--",
        "font.size":          11,
    })

    with tab1:
        fig, ax = plt.subplots(figsize=(9, 4))
        fig.patch.set_facecolor("#ffffff")
        ax.set_facecolor("#fafbfc")
        sns.histplot(df["price"], bins=25, kde=True,
                     color="#2d5a8e", edgecolor="white", linewidth=0.5, ax=ax)
        ax.lines[0].set_color("#e74c3c")
        ax.lines[0].set_linewidth(2)
        ax.set_title("House Price Distribution", fontsize=14, fontweight="bold", pad=12)
        ax.set_xlabel("Price", fontsize=11)
        ax.set_ylabel("Number of Houses", fontsize=11)
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"₹{x/1e5:.0f}L"))
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
        st.caption("Distribution of house prices in the synthetic dataset. The red curve is the KDE (density estimate).")

    with tab2:
        fig, ax = plt.subplots(figsize=(9, 4))
        fig.patch.set_facecolor("#ffffff")
        ax.set_facecolor("#fafbfc")
        sc = ax.scatter(
            df["area_sqft"], df["price"],
            c=df["price"], cmap="YlOrRd",
            alpha=0.75, edgecolors="#999", linewidths=0.3, s=55
        )
        cbar = plt.colorbar(sc, ax=ax)
        cbar.set_label("Price (₹)", fontsize=10)
        cbar.ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"₹{x/1e5:.0f}L"))
        ax.set_title("Area vs Price", fontsize=14, fontweight="bold", pad=12)
        ax.set_xlabel("Area (sq ft)", fontsize=11)
        ax.set_ylabel("Price", fontsize=11)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"₹{x/1e5:.0f}L"))
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
        st.caption("Larger houses generally command higher prices. Colour intensity also reflects price level.")

    with tab3:
        avg_price = (
            df.groupby("location")["price"]
              .mean()
              .sort_values(ascending=False)
              .reset_index()
        )
        fig, ax = plt.subplots(figsize=(8, 4))
        fig.patch.set_facecolor("#ffffff")
        ax.set_facecolor("#fafbfc")
        palette = ["#1e3a5f", "#2d5a8e", "#3b6eb5", "#6b9fd4"]
        bars = ax.barh(avg_price["location"], avg_price["price"],
                       color=palette, edgecolor="white", linewidth=0.5)
        for bar, val in zip(bars, avg_price["price"]):
            ax.text(
                bar.get_width() + 20_000, bar.get_y() + bar.get_height() / 2,
                f"₹{val/1e5:.1f}L",
                va="center", ha="left", fontsize=10, color="#374151"
            )
        ax.set_title("Average Price by Location", fontsize=14, fontweight="bold", pad=12)
        ax.set_xlabel("Average Price", fontsize=11)
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"₹{x/1e5:.0f}L"))
        ax.invert_yaxis()
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
        st.caption("City_Center and Downtown locations carry higher price premiums than Suburban or Rural areas.")

st.markdown("---")

# ══════════════════════════════════════════════════════════════════════════════
# MODEL PERFORMANCE
# ══════════════════════════════════════════════════════════════════════════════
st.markdown('<p class="section-header">🤖 Model Performance Comparison</p>', unsafe_allow_html=True)

eval_df, best_model_name = load_eval()

if eval_df is None:
    st.info("Run `python train_model.py` to generate evaluation results.")
else:
    perf_cols = st.columns(3)
    colors    = ["#e8f5e9", "#fff3e0", "#e3f2fd"]
    icons     = ["🥇", "🥈", "🥉"]
    sorted_eval = eval_df.sort_values("R2", ascending=False).reset_index(drop=True)

    for i, (_, row) in enumerate(sorted_eval.iterrows()):
        badge = " ✓ Selected" if row["Model"] == best_model_name else ""
        with perf_cols[i]:
            st.markdown(
                f"""
                <div style="
                    background:{colors[i]};
                    border-radius:12px;
                    padding:20px;
                    text-align:center;
                    border:1px solid #e2e8f0;
                ">
                    <div style="font-size:28px;">{icons[i]}</div>
                    <div style="font-weight:700;font-size:15px;color:#1e3a5f;margin:6px 0;">
                        {row['Model']}{badge}
                    </div>
                    <div style="font-size:13px;color:#4b5563;margin-top:8px;">
                        <b>R²</b> &nbsp;{row['R2']:.4f}<br>
                        <b>MAE</b> &nbsp;₹{row['MAE']/1e5:.2f}L<br>
                        <b>RMSE</b> &nbsp;₹{row['RMSE']/1e5:.2f}L
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown("")

    # Detailed table in expander
    with st.expander("View full metrics table"):
        disp = eval_df.copy()
        disp["MAE"]  = disp["MAE"].apply(lambda x: f"₹ {x:,.0f}")
        disp["RMSE"] = disp["RMSE"].apply(lambda x: f"₹ {x:,.0f}")
        disp["R2"]   = disp["R2"].apply(lambda x: f"{x:.4f}")
        disp = disp.rename(columns={"R2": "R² Score"})

        def highlight_best(row):
            bg = "background-color:#d4edda;" if row["Model"] == best_model_name else ""
            return [bg] * len(row)

        st.dataframe(
            disp.style.apply(highlight_best, axis=1),
            hide_index=True,
            use_container_width=True,
        )
        st.caption(
            "Higher R² is better (max 1.0).  "
            "Lower MAE and RMSE indicate less prediction error.  "
            "The highlighted row is the saved model."
        )

st.markdown("---")

# ══════════════════════════════════════════════════════════════════════════════
# DATASET PREVIEW
# ══════════════════════════════════════════════════════════════════════════════
df = load_data()
if df is not None:
    with st.expander("🗂️  Dataset Preview  (first 10 rows)"):
        st.dataframe(df.head(10), use_container_width=True, hide_index=True)
        st.caption(
            f"Total: {len(df)} records  |  "
            "This is a synthetic dataset created for educational purposes only."
        )

st.markdown("---")

# ══════════════════════════════════════════════════════════════════════════════
# ABOUT
# ══════════════════════════════════════════════════════════════════════════════
st.markdown('<p class="section-header">ℹ️ About This Project</p>', unsafe_allow_html=True)

st.markdown(
    """
    <div class="info-box">
    This project demonstrates an end-to-end Machine Learning workflow for house price
    prediction using a synthetic educational dataset. The system preprocesses house
    information, trains regression models, evaluates their performance, and provides
    predictions through an interactive web interface.
    <br><br>
    <strong>Workflow:</strong>
    Data Generation &rarr; Preprocessing (StandardScaler + OneHotEncoder) &rarr;
    Model Training (Linear Regression, Decision Tree, Random Forest) &rarr;
    Evaluation (MAE, RMSE, R&sup2;) &rarr; Best Model Selection &rarr; Streamlit Deployment
    <br><br>
    <em>&#9888; The dataset is entirely synthetic and does <strong>not</strong> represent
    real housing-market data or actual property prices.</em>
    </div>
    """,
    unsafe_allow_html=True,
)

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown(
    '<div class="footer">House Price Prediction System &nbsp;|&nbsp; Machine Learning Project</div>',
    unsafe_allow_html=True,
)
