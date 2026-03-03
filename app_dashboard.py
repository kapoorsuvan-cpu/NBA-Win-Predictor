# app_dashboard.py
"""
Complete Streamlit dashboard for your NBA Win% model.

Place this next to main.py and your csv/ folder. Then run:
    pip install -r requirements.txt
    streamlit run app_dashboard.py

Features:
 - load CSVs from repo csv/ or via upload
 - train / retrain model (cached)
 - show model coefficients, RMSE/R^2 (training)
 - feature importance bar chart (absolute coeff)
 - historical scatter (prev_win_pct vs true_win_pct)
 - per-team historical view
 - interactive predictor: previous wins -> predicted win% and wins
 - download coefficients and training CSV
"""

from typing import Optional, Tuple
import io

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

# Import functions/constants from your project
from main import (
    train_from_csvs,
    _load_csv,
    _validate_inputs,
    build_training_dataframe,
    predict_team_win_pct,
    predict_team_wins,
    MODEL,
    CSV_PREV_WIN,
    CSV_COACH,
    CSV_TALENT,
    CSV_TRUE_WIN,
)

st.set_page_config(page_title="NBA Win% Dashboard", layout="wide")
st.title("NBA Win% Model — Dashboard")

# -------------------------
# Sidebar: data & controls
# -------------------------
st.sidebar.header("Data & Model")
use_uploads = st.sidebar.checkbox("Use uploaded CSVs (override repo csv/)", value=False)

if use_uploads:
    upload_prev = st.sidebar.file_uploader("prev_win_pct.csv", type="csv")
    upload_coach = st.sidebar.file_uploader("coach_continuity.csv", type="csv")
    upload_talent = st.sidebar.file_uploader("roster_talent.csv", type="csv")
    upload_true = st.sidebar.file_uploader("true_win_pct.csv", type="csv")
else:
    upload_prev = upload_coach = upload_talent = upload_true = None

retrain_btn = st.sidebar.button("Train / Retrain model")
st.sidebar.markdown("---")
st.sidebar.write("Tip: upload all 4 CSVs to test alternate datasets.")

# -------------------------
# Helpers to load data
# -------------------------
@st.cache_data
def _read_buffer(buffer: io.BytesIO) -> pd.DataFrame:
    return pd.read_csv(buffer, index_col=0, encoding="utf-8-sig")

def load_data_or_repo() -> Optional[Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]]:
    """Load data either from uploaded files (if enabled) or from repo CSVs."""
    if use_uploads:
        if not (upload_prev and upload_coach and upload_talent and upload_true):
            st.sidebar.warning("Please upload all 4 CSVs to use uploaded data.")
            return None
        try:
            prev = _read_buffer(upload_prev)
            coach = _read_buffer(upload_coach)
            talent = _read_buffer(upload_talent)
            true = _read_buffer(upload_true)
        except Exception as e:
            st.sidebar.error(f"Error reading uploaded CSVs: {e}")
            return None
    else:
        try:
            prev = _load_csv(CSV_PREV_WIN)
            coach = _load_csv(CSV_COACH)
            talent = _load_csv(CSV_TALENT)
            true = _load_csv(CSV_TRUE_WIN)
        except Exception as e:
            st.sidebar.error(f"Error loading repo CSVs: {e}")
            return None

    # normalize column names and index done in _load_csv for repo; ensure uploaded gets same treatment
    prev.columns = [str(c).strip() for c in prev.columns]
    coach.columns = [str(c).strip() for c in coach.columns]
    talent.columns = [str(c).strip() for c in talent.columns]
    true.columns = [str(c).strip() for c in true.columns]

    prev.index = prev.index.astype(str).str.strip()
    coach.index = coach.index.astype(str).str.strip()
    talent.index = talent.index.astype(str).str.strip()
    true.index = true.index.astype(str).str.strip()

    return prev, coach, talent, true

data = load_data_or_repo()
if data is None:
    st.info("Upload CSVs in the sidebar or place CSVs in csv/ directory (prev_win_pct.csv, coach_continuity.csv, roster_talent.csv, true_win_pct.csv).")
    st.stop()

prev_df, coach_df, talent_df, true_df = data

# validate and build training DataFrame
try:
    _validate_inputs(prev_df, coach_df, talent_df, true_df)
    train_df = build_training_dataframe(prev_df, coach_df, talent_df, true_df)
except Exception as e:
    st.error(f"Error preparing training data: {e}")
    st.stop()

# -------------------------
# Train model (cached)
# -------------------------
@st.cache_resource
def _ensure_trained_model():
    """Train model by calling main.train_from_csvs(). Returns the trained MODEL object from main."""
    train_from_csvs()
    from main import MODEL as m
    return m

# retrain if user requests - clear cache_resource isn't exposed; calling to ensure trained (Streamlit will cache)
if retrain_btn:
    st.sidebar.info("Training model... (this may take a moment)")
model = _ensure_trained_model()
if model is None:
    st.error("Model not available after training.")
    st.stop()

# -------------------------
# Model diagnostics
# -------------------------
st.header("Model diagnostics & data snapshot")
col1, col2 = st.columns([1.2, 1])

with col1:
    st.subheader("Training data (sample)")
    st.dataframe(train_df.head(8))

with col2:
    st.subheader("Coefficients")
    coef_index = ["prev_win_pct", "coach_continuity", "roster_talent"]
    coefs = pd.Series(model.coef_, index=coef_index)
    intercept = float(model.intercept_)
    coef_df = pd.DataFrame({"coefficient": coefs, "abs": coefs.abs()})
    st.table(coef_df)

    # quick train diagnostics on full data (not holdout)
    X = train_df[["prev_win_pct", "coach_continuity", "roster_talent"]]
    y = train_df["true_win_pct"]
    y_pred = model.predict(X)
    rmse = float(np.sqrt(((y - y_pred) ** 2).mean()))
    r2 = float(np.corrcoef(y, y_pred)[0, 1] ** 2) if len(y) > 1 else float("nan")
    st.metric("RMSE (train)", f"{rmse:.4f}")
    st.metric("R² (train approx.)", f"{r2:.4f}")
    st.markdown(f"**Intercept:** {intercept:.6f}")

# -------------------------
# Visualizations
# -------------------------
st.header("Visualizations")

# Feature importance bar (absolute value, sorted)
st.subheader("Feature importance (by |coefficient|)")
abs_coefs = coefs.abs()
feat_df = (
    pd.DataFrame({
        "feature": abs_coefs.index,
        "abs_coef": abs_coefs.values,
        "raw_coef": coefs.values
    })
    .sort_values("abs_coef", ascending=True)
)

fig_imp = px.bar(
    feat_df,
    x="abs_coef",
    y="feature",
    orientation="h",
    text=feat_df["raw_coef"].round(4),
    labels={"abs_coef": "Absolute coefficient"},
    title="Feature importance (absolute coefficient)"
)
fig_imp.update_layout(height=320, margin=dict(l=40, r=10, t=40, b=10))
st.plotly_chart(fig_imp, use_container_width=True)

# Historical scatter
st.subheader("Historical: Previous season win% vs True win%")
corr = np.corrcoef(train_df["prev_win_pct"], train_df["true_win_pct"])[0, 1]
fig_scatter = px.scatter(
    train_df,
    x="prev_win_pct",
    y="true_win_pct",
    hover_data=["team", "season"],
    labels={"prev_win_pct": "Previous season win%", "true_win_pct": "True win%"},
    title=f"Prev win% vs True win%  (corr={corr:.3f})"
)
st.plotly_chart(fig_scatter, use_container_width=True)

# -------------------------
# Per-team historical inspector
# -------------------------
st.header("Team history inspector")
teams = sorted(train_df["team"].unique())
team_choice = st.selectbox("Choose a team to inspect", options=teams, index=teams.index(teams[0]) if teams else 0)

if team_choice:
    team_rows = train_df[train_df["team"] == team_choice].sort_values("season")
    st.subheader(f"{team_choice} — historical seasons")
    st.table(team_rows[["season", "prev_win_pct", "coach_continuity", "roster_talent", "true_win_pct"]])

    # small line chart of true vs prev
    fig_team = px.line(
        team_rows.melt(id_vars="season", value_vars=["prev_win_pct", "true_win_pct"], var_name="type", value_name="win_pct"),
        x="season", y="win_pct", color="type", markers=True,
        title=f"{team_choice}: previous vs true win% by season"
    )
    st.plotly_chart(fig_team, use_container_width=True)

# -------------------------
# Interactive predictor
# -------------------------
st.header("Interactive predictor (wins input → prediction)")

with st.form("predict_form", clear_on_submit=False):
    c1, c2, c3 = st.columns(3)
    with c1:
        prev_wins = st.number_input("Previous season wins (0–82)", min_value=0, max_value=82, value=41, step=1)
    with c2:
        roster_talent_input = st.number_input("Roster talent (numeric)", min_value=-10.0, max_value=20.0, value=0.0, step=1.0)
    with c3:
        coach_cont_input = st.selectbox("Coach continuity", options=[1, 0], index=0, format_func=lambda x: "Yes" if x==1 else "No")

    submit = st.form_submit_button("Predict")

if submit:
    prev_win_pct_val = float(prev_wins) / 82.0
    try:
        pred_pct_val = predict_team_win_pct(prev_win_pct_val, int(coach_cont_input), float(roster_talent_input))
        pred_wins_val = predict_team_wins(prev_win_pct_val, int(coach_cont_input), float(roster_talent_input))
    except Exception as e:
        st.error(f"Prediction failed: {e}")
    else:
        st.success(f"Predicted win% = {pred_pct_val:.3f} → Predicted wins (82-game) = {pred_wins_val}")
        gauge = px.bar(x=[pred_pct_val, 1 - pred_pct_val], y=["Predicted", ""], orientation="h", labels={"x": "Win %"}, title="Predicted Win%")
        gauge.update_layout(showlegend=False, height=140, margin=dict(t=10, b=10))
        st.plotly_chart(gauge, use_container_width=True)

        # overlay on historical scatter
        fig_overlay = fig_scatter
        fig_overlay.add_scatter(x=[prev_win_pct_val], y=[pred_pct_val], mode="markers", marker=dict(size=14, symbol="x", color="red"), name="Prediction")
        st.plotly_chart(fig_overlay, use_container_width=True)

# -------------------------
# Export & download
# -------------------------
st.header("Export & downloads")
if st.button("Download coefficients CSV"):
    out = pd.DataFrame({"feature": coefs.index, "coefficient": coefs.values})
    st.download_button("Download coefficients CSV", data=out.to_csv(index=False).encode("utf-8"), file_name="coefficients.csv", mime="text/csv")

if st.button("Download training data CSV"):
    st.download_button("Download training CSV", data=train_df.to_csv(index=False).encode("utf-8"), file_name="training_data.csv", mime="text/csv")

st.markdown("---")
st.caption("Notes: predictions assume an 82-game season. Roster talent and coach continuity should match your CSV encodings.")
