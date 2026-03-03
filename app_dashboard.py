# app_dashboard.py
"""
Streamlit dashboard (auto-detects CSV location: csv/ or repo root).

Place this next to main.py and your CSVs (either in csv/ or in repo root).
Then:
    git add app_dashboard.py
    git commit -m "Update dashboard with all-teams predictions table"
    git push
    (On Streamlit Cloud) Manage App -> Reboot
"""

from typing import Optional, Tuple
import io
from pathlib import Path

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

# Import functions/constants from your project
from main import (
    train_from_csvs,
    _validate_inputs,
    build_training_dataframe,
    predict_team_win_pct,
    predict_team_wins,
    MODEL,
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
st.sidebar.write("Tip: upload all 4 CSVs to test alternate datasets or place CSVs in csv/ folder or repo root.")

# -------------------------
# Helper: read buffer
# -------------------------
@st.cache_data
def _read_buffer(buffer: io.BytesIO) -> pd.DataFrame:
    return pd.read_csv(buffer, index_col=0, encoding="utf-8-sig")

# -------------------------
# Helper: robust repo loader
# -------------------------
REQUIRED_FILES = {
    "prev": "prev_win_pct.csv",
    "coach": "coach_continuity.csv",
    "talent": "roster_talent.csv",
    "true": "true_win_pct.csv",
}

def _try_load_path(path: Path) -> Optional[pd.DataFrame]:
    """Return DataFrame if file exists and loads, otherwise None."""
    try:
        if not path.exists():
            return None
        df = pd.read_csv(path, index_col=0, encoding="utf-8-sig")
        # normalize
        df.columns = [str(c).strip() for c in df.columns]
        df.index = df.index.astype(str).str.strip()
        return df
    except Exception:
        return None

def load_data_or_repo() -> Optional[Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]]:
    """
    Load CSVs in this order:
      1) If use_uploads: load uploaded files (requires all 4)
      2) If csv/ directory exists: load csv/<file>.csv
      3) Else try repo root: <file>.csv
    """
    # 1) uploads
    if use_uploads:
        if not (upload_prev and upload_coach and upload_talent and upload_true):
            st.sidebar.warning("Please upload all 4 CSVs to use uploaded data.")
            return None
        try:
            prev = _read_buffer(upload_prev)
            coach = _read_buffer(upload_coach)
            talent = _read_buffer(upload_talent)
            true = _read_buffer(upload_true)
            return prev, coach, talent, true
        except Exception as e:
            st.sidebar.error(f"Error reading uploaded CSVs: {e}")
            return None

    # 2) repo files
    base = Path(__file__).resolve().parent
    csv_dir = base / "csv"

    # prefer csv/ directory if it exists and contains the files
    candidate_dirs = []
    if csv_dir.exists():
        candidate_dirs.append(csv_dir)
    # always try root as fallback
    candidate_dirs.append(base)

    found = {}
    for d in candidate_dirs:
        ok = True
        for key, fname in REQUIRED_FILES.items():
            df = _try_load_path(d / fname)
            if df is None:
                ok = False
                break
            found[key] = df
        if ok:
            return found["prev"], found["coach"], found["talent"], found["true"]

    # If we reach here, nothing loaded successfully
    # Provide helpful error message listing what paths were tried
    tried = [str((base / "csv") / fname) for fname in REQUIRED_FILES.values()] + [str((base / fname)) for fname in REQUIRED_FILES.values()]
    st.error(
        "Could not find all required CSV files. I tried these locations:\n\n"
        + "\n".join(tried)
        + "\n\nPlace the files either in a folder named `csv/` or directly in the repo root, or upload them via the sidebar."
    )
    return None

# -------------------------
# Load data
# -------------------------
data = load_data_or_repo()
if data is None:
    st.info("Upload CSVs in the sidebar or place CSVs in csv/ or repo root (prev_win_pct.csv, coach_continuity.csv, roster_talent.csv, true_win_pct.csv).")
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
# All teams predictions table
# -------------------------
st.header("Predicted win totals for all teams")

# Determine which columns to use (most recent available)
# Use last column in prev_df for previous season; last column in coach_df and talent_df for inputs
prev_col = prev_df.columns[-1]
coach_col = coach_df.columns[-1]
talent_col = talent_df.columns[-1]

teams = sorted(prev_df.index.tolist())

rows = []
for team in teams:
    try:
        prev_win_pct = float(prev_df.loc[team, prev_col])
    except Exception:
        # if missing or invalid, skip
        prev_win_pct = np.nan
    try:
        coach_val = int(coach_df.loc[team, coach_col])
    except Exception:
        coach_val = 0
    try:
        talent_val = float(talent_df.loc[team, talent_col])
    except Exception:
        talent_val = np.nan

    if np.isnan(prev_win_pct) or np.isnan(talent_val):
        pred_pct = np.nan
        pred_wins = np.nan
    else:
        pred_pct = predict_team_win_pct(prev_win_pct, int(coach_val), float(talent_val))
        pred_wins = predict_team_wins(prev_win_pct, int(coach_val), float(talent_val))

    rows.append({
        "team": team,
        "prev_col_used": prev_col,
        "prev_win_pct": prev_win_pct,
        "coach_col_used": coach_col,
        "coach_continuity": coach_val,
        "talent_col_used": talent_col,
        "roster_talent": talent_val,
        "pred_win_pct": pred_pct,
        "pred_wins": pred_wins,
    })

pred_df = pd.DataFrame(rows)
# convert to readable percentages / sorting
pred_df["prev_win_pct"] = pred_df["prev_win_pct"].astype(float)
pred_df["pred_win_pct"] = pred_df["pred_win_pct"].astype(float)
pred_df["pred_wins"] = pred_df["pred_wins"].astype(float)

# sort by predicted wins descending
pred_df_sorted = pred_df.sort_values("pred_wins", ascending=False).reset_index(drop=True)

# show table
st.dataframe(pred_df_sorted.style.format({
    "prev_win_pct": "{:.3f}",
    "pred_win_pct": "{:.3f}",
    "pred_wins": "{:.1f}"
}), height=500)

# Offer CSV download
csv_bytes = pred_df_sorted.to_csv(index=False).encode("utf-8")
st.download_button("Download all-team predictions (CSV)", data=csv_bytes, file_name="all_team_predictions.csv", mime="text/csv")

# -------------------------
# Per-team historical inspector
# -------------------------
st.header("Team history inspector")
teams = sorted(train_df["team"].unique())
team_choice = st.selectbox("Choose a team to inspect", options=teams, index=0 if teams else None)

if team_choice:
    team_rows = train_df[train_df["team"] == team_choice].sort_values("season")
    st.subheader(f"{team_choice} — historical seasons")
    st.table(team_rows[["season", "prev_win_pct", "coach_continuity", "roster_talent", "true_win_pct"]])

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
