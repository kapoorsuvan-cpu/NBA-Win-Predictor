# app_dashboard.py
"""
Streamlit dashboard (robust CSV loader + in-memory training).

Place next to main.py and CSVs (either in csv/ or repo root), or upload CSVs via sidebar.
Then:
    pip install -r requirements.txt
    streamlit run app_dashboard.py

This version trains from the in-memory training DataFrame using main.train_model()
to avoid writing files on Streamlit Cloud.
"""

from typing import Optional, Tuple
import io
from pathlib import Path

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

# Import project functions/constants
from main import (
    _validate_inputs,
    build_training_dataframe,
    predict_team_win_pct,
    predict_team_wins,
    train_model,
)
import main as main_module  # used to set main_module.MODEL after training

st.set_page_config(page_title="NBA Win% Dashboard", layout="wide")
st.title("NBA Win% Model — Dashboard")

# -------------------------
# Sidebar / upload controls
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
# Helpers to load CSVs
# -------------------------
@st.cache_data
def _read_buffer(buffer: io.BytesIO) -> pd.DataFrame:
    return pd.read_csv(buffer, index_col=0, encoding="utf-8-sig")


REQUIRED_FILES = {
    "prev": "prev_win_pct.csv",
    "coach": "coach_continuity.csv",
    "talent": "roster_talent.csv",
    "true": "true_win_pct.csv",
}


def _try_load_path(path: Path) -> Optional[pd.DataFrame]:
    try:
        if not path.exists():
            return None
        df = pd.read_csv(path, index_col=0, encoding="utf-8-sig")
        df.columns = [str(c).strip() for c in df.columns]
        df.index = df.index.astype(str).str.strip()
        return df
    except Exception:
        return None


def load_data_or_repo() -> Optional[Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]]:
    # 1) uploaded files
    if use_uploads:
        if not (upload_prev and upload_coach and upload_talent and upload_true):
            st.sidebar.warning("Please upload all 4 CSVs to use uploaded data.")
            return None
        try:
            prev = _read_buffer(upload_prev)
            coach = _read_buffer(upload_coach)
            talent = _read_buffer(upload_talent)
            true = _read_buffer(upload_true)
            # normalize
            prev.columns = [str(c).strip() for c in prev.columns]
            coach.columns = [str(c).strip() for c in coach.columns]
            talent.columns = [str(c).strip() for c in talent.columns]
            true.columns = [str(c).strip() for c in true.columns]
            prev.index = prev.index.astype(str).str.strip()
            coach.index = coach.index.astype(str).str.strip()
            talent.index = talent.index.astype(str).str_strip() if hasattr(talent.index, "astype") else talent.index
            true.index = true.index.astype(str).str.strip()
            return prev, coach, talent, true
        except Exception as e:
            st.sidebar.error(f"Error reading uploaded CSVs: {e}")
            return None

    # 2) try csv/ then repo root
    base = Path(__file__).resolve().parent
    csv_dir = base / "csv"
    candidate_dirs = []
    if csv_dir.exists():
        candidate_dirs.append(csv_dir)
    candidate_dirs.append(base)

    for d in candidate_dirs:
        ok = True
        found = {}
        for key, fname in REQUIRED_FILES.items():
            df = _try_load_path(d / fname)
            if df is None:
                ok = False
                break
            found[key] = df
        if ok:
            return found["prev"], found["coach"], found["talent"], found["true"]

    # helpful error message listing tried paths
    tried = [str((base / "csv") / fname) for fname in REQUIRED_FILES.values()] + [str((base / fname)) for fname in REQUIRED_FILES.values()]
    st.error("Could not find all required CSVs. Tried these paths:\n\n" + "\n".join(tried) + "\n\nPlace files in csv/ or repo root, or upload via sidebar.")
    return None


# -------------------------
# Load data
# -------------------------
data = load_data_or_repo()
if data is None:
    st.info("Upload CSVs in the sidebar or place CSVs in csv/ or repo root (prev_win_pct.csv, coach_continuity.csv, roster_talent.csv, true_win_pct.csv).")
    st.stop()

prev_df, coach_df, talent_df, true_df = data

# validate and build train_df
try:
    _validate_inputs(prev_df, coach_df, talent_df, true_df)
    train_df = build_training_dataframe(prev_df, coach_df, talent_df, true_df)
except Exception as e:
    st.error(f"Error preparing training data: {e}")
    st.stop()


# -------------------------
# Train model from train_df (no disk writes)
# -------------------------
@st.cache_resource
def ensure_trained_from_df(train_df_local: pd.DataFrame):
    """
    Train using main.train_model(train_df) and set main_module.MODEL so
    other helpers predict_team_win_pct/predict_team_wins continue to work.
    """
    model = train_model(train_df_local)
    # set global in main module for downstream predict_* functions
    main_module.MODEL = model
    return model

if retrain_btn:
    st.sidebar.info("Retraining model from in-memory dataframe...")

model = ensure_trained_from_df(train_df)
if model is None:
    st.error("Model training failed.")
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
# Feature importance (% total)
# -------------------------
st.header("Feature importance")
st.subheader("Feature importance (% of total absolute weight)")

abs_coefs = coefs.abs()
total_abs = float(abs_coefs.sum())
if total_abs == 0:
    importance_pct = abs_coefs * 0.0
else:
    importance_pct = (abs_coefs / total_abs) * 100.0

feat_df = (
    pd.DataFrame({
        "feature": importance_pct.index,
        "importance_pct": importance_pct.values,
        "raw_coef": coefs.values
    })
    .sort_values("importance_pct", ascending=True)
)

fig_imp = px.bar(
    feat_df,
    x="importance_pct",
    y="feature",
    orientation="h",
    text=feat_df["importance_pct"].round(1),
    labels={"importance_pct": "Importance (%)"},
    title="Feature importance (percentage of total absolute weight)"
)
fig_imp.update_layout(height=320, margin=dict(l=40, r=10, t=40, b=10))
st.plotly_chart(fig_imp, use_container_width=True)

pct_table = feat_df[["feature", "importance_pct"]].sort_values("importance_pct", ascending=False).reset_index(drop=True)
pct_table["importance_pct"] = pct_table["importance_pct"].round(2)
st.table(pct_table.rename(columns={"importance_pct": "importance (%)"}))


# -------------------------
# Visualizations: scatter
# -------------------------
st.header("Historical: Previous season win% vs True win%")
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
# All teams predictions (2025-26 inputs provided)
# -------------------------
st.header("Predicted win totals for all teams (2025-26 inputs)")

teams_2026 = {
    "ATL": {"prev_win_pct": 0.49, "coach_continuity": 1, "roster_talent": 3},
    "BOS": {"prev_win_pct": 0.74, "coach_continuity": 1, "roster_talent": 5},
    "BRK": {"prev_win_pct": 0.32, "coach_continuity": 1, "roster_talent": 0},
    "CHI": {"prev_win_pct": 0.48, "coach_continuity": 1, "roster_talent": 0},
    "CHO": {"prev_win_pct": 0.23, "coach_continuity": 1, "roster_talent": 0},
    "CLE": {"prev_win_pct": 0.78, "coach_continuity": 1, "roster_talent": 10},
    "DAL": {"prev_win_pct": 0.48, "coach_continuity": 1, "roster_talent": 2},
    "DEN": {"prev_win_pct": 0.61, "coach_continuity": 0, "roster_talent": 4},
    "DET": {"prev_win_pct": 0.54, "coach_continuity": 1, "roster_talent": 5},
    "GSW": {"prev_win_pct": 0.59, "coach_continuity": 1, "roster_talent": 2},
    "HOU": {"prev_win_pct": 0.63, "coach_continuity": 1, "roster_talent": 4},
    "IND": {"prev_win_pct": 0.61, "coach_continuity": 1, "roster_talent": 3},
    "LAC": {"prev_win_pct": 0.61, "coach_continuity": 1, "roster_talent": 2},
    "LAL": {"prev_win_pct": 0.61, "coach_continuity": 1, "roster_talent": 3},
    "MEM": {"prev_win_pct": 0.59, "coach_continuity": 0, "roster_talent": 2},
    "MIA": {"prev_win_pct": 0.45, "coach_continuity": 1, "roster_talent": 1},
    "MIL": {"prev_win_pct": 0.59, "coach_continuity": 1, "roster_talent": 4},
    "MIN": {"prev_win_pct": 0.60, "coach_continuity": 1, "roster_talent": 3},
    "NOP": {"prev_win_pct": 0.26, "coach_continuity": 1, "roster_talent": 0},
    "NYK": {"prev_win_pct": 0.62, "coach_continuity": 0, "roster_talent": 5},
    "OKC": {"prev_win_pct": 0.83, "coach_continuity": 1, "roster_talent": 8},
    "ORL": {"prev_win_pct": 0.50, "coach_continuity": 1, "roster_talent": 0},
    "PHI": {"prev_win_pct": 0.29, "coach_continuity": 1, "roster_talent": 0},
    "PHO": {"prev_win_pct": 0.44, "coach_continuity": 0, "roster_talent": 0},
    "POR": {"prev_win_pct": 0.44, "coach_continuity": 0, "roster_talent": 0},
    "SAC": {"prev_win_pct": 0.49, "coach_continuity": 1, "roster_talent": 0},
    "SAS": {"prev_win_pct": 0.42, "coach_continuity": 0, "roster_talent": 1},
    "TOR": {"prev_win_pct": 0.37, "coach_continuity": 1, "roster_talent": 0},
    "UTA": {"prev_win_pct": 0.21, "coach_continuity": 1, "roster_talent": 0},
    "WAS": {"prev_win_pct": 0.22, "coach_continuity": 1, "roster_talent": 0},
}

rows = []
for team, vals in teams_2026.items():
    prev_pct = float(vals["prev_win_pct"])
    coach_val = int(vals["coach_continuity"])
    talent_val = float(vals["roster_talent"])
    try:
        pred_pct = predict_team_win_pct(prev_pct, coach_val, talent_val)
        pred_wins = predict_team_wins(prev_pct, coach_val, talent_val)
    except Exception:
        pred_pct = np.nan
        pred_wins = np.nan

    rows.append({
        "team": team,
        "prev_win_pct_input": prev_pct,
        "coach_continuity_input": coach_val,
        "roster_talent_input": talent_val,
        "pred_win_pct": pred_pct,
        "pred_wins": pred_wins,
    })

pred_teams_df = pd.DataFrame(rows)
pred_teams_df["pred_wins"] = pred_teams_df["pred_wins"].astype(float)
pred_teams_df["pred_win_pct"] = pred_teams_df["pred_win_pct"].astype(float)

pred_teams_df_sorted = pred_teams_df.sort_values("pred_wins", ascending=False).reset_index(drop=True)
st.dataframe(pred_teams_df_sorted.style.format({
    "prev_win_pct_input": "{:.3f}",
    "pred_win_pct": "{:.3f}",
    "pred_wins": "{:.1f}"
}), height=520)

st.download_button("Download 2025-26 predictions (CSV)", data=pred_teams_df_sorted.to_csv(index=False).encode("utf-8"), file_name="predictions_2025_26.csv", mime="text/csv")


# -------------------------
# Team inspector
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
# Export & downloads
# -------------------------
st.header("Export & downloads")
if st.button("Download coefficients CSV"):
    out = pd.DataFrame({"feature": coefs.index, "coefficient": coefs.values})
    st.download_button("Download coefficients CSV", data=out.to_csv(index=False).encode("utf-8"), file_name="coefficients.csv", mime="text/csv")

if st.button("Download training data CSV"):
    st.download_button("Download training CSV", data=train_df.to_csv(index=False).encode("utf-8"), file_name="training_data.csv", mime="text/csv")

st.markdown("---")
st.caption("Notes: predictions assume an 82-game season. Roster talent and coach continuity should match your CSV encodings.")
