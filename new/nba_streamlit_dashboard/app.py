from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from sklearn.isotonic import IsotonicRegression

# =============================
# Model constants from validation run
# =============================
W_PREV = 0.60
W_COACH = 0.00
W_TALENT = 0.40
TALENT_MIN = 0.00
TALENT_MAX = 10.00
VALIDATION_MAE_2425 = 0.0990
NBA_GAMES = 82

DATA_DIR = Path(__file__).resolve().parent
PREDICTIONS_FILE = DATA_DIR / "nba_25_26_predictions.csv"
TRUE_WIN_FILE = DATA_DIR / "true_win_pct.csv"
ROSTER_TALENT_FILE = DATA_DIR / "roster_talent-2.csv"

st.set_page_config(
    page_title="NBA Win Percentage Predictor",
    page_icon=None,
    layout="wide",
)

# =============================
# Styling
# =============================
st.markdown(
    """
    <style>
        .block-container {padding-top: 1.4rem; padding-bottom: 2rem;}
        h1, h2, h3 {letter-spacing: -0.02em;}
        div[data-testid="stMetric"] {
            background-color: #f7f7f7;
            border: 1px solid #e5e5e5;
            padding: 14px 16px;
            border-radius: 10px;
        }
        .small-note {
            color: #555;
            font-size: 0.92rem;
            line-height: 1.45;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# =============================
# Data loading
# =============================
@st.cache_data
def load_predictions() -> pd.DataFrame:
    df = pd.read_csv(PREDICTIONS_FILE)
    expected = {"Team", "prev", "talent", "talent_scaled", "raw_score", "pred_win_pct", "pred_wins"}
    missing = expected.difference(df.columns)
    if missing:
        raise ValueError(f"Prediction file is missing columns: {sorted(missing)}")

    df = df.copy()
    df["Team"] = df["Team"].astype(str)
    df["pred_win_pct"] = df["pred_win_pct"].astype(float)
    df["pred_wins"] = df["pred_wins"].astype(int)
    df["prev"] = df["prev"].astype(float)
    df["talent"] = df["talent"].astype(float)
    df["raw_score"] = df["raw_score"].astype(float)
    return df.sort_values("pred_wins", ascending=False).reset_index(drop=True)


@st.cache_data
def load_true_win_pct() -> pd.DataFrame:
    df = pd.read_csv(TRUE_WIN_FILE)
    df["Team"] = df["Team"].astype(str)
    return df


@st.cache_data
def load_roster_talent() -> pd.DataFrame:
    df = pd.read_csv(ROSTER_TALENT_FILE)
    df = df.rename(columns={df.columns[0]: "Team"})
    df["Team"] = df["Team"].astype(str)
    season_cols = [c for c in df.columns if str(c).count("-") == 1]
    return df[["Team"] + season_cols]


pred_df = load_predictions()
true_df = load_true_win_pct()
talent_history_df = load_roster_talent()

# Fit a simple calibration curve from the final projection file.
# This lets the custom user input produce predictions on the same calibrated scale
# as the saved 25-26 output rather than only returning the uncalibrated raw score.
calibration_model = IsotonicRegression(out_of_bounds="clip")
calibration_model.fit(pred_df["raw_score"], pred_df["pred_win_pct"])

# =============================
# Helper functions
# =============================
def scale_talent(talent_value: float) -> float:
    if TALENT_MAX == TALENT_MIN:
        return 0.0
    return (talent_value - TALENT_MIN) / (TALENT_MAX - TALENT_MIN)


def predict_win_pct(prev_win_pct: float, roster_talent: float) -> tuple[float, float, float]:
    talent_scaled = scale_talent(roster_talent)
    raw_score = W_PREV * prev_win_pct + W_COACH * 0 + W_TALENT * talent_scaled
    calibrated = float(calibration_model.predict([raw_score])[0])
    wins = calibrated * NBA_GAMES
    return calibrated, wins, raw_score


def make_team_timeline(team: str) -> pd.DataFrame:
    actual_row = true_df[true_df["Team"] == team].iloc[0]
    pred_row = pred_df[pred_df["Team"] == team].iloc[0]

    actual_cols = [c for c in true_df.columns if c != "Team"]
    actual_part = pd.DataFrame({
        "Season": actual_cols,
        "Win Percentage": [float(actual_row[c]) for c in actual_cols],
        "Series": "Actual win percentage",
    })
    pred_part = pd.DataFrame({
        "Season": ["25-26"],
        "Win Percentage": [float(pred_row["pred_win_pct"])],
        "Series": "Predicted win percentage",
    })
    return pd.concat([actual_part, pred_part], ignore_index=True)


def make_actual_vs_pred_all_teams(selected_teams: list[str]) -> pd.DataFrame:
    rows = []
    seasons = [c for c in true_df.columns if c != "Team"]
    for team in selected_teams:
        actual_row = true_df[true_df["Team"] == team].iloc[0]
        pred_row = pred_df[pred_df["Team"] == team].iloc[0]
        for season in seasons:
            rows.append({
                "Team": team,
                "Season": season,
                "Win Percentage": float(actual_row[season]),
                "Series": "Actual",
            })
        rows.append({
            "Team": team,
            "Season": "25-26",
            "Win Percentage": float(pred_row["pred_win_pct"]),
            "Series": "Prediction",
        })
    return pd.DataFrame(rows)

# =============================
# Header
# =============================
st.title("NBA Win Percentage Predictor")
st.markdown(
    "A clean dashboard for testing win percentage projections using previous win percentage and roster talent. "
    "The saved 25-26 team predictions use the final calibrated output from the model run."
)

# =============================
# Top metrics
# =============================
leader = pred_df.iloc[0]
median_team = pred_df.iloc[(len(pred_df) // 2)]

c1, c2, c3, c4 = st.columns(4)
c1.metric("Validation MAE on 24-25", f"{VALIDATION_MAE_2425:.4f}")
c2.metric("Model weights", f"Prev {W_PREV:.2f} / Talent {W_TALENT:.2f}")
c3.metric("Highest projected team", f"{leader['Team']}", f"{leader['pred_wins']} wins")
c4.metric("Median projected wins", f"{int(round(pred_df['pred_wins'].median()))}")

st.divider()

# =============================
# Sidebar filters
# =============================
st.sidebar.header("Filters")
all_teams = sorted(pred_df["Team"].unique())
default_teams = ["BOS", "DEN", "LAL", "OKC"] if set(["BOS", "DEN", "LAL", "OKC"]).issubset(all_teams) else all_teams[:4]
selected_teams = st.sidebar.multiselect(
    "Teams for line plot",
    options=all_teams,
    default=default_teams,
)
selected_single_team = st.sidebar.selectbox("Single-team view", options=all_teams, index=all_teams.index("OKC") if "OKC" in all_teams else 0)
min_projected_wins = st.sidebar.slider(
    "Minimum projected wins in table",
    min_value=int(pred_df["pred_wins"].min()),
    max_value=int(pred_df["pred_wins"].max()),
    value=int(pred_df["pred_wins"].min()),
)

filtered_pred = pred_df[pred_df["pred_wins"] >= min_projected_wins].copy()

# =============================
# User input simulator
# =============================
st.header("Custom Prediction")
left, right = st.columns([1, 1])

with left:
    prev_input = st.slider(
        "Previous season win percentage",
        min_value=0.00,
        max_value=1.00,
        value=0.50,
        step=0.01,
        help="Example: 0.50 means a .500 team, or about 41 wins over 82 games.",
    )
    talent_input = st.slider(
        "Roster talent factor",
        min_value=0.00,
        max_value=10.00,
        value=5.00,
        step=0.25,
        help="The training scale runs from 0 to 10, with 10 representing the strongest award-based talent score in the training data.",
    )

custom_pred_pct, custom_pred_wins, custom_raw_score = predict_win_pct(prev_input, talent_input)

with right:
    r1, r2, r3 = st.columns(3)
    r1.metric("Predicted win percentage", f"{custom_pred_pct:.3f}")
    r2.metric("Projected wins", f"{custom_pred_wins:.1f}")
    r3.metric("Raw model score", f"{custom_raw_score:.3f}")

    st.markdown(
        f"""
        <div class="small-note">
        Formula before calibration: <b>0.60 × previous win percentage + 0.40 × scaled roster talent</b>. 
        Coach continuity is included in the original structure but has a 0.00 weight in this run, so it does not move the prediction.
        </div>
        """,
        unsafe_allow_html=True,
    )

st.subheader("How roster talent is calculated")
st.markdown(
    "Roster talent is an award-based score meant to approximate high-end player quality. "
    "A team receives points for players with major recognition, then the total is scaled using the training-season range of 0 to 10. "
    "The model uses the scaled value, not the raw point total. In this model run, talent receives a 0.40 weight and previous win percentage receives a 0.60 weight."
)

st.markdown(
    "Scoring used in the projection script: All-NBA First Team = 4 points, All-NBA Second Team = 3 points, "
    "All-NBA Third Team = 2 points, All-Defense / major defensive recognition = 2 points, All-Star only = 1 point, "
    "with additional listed player-recognition adjustments where applicable."
)

# =============================
# Charts
# =============================
st.divider()
st.header("Model Output Visualizations")

chart_col1, chart_col2 = st.columns([1.2, 1])

with chart_col1:
    st.subheader("Actual vs predicted win percentage by team")
    if selected_teams:
        line_df = make_actual_vs_pred_all_teams(selected_teams)
        fig = px.line(
            line_df,
            x="Season",
            y="Win Percentage",
            color="Team",
            line_dash="Series",
            markers=True,
            hover_data={"Win Percentage": ":.3f"},
        )
        fig.update_layout(
            height=440,
            margin=dict(l=10, r=10, t=30, b=10),
            yaxis=dict(range=[0, 1], tickformat=".0%"),
            legend_title_text="Team / Series",
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Select at least one team to display the line plot.")

with chart_col2:
    st.subheader("Single-team timeline")
    team_timeline = make_team_timeline(selected_single_team)
    fig_team = px.line(
        team_timeline,
        x="Season",
        y="Win Percentage",
        color="Series",
        markers=True,
        hover_data={"Win Percentage": ":.3f"},
    )
    fig_team.update_layout(
        height=440,
        margin=dict(l=10, r=10, t=30, b=10),
        yaxis=dict(range=[0, 1], tickformat=".0%"),
        legend_title_text="",
    )
    st.plotly_chart(fig_team, use_container_width=True)

bar_col, scatter_col = st.columns([1, 1])

with bar_col:
    st.subheader("Projected wins by team")
    fig_bar = px.bar(
        filtered_pred.sort_values("pred_wins", ascending=True),
        x="pred_wins",
        y="Team",
        orientation="h",
        hover_data={"pred_win_pct": ":.3f", "prev": ":.3f", "talent": ":.1f"},
        labels={"pred_wins": "Projected Wins", "Team": "Team"},
    )
    fig_bar.update_layout(height=560, margin=dict(l=10, r=10, t=30, b=10), showlegend=False)
    st.plotly_chart(fig_bar, use_container_width=True)

with scatter_col:
    st.subheader("Prediction drivers")
    fig_scatter = px.scatter(
        pred_df,
        x="prev",
        y="talent",
        size="pred_wins",
        hover_name="Team",
        hover_data={"pred_win_pct": ":.3f", "pred_wins": True, "raw_score": ":.3f"},
        labels={"prev": "Previous Win Percentage", "talent": "Roster Talent Factor", "pred_wins": "Projected Wins"},
    )
    fig_scatter.update_layout(
        height=560,
        margin=dict(l=10, r=10, t=30, b=10),
        xaxis=dict(tickformat=".0%"),
    )
    st.plotly_chart(fig_scatter, use_container_width=True)

# =============================
# Tables and key numbers
# =============================
st.divider()
st.header("Key Numbers")

summary_col1, summary_col2 = st.columns([1, 1])

with summary_col1:
    st.subheader("25-26 predictions")
    display_cols = ["Team", "prev", "talent", "raw_score", "pred_win_pct", "pred_wins"]
    table = filtered_pred[display_cols].copy()
    table = table.rename(columns={
        "prev": "Prev Win %",
        "talent": "Roster Talent",
        "raw_score": "Raw Score",
        "pred_win_pct": "Predicted Win %",
        "pred_wins": "Projected Wins",
    })
    st.dataframe(
        table.style.format({
            "Prev Win %": "{:.3f}",
            "Roster Talent": "{:.1f}",
            "Raw Score": "{:.3f}",
            "Predicted Win %": "{:.3f}",
            "Projected Wins": "{:d}",
        }),
        use_container_width=True,
        hide_index=True,
    )

with summary_col2:
    st.subheader("Distribution summary")
    summary = pd.DataFrame({
        "Metric": [
            "Average projected win percentage",
            "Average projected wins",
            "Highest projected wins",
            "Lowest projected wins",
            "Average roster talent factor",
            "Average previous win percentage",
        ],
        "Value": [
            f"{pred_df['pred_win_pct'].mean():.3f}",
            f"{pred_df['pred_wins'].mean():.1f}",
            f"{pred_df['pred_wins'].max():.0f}",
            f"{pred_df['pred_wins'].min():.0f}",
            f"{pred_df['talent'].mean():.2f}",
            f"{pred_df['prev'].mean():.3f}",
        ],
    })
    st.dataframe(summary, use_container_width=True, hide_index=True)

    st.subheader("Roster talent history")
    talent_long = talent_history_df.melt(id_vars="Team", var_name="Season", value_name="Roster Talent")
    talent_team = talent_long[talent_long["Team"] == selected_single_team]
    fig_talent = px.line(
        talent_team,
        x="Season",
        y="Roster Talent",
        markers=True,
        title=f"{selected_single_team} roster talent trend",
    )
    fig_talent.update_layout(height=300, margin=dict(l=10, r=10, t=45, b=10))
    st.plotly_chart(fig_talent, use_container_width=True)

st.caption(
    "Model note: Final 25-26 predictions use isotonic ordering plus empirical quantile calibration. "
    "The custom input tool uses the same feature weights and a calibration curve fit from the saved 25-26 model output."
)
