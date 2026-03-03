import os
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL = None
FEATURE_NAMES = None


def read_wide_csv(path):
    df = pd.read_csv(path, index_col=0)
    df = df.reset_index()
    if df.columns[0] != "Team":
        df = df.rename(columns={df.columns[0]: "Team"})
    df = df.dropna(axis=1, how="all")
    return df


def wide_to_long(df, value_name):
    season_cols = [c for c in df.columns if c != "Team"]
    df_long = df.melt(
        id_vars=["Team"],
        value_vars=season_cols,
        var_name="season",
        value_name=value_name
    )
    df_long["Team"] = df_long["Team"].astype(str).str.strip()
    return df_long


def build_dataset():
    coach = wide_to_long(read_wide_csv(os.path.join(BASE_DIR, "coach_continuity.csv")), "coach_continuity")
    prev = wide_to_long(read_wide_csv(os.path.join(BASE_DIR, "prev_win_pct.csv")), "prev_win_pct")
    true = wide_to_long(read_wide_csv(os.path.join(BASE_DIR, "true_win_pct.csv")), "true_win_pct")
    roster = wide_to_long(read_wide_csv(os.path.join(BASE_DIR, "roster_talent.csv")), "roster_talent")

    df = true.merge(prev, on=["Team", "season"])
    df = df.merge(coach, on=["Team", "season"])
    df = df.merge(roster, on=["Team", "season"])

    df = df.dropna()

    return df



def train_model():
    global MODEL, FEATURE_NAMES

    df = build_dataset()

    X = df[["prev_win_pct", "coach_continuity", "roster_talent"]]
    y = df["true_win_pct"]

    FEATURE_NAMES = list(X.columns)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestRegressor(
        n_estimators=300,
        random_state=42
    )

    model.fit(X_train, y_train)

    r2 = model.score(X_test, y_test)
    logger.info(f"Model R²: {r2:.4f}")

    MODEL = model


def get_importance():
    if MODEL is None:
        raise RuntimeError("Model not trained.")

    raw = MODEL.feature_importances_
    pct = raw / raw.sum()

    order = np.argsort(-pct)

    names = [FEATURE_NAMES[i] for i in order]
    pct = pct[order]

    return names, pct


def plot_feature_importance():
    names, pct = get_importance()

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(8, 5))

    bars = ax.barh(
        names,
        pct * 100,
        color="#2a76c2",
        edgecolor="#164b7a",
        height=0.6
    )

    ax.set_xlabel("Percent Importance (%)", fontsize=12)
    ax.set_title(
        "Feature Importance\n(Previous Season Win %, Coach Continuity, Roster Talent)",
        fontsize=14,
        pad=12
    )

    ax.invert_yaxis()
    ax.grid(axis="x", linestyle=":", linewidth=0.8)

    for i, bar in enumerate(bars):
        width = bar.get_width()
        label = f"{pct[i]*100:.2f}%"

        if width > 15:
            ax.text(width - 2, bar.get_y() + bar.get_height()/2,
                    label, va="center", ha="right",
                    color="white", fontsize=11, fontweight="bold")
        else:
            ax.text(width + 1, bar.get_y() + bar.get_height()/2,
                    label, va="center", ha="left",
                    color="black", fontsize=11)

    for spine in ["top", "right", "left"]:
        ax.spines[spine].set_visible(False)

    plt.tight_layout()
    plt.savefig("feature_importance_pretty.png", dpi=300)
    plt.show()

if __name__ == "__main__":
    train_model()
    plot_feature_importance()
