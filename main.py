from pathlib import Path
import pandas as pd
import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score


BASE_DIR = Path(__file__).resolve().parent
CSV_DIR = BASE_DIR / "csv"

CSV_PREV_WIN = CSV_DIR / "prev_win_pct.csv"
CSV_COACH = CSV_DIR / "coach_continuity.csv"
CSV_TALENT = CSV_DIR / "roster_talent.csv"
CSV_TRUE_WIN = CSV_DIR / "true_win_pct.csv"


#  columns

COLS = ["20-21", "21-22", "22-23", "23-24", "24-25"]

#only train on seasons that have a previous season in the dataset
TARGET_YEARS = ["21-22", "22-23", "23-24", "24-25"]
PREV_YEARS = ["20-21", "21-22", "22-23", "23-24"]

# global model
MODEL: LinearRegression | None = None


# load files
def _load_csv(path: Path) -> pd.DataFrame:
    """Load CSV with teams as index, seasons as columns."""
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    df = pd.read_csv(path, index_col=0, encoding="utf-8-sig")

    df.columns = [str(c).strip() for c in df.columns]
    df.index = df.index.astype(str).str.strip()
    return df


def _validate_inputs(prev_win: pd.DataFrame,
                     coach: pd.DataFrame,
                     talent: pd.DataFrame,
                     true_win: pd.DataFrame) -> None:
    """Ensure all four dataframes align and have the correct season columns."""
    for name, df in [
        ("prev_win_pct", prev_win),
        ("coach_continuity", coach),
        ("roster_talent", talent),
        ("true_win_pct", true_win),
    ]:
        missing = [c for c in COLS if c not in df.columns]
        if missing:
            raise ValueError(
                f"{name} missing columns {missing}. "
                f"Found columns: {list(df.columns)}"
            )

    if not (set(prev_win.index) == set(coach.index) == set(talent.index) == set(true_win.index)):
        raise ValueError(
            "Team rows (index) do not match across CSVs. "
            "Make sure the first column in each CSV is the same team abbreviations."
        )

    teams_sorted = sorted(prev_win.index)
    prev_win.sort_index(inplace=True)
    coach.sort_index(inplace=True)
    talent.sort_index(inplace=True)
    true_win.sort_index(inplace=True)

    if sorted(prev_win.index) != teams_sorted:
        raise ValueError("Unexpected issue sorting team indices.")


def build_training_dataframe(prev_win: pd.DataFrame,
                             coach: pd.DataFrame,
                             talent: pd.DataFrame,
                             true_win: pd.DataFrame) -> pd.DataFrame:
    """
    Long-format dataset: one row per (team, target_year).
    Uses prev_win from PREV_YEARS to predict true_win in TARGET_YEARS.
    """
    rows = []
    teams = prev_win.index.tolist()

    for team in teams:
        for y, py in zip(TARGET_YEARS, PREV_YEARS):
            rows.append({
                "team": team,
                "season": y,
                "prev_win_pct": prev_win.loc[team, py],
                "coach_continuity": coach.loc[team, y],
                "roster_talent": talent.loc[team, y],
                "true_win_pct": true_win.loc[team, y],
            })

    df = pd.DataFrame(rows)

    for col in ["prev_win_pct", "coach_continuity", "roster_talent", "true_win_pct"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # check for missing/invalid
    if df.isnull().any().any():
        bad = df[df.isnull().any(axis=1)]
        raise ValueError(
            "Found missing/invalid numeric values after building training data. "
            "Fix these rows:\n" + bad.to_string(index=False)
        )

    return df



# train 

def train_model(df: pd.DataFrame,
                test_size: float = 0.2,
                random_state: int = 42) -> LinearRegression:
    X = df[["prev_win_pct", "coach_continuity", "roster_talent"]]
    y = df["true_win_pct"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    model = LinearRegression()
    model.fit(X_train, y_train)

    # evaluate
    y_pred = model.predict(X_test)
    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
    r2 = float(r2_score(y_test, y_pred))

   # print("Model Evaluation")
   # print(f"RMSE (win%): {rmse:.4f}")
   # print(f"R^2:        {r2:.4f}")
   # print()

    coefs = pd.Series(model.coef_, index=X.columns)
    print("Coefficients")
    print(coefs.to_string())
    print(f"Intercept: {model.intercept_:.6f}")
    print()

    return model



# prediction API 

def clamp_win_pct(win_pct: float) -> float:
    """Keep predicted win% in [0, 1] to avoid nonsense outputs."""
    return float(max(0.0, min(1.0, win_pct)))


def predict_team_win_pct(prev_win_pct: float,
                         coach_continuity: int,
                         roster_talent: float,
                         clamp: bool = True) -> float:
    """
    Predict win% from inputs using the trained global MODEL.
    """
    if MODEL is None:
        raise RuntimeError("MODEL is not trained. Run train_from_csvs() first.")

    x = np.array([[float(prev_win_pct), int(coach_continuity), float(roster_talent)]])
    win_pct = float(MODEL.predict(x)[0])
    return clamp_win_pct(win_pct) if clamp else win_pct


def predict_team_wins(prev_win_pct: float,
                      coach_continuity: int,
                      roster_talent: float,
                      clamp: bool = True) -> float:
    """
    Returns predicted total wins:
      predicted_win_pct * 82, rounded to 1 decimal place.
    """
    win_pct = predict_team_win_pct(prev_win_pct, coach_continuity, roster_talent, clamp=clamp)
    return round(win_pct * 82.0, 1)



# training from csvs
def train_from_csvs() -> None:
    global MODEL

    if not CSV_DIR.exists():
        raise FileNotFoundError(
            f"CSV folder not found: {CSV_DIR}\n"
            f"Create it and put your 4 CSVs there."
        )

   # print("CSV directory:", CSV_DIR)
   # print("CSV files found:", [p.name for p in CSV_DIR.glob("*.csv")])
   # print()

    prev_win = _load_csv(CSV_PREV_WIN)
    coach = _load_csv(CSV_COACH)
    talent = _load_csv(CSV_TALENT)
    true_win = _load_csv(CSV_TRUE_WIN)

    _validate_inputs(prev_win, coach, talent, true_win)

    df = build_training_dataframe(prev_win, coach, talent, true_win)

   # print("Training Data Check")
   # print(f"Rows: {len(df)} (expected 30 teams * 4 seasons = 120)")
   # print(df.head(10).to_string(index=False))
   # print()

    MODEL = train_model(df)



# main script
if __name__ == "__main__":
    train_from_csvs()

    kings_prev_w_pct = 0.488
    kings_coach_continuity = 1
    kings_roster_talent = 0

    heat_prev_w_pct = 0.451
    heat_coach_continuity = 1
    heat_roster_talent = 1

    grizz_prev_w_pct = 0.585
    grizz_coach_continuity = 0
    grizz_roster_talent = 2

    kings_pred_win_pct = predict_team_win_pct(kings_prev_w_pct, kings_coach_continuity, kings_roster_talent)
    kings_pred_wins = predict_team_wins(kings_prev_w_pct, kings_coach_continuity, kings_roster_talent)

    print("Kings Prediction")
    print("Predicted win% :", round(kings_pred_win_pct, 4))
    print("Predicted wins :", round(kings_pred_wins))

    heat_pred_win_pct = predict_team_win_pct(heat_prev_w_pct, heat_coach_continuity, heat_roster_talent)
    heat_pred_wins = predict_team_wins(heat_prev_w_pct, heat_coach_continuity, heat_roster_talent)

    print("Heat Prediction")
    print("Predicted win% :", round(heat_pred_win_pct, 4))
    print("Predicted wins :", round(heat_pred_wins))

    grizz_pred_win_pct = predict_team_win_pct(grizz_prev_w_pct, grizz_coach_continuity, grizz_roster_talent)
    grizz_pred_wins = predict_team_wins(grizz_prev_w_pct, grizz_coach_continuity, grizz_roster_talent)

    print("Grizzlies Prediction")
    print("Predicted win% :", round(grizz_pred_win_pct, 4))
    print("Predicted wins :", round(grizz_pred_wins))
