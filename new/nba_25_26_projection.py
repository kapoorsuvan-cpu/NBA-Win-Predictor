from pathlib import Path
import numpy as np
import pandas as pd
from collections import defaultdict
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import mean_absolute_error

BASE = Path(__file__).resolve().parent
INPUT = BASE / "input"
AUG = BASE / "augmented"
OUT = BASE / "output"

TEAMS = ["ATL","BOS","BRK","CHI","CHO","CLE","DAL","DEN","DET","GSW","HOU","IND","LAC","LAL","MEM","MIA","MIL","MIN","NOP","NYK","OKC","ORL","PHI","PHO","POR","SAC","SAS","TOR","UTA","WAS"]
SEASONS = ["20-21","21-22","22-23","23-24","24-25"]

def build_augmented_csvs():
    prev_df = pd.read_csv(INPUT / "prev_win_pct.csv")
    coach_df = pd.read_csv(INPUT / "coach_continuity.csv")
    talent_df = pd.read_csv(INPUT / "roster_talent.csv")
    true_df = pd.read_csv(INPUT / "true_win_pct.csv")

    true_2425 = dict(zip(true_df["Team"], true_df["24-25"]))
    prev_df["25-26"] = prev_df["Team"].map(true_2425)
    coach_df["25-26"] = coach_df["Team"].apply(lambda t: 0 if t in {"NYK", "PHO"} else 1)

    awards = defaultdict(int)
    def add(team, pts):
        awards[team] += pts

    for team in ["MIL", "OKC", "DEN", "CLE", "BOS"]: add(team, 4)
    for team in ["NYK", "MIN", "GSW", "LAL", "CLE"]: add(team, 3)
    for team in ["DET", "IND", "LAC", "NYK", "OKC"]: add(team, 2)
    for team in ["ATL", "OKC", "GSW", "CLE", "HOU"]: add(team, 2)
    for team in ["POR", "MIN", "MEM", "OKC", "LAC"]: add(team, 1)
    for team, pts in {"BOS": 1, "GSW": 1, "PHO": 1, "LAC": 1, "DAL": 2, "LAL": 1, "MIL": 1, "CLE": 1, "MIA": 1, "MEM": 1, "HOU": 1, "IND": 1, "SAS": 1, "ATL": 1}.items():
        add(team, pts)

    talent_df["25-26"] = talent_df["Team"].map(lambda t: awards[t])
    true_df["25-26"] = np.nan

    prev_df.to_csv(AUG / "prev_win_pct_augmented.csv", index=False)
    coach_df.to_csv(AUG / "coach_continuity_augmented.csv", index=False)
    talent_df.to_csv(AUG / "roster_talent_augmented.csv", index=False)
    true_df.to_csv(AUG / "true_win_pct_augmented.csv", index=False)

    return prev_df, coach_df, talent_df, true_df


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    prev_df, coach_df, talent_df, true_df = build_augmented_csvs()

    rows = []
    for t in TEAMS:
        for s in SEASONS:
            rows.append({
                "Team": t,
                "Season": s,
                "prev": float(prev_df.loc[prev_df.Team == t, s].iloc[0]),
                "coach": float(coach_df.loc[coach_df.Team == t, s].iloc[0]),
                "talent": float(talent_df.loc[talent_df.Team == t, s].iloc[0]),
                "true": float(true_df.loc[true_df.Team == t, s].iloc[0]),
            })
    long = pd.DataFrame(rows)
    train = long[long["Season"] != "24-25"].copy()
    val = long[long["Season"] == "24-25"].copy()

    w_prev, w_coach, w_talent = 0.60, 0.00, 0.40
    lo = float(train["talent"].min())
    hi = float(train["talent"].max())
    train_score = w_prev * train["prev"] + w_coach * train["coach"] + w_talent * ((train["talent"] - lo) / (hi - lo))
    val_score = w_prev * val["prev"] + w_coach * val["coach"] + w_talent * ((val["talent"] - lo) / (hi - lo))

    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(train_score, train["true"])
    val_mae = mean_absolute_error(val["true"], iso.predict(val_score))

    pred = pd.DataFrame({
        "Team": TEAMS,
        "prev": prev_df["25-26"].values,
        "coach": coach_df["25-26"].values,
        "talent": talent_df["25-26"].values,
    })
    pred["talent_scaled"] = (pred["talent"] - lo) / (hi - lo)
    pred["raw_score"] = w_prev * pred["prev"] + w_coach * pred["coach"] + w_talent * pred["talent_scaled"]
    pred["iso_win_pct"] = iso.predict(pred["raw_score"])

    q = np.linspace(0, 1, len(pred), endpoint=False) + 0.5 / len(pred)
    ref = np.quantile(long["true"], q)
    order = np.argsort(pred["iso_win_pct"].values)
    quantile_win_pct = np.empty(len(pred), dtype=float)
    quantile_win_pct[order] = np.sort(ref)
    pred["pred_win_pct"] = quantile_win_pct
    # Keep the full distribution shape without forcing a hard floor.
    # This preserves low-end separation so bad teams can still project in the
    # high teens/low 20s when the features justify it, instead of being
    # artificially pushed up toward 30 wins.
    pred["pred_wins"] = np.rint(pred["pred_win_pct"] * 82).astype(int)
    pred = pred.sort_values(["pred_wins", "pred_win_pct"], ascending=False).reset_index(drop=True)
    pred.to_csv(OUT / "nba_25_26_predictions.csv", index=False)

    pd.DataFrame([
        {"feature": "prev_win_pct", "weight": w_prev},
        {"feature": "coach_continuity", "weight": w_coach},
        {"feature": "roster_talent", "weight": w_talent},
    ]).to_csv(OUT / "chosen_weights.csv", index=False)

    with open(OUT / "model_summary.txt", "w") as f:
        f.write(f"Validation MAE on 24-25: {val_mae:.4f}\n")
        f.write(f"Weights: prev={w_prev:.2f}, coach={w_coach:.2f}, talent={w_talent:.2f}\n")
        f.write(f"Roster talent scale min/max from training seasons: {lo:.2f}/{hi:.2f}\n")
        f.write("Final 25-26 predictions use isotonic ordering + empirical quantile calibration.\n")

if __name__ == '__main__':
    main()
