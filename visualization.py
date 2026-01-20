# visualization.py
# Scatter plot of historical training data + highlighted new predictions (Kings, Heat, Grizzlies)
import numpy as np
import matplotlib.pyplot as plt

from main import (
    train_from_csvs,
    _load_csv,
    _validate_inputs,
    build_training_dataframe,
    predict_team_win_pct,
    CSV_PREV_WIN,
    CSV_COACH,
    CSV_TALENT,
    CSV_TRUE_WIN,
)


def plot_win_pct_scatter(new_teams: dict) -> None:
    """
    Scatter plot:
      - Historical: x = prev_win_pct, y = true_win_pct
      - New points: x = prev_win_pct, y = predicted_win_pct (highlighted)

    new_teams format:
    {
      "Kings": {"prev_win_pct": 0.488, "coach_continuity": 1, "roster_talent": 0},
      ...
    }
    """
    # Train model (silent in your main.py version)
    train_from_csvs()

    # Load data for plotting historical points
    prev_win = _load_csv(CSV_PREV_WIN)
    coach = _load_csv(CSV_COACH)
    talent = _load_csv(CSV_TALENT)
    true_win = _load_csv(CSV_TRUE_WIN)

    _validate_inputs(prev_win, coach, talent, true_win)
    df = build_training_dataframe(prev_win, coach, talent, true_win)
    r = np.corrcoef(df["prev_win_pct"], df["true_win_pct"])[0, 1]

    plt.figure(figsize=(10, 7))
    plt.scatter(df["prev_win_pct"], df["true_win_pct"], alpha=0.6, label="Historical team-seasons")

    for team, vals in new_teams.items():
        pred_win_pct = predict_team_win_pct(
            vals["prev_win_pct"],
            vals["coach_continuity"],
            vals["roster_talent"]
        )

        plt.scatter(vals["prev_win_pct"], pred_win_pct, marker="X", s=140, label=f"{team} (prediction)")
        plt.text(vals["prev_win_pct"] + 0.003, pred_win_pct + 0.003, team, fontsize=9, weight="bold")

    # --- Formatting ---
    plt.xlabel("Previous Season Win %")
    plt.ylabel("Win % (True for historical, Predicted for highlighted)")
    plt.title("NBA Win %: Historical Data vs Highlighted New Predictions")
    
    plt.text(0.02, 0.95, f"Correlation = {r:.3f}",transform=plt.gca().transAxes, fontsize=11,verticalalignment="top")
    
    plt.grid(True)
    plt.legend()
    plt.show()


if __name__ == "__main__":
    teams = {
        "Kings": {"prev_win_pct": 0.488, "coach_continuity": 1, "roster_talent": 0},
        "Heat": {"prev_win_pct": 0.451, "coach_continuity": 1, "roster_talent": 1},
        "Grizzlies": {"prev_win_pct": 0.585, "coach_continuity": 0, "roster_talent": 2},
    }

    plot_win_pct_scatter(teams)
