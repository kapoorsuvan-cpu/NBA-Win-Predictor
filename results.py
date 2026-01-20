# predict_all_teams.py
# Predict and print projected wins for all 30 NBA teams (based on your 2025-26 inputs)

from main import train_from_csvs, predict_team_wins

def main():
    # Train the regression model using your historical CSVs in /csv
    train_from_csvs()

    # 2025-26 inputs (prev_w_pct, coach_continuity, roster_talent)
    teams = {
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

    # Predict wins, round to nearest whole win, and print sorted by wins (descending)
    results = []
    for team, v in teams.items():
        wins = predict_team_wins(v["prev_win_pct"], v["coach_continuity"], v["roster_talent"])
        results.append((team, round(wins)))

    results.sort(key=lambda x: x[1], reverse=True)

    for team, wins in results:
        print(f"{team}: {wins} wins")

if __name__ == "__main__":
    main()
