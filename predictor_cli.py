import argparse
import sys

from typing import Optional

# Import functions from your project
try:
    from main import (
        train_from_csvs,
        predict_team_win_pct,
        predict_team_wins,
        CSV_PREV_WIN,
        CSV_COACH,
        CSV_TALENT,
        CSV_TRUE_WIN,
        _load_csv,
        _validate_inputs,
        build_training_dataframe,
        MODEL,
    )
except Exception as e:
    print("Error importing project modules from main.py:", e, file=sys.stderr)
    raise

def ensure_trained_model() -> None:
    """
    Train the model by calling train_from_csvs() if needed.
    Will raise if CSVs missing or training fails.
    """
    # Only call train_from_csvs if MODEL is None to avoid retraining unnecessarily.
    global MODEL
    try:
        if MODEL is None:
            train_from_csvs()
    except Exception:
        # Re-raise with context so caller can handle / print
        raise

def predict_from_wins(prev_wins: float, roster_talent: float, coach_continuity: int, clamp: bool = True):
    """
    prev_wins: number between 0 and 82
    roster_talent: numeric (float or int)
    coach_continuity: 0 or 1
    returns (pred_pct, pred_wins)
    """
    prev_win_pct = float(prev_wins) / 82.0
    pred_pct = predict_team_win_pct(prev_win_pct, int(coach_continuity), float(roster_talent), clamp=clamp)
    pred_wins = predict_team_wins(prev_win_pct, int(coach_continuity), float(roster_talent), clamp=clamp)
    return pred_pct, pred_wins

def parse_args():
    p = argparse.ArgumentParser(description="Predict NBA win% and win total using the trained model (main.py).")
    p.add_argument("--prev-wins", type=float, help="Previous season wins (0-82). If omitted, will prompt interactively.")
    p.add_argument("--roster-talent", type=float, help="Roster talent metric (numeric). If omitted, will prompt interactively.")
    p.add_argument("--coach-continuity", type=int, choices=[0,1], help="Coach continuity (0=no, 1=yes). If omitted, will prompt interactively.")
    p.add_argument("--no-clamp", action="store_true", help="Disable clamping predicted win% to [0,1].")
    p.add_argument("--show-historical-summary", action="store_true", help="Attempt to print a very small historical summary (min/max prev win% in the training data).")
    return p.parse_args()

def prompt_float(prompt_text: str, default: Optional[float] = None, min_val: Optional[float] = None, max_val: Optional[float] = None) -> float:
    while True:
        raw = input(f"{prompt_text}" + (f" [{default}]" if default is not None else "") + ": ").strip()
        if raw == "" and default is not None:
            val = default
            break
        try:
            val = float(raw)
        except ValueError:
            print("Please enter a valid number.")
            continue
        if (min_val is not None and val < min_val) or (max_val is not None and val > max_val):
            print(f"Value must be between {min_val} and {max_val}.")
            continue
        break
    return val

def prompt_int_choice(prompt_text: str, choices, default: Optional[int] = None) -> int:
    choices_str = "/".join(str(c) for c in choices)
    while True:
        raw = input(f"{prompt_text} ({choices_str})" + (f" [{default}]" if default is not None else "") + ": ").strip()
        if raw == "" and default is not None:
            return default
        try:
            val = int(raw)
        except ValueError:
            print("Please enter a valid integer.")
            continue
        if val not in choices:
            print(f"Please choose from {choices}.")
            continue
        return val

def main():
    args = parse_args()

    # Get inputs either from args or interactively
    if args.prev_wins is None:
        prev_wins = prompt_float("Previous season wins (0 - 82)", default=41.0, min_val=0.0, max_val=82.0)
    else:
        prev_wins = args.prev_wins
        if not (0.0 <= prev_wins <= 82.0):
            print("Error: --prev-wins must be between 0 and 82.", file=sys.stderr)
            sys.exit(2)

    if args.roster_talent is None:
        roster_talent = prompt_float("Roster talent metric (numeric, e.g. 0,1,2)", default=0.0)
    else:
        roster_talent = args.roster_talent

    if args.coach_continuity is None:
        coach_cont = prompt_int_choice("Coach continuity? enter 1 for yes, 0 for no", choices=[0,1], default=1)
    else:
        coach_cont = args.coach_continuity

    # Train / ensure model available
    try:
        ensure_trained_model()
    except Exception as e:
        print("Failed to train/load model. Ensure csv/ folder exists with the required files and are valid.", file=sys.stderr)
        print("Underlying error:", e, file=sys.stderr)
        sys.exit(1)

    clamp = not args.no_clamp
    try:
        pred_pct, pred_wins = predict_from_wins(prev_wins, roster_talent, coach_cont, clamp=clamp)
    except Exception as e:
        print("Prediction failed:", e, file=sys.stderr)
        sys.exit(1)

    print("\n=== Prediction Result ===")
    print(f"Previous season wins: {prev_wins}  → prev win% = {prev_wins/82.0:.4f}")
    print(f"Roster talent: {roster_talent}")
    print(f"Coach continuity: {coach_cont} ({'yes' if coach_cont==1 else 'no'})")
    print(f"\nPredicted win%: {pred_pct:.4f}")
    print(f"Predicted wins (82-game): {pred_wins:.1f}")
    print("=========================\n")

    if args.show_historical_summary:
        # Attempt to load CSVs and report simple stats
        try:
            prev_df = _load_csv(CSV_PREV_WIN)
            true_df = _load_csv(CSV_TRUE_WIN)
            coach_df = _load_csv(CSV_COACH)
            talent_df = _load_csv(CSV_TALENT)
            _validate_inputs(prev_df, coach_df, talent_df, true_df)
            hist = build_training_dataframe(prev_df, coach_df, talent_df, true_df)
            print("Historical training data summary (per-season rows):")
            print(f"  Rows: {len(hist)}")
            print(f"  Prev win%  min: {hist['prev_win_pct'].min():.3f}, max: {hist['prev_win_pct'].max():.3f}, mean: {hist['prev_win_pct'].mean():.3f}")
            print(f"  True win%  min: {hist['true_win_pct'].min():.3f}, max: {hist['true_win_pct'].max():.3f}, mean: {hist['true_win_pct'].mean():.3f}")
        except Exception as e:
            print("Could not load historical summary (CSV missing or invalid):", e, file=sys.stderr)

if __name__ == "__main__":
    main()
