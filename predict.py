"""
predict.py
----------
Given a home team and away team (by abbreviation), fetches their recent
rolling stats and returns a win probability + SHAP explanation.

Usage (CLI):
    python predict.py --home BOS --away MIA

Usage (as module):
    from predict import predict_game
    result = predict_game("BOS", "MIA")
"""

import argparse
import joblib
import numpy as np
import pandas as pd

MODEL_PATH = "nba_model.joblib"
EXPLAINER_PATH = "shap_explainer.joblib"
FEATURE_COLS_PATH = "feature_cols.txt"
FEATURES_PATH = "features.csv"

# ── Load artifacts ─────────────────────────────────────────────────────────────
def load_artifacts():
    model = joblib.load(MODEL_PATH)
    explainer = joblib.load(EXPLAINER_PATH)
    with open(FEATURE_COLS_PATH) as f:
        feature_cols = [line.strip() for line in f.readlines()]
    return model, explainer, feature_cols

# ── Get latest rolling stats for a team ───────────────────────────────────────
def get_latest_team_stats(df: pd.DataFrame, team_abbr: str, side: str) -> dict:
    """
    Pulls the most recent row for a team from the features CSV.
    side: "HOME" or "AWAY"
    """
    team_rows = df[df[f"{side}_TEAM"] == team_abbr].sort_values("GAME_DATE")
    if team_rows.empty:
        raise ValueError(f"Team '{team_abbr}' not found in features data as {side}.")
    return team_rows.iloc[-1]

# ── Build feature vector ───────────────────────────────────────────────────────
def build_feature_vector(home_team: str, away_team: str, feature_cols: list) -> pd.DataFrame:
    df = pd.read_csv(FEATURES_PATH, parse_dates=["GAME_DATE"])

    # Get most recent stats for each team regardless of home/away role
    # We take the last game they appeared in (either side) for rolling stats
    all_home = df[df["HOME_TEAM"] == home_team].sort_values("GAME_DATE")
    all_away_as_home = df[df["HOME_TEAM"] == away_team].sort_values("GAME_DATE")
    all_home_as_away = df[df["AWAY_TEAM"] == home_team].sort_values("GAME_DATE")
    all_away = df[df["AWAY_TEAM"] == away_team].sort_values("GAME_DATE")

    def latest_stats(team, as_home_df, as_away_df, prefix_in_home, prefix_in_away):
        """Get the most recent rolling stats for a team."""
        rows = []
        if not as_home_df.empty:
            row = as_home_df.iloc[-1]
            rows.append((row["GAME_DATE"], row, prefix_in_home))
        if not as_away_df.empty:
            row = as_away_df.iloc[-1]
            rows.append((row["GAME_DATE"], row, prefix_in_away))
        if not rows:
            raise ValueError(f"No data found for team '{team}'")
        rows.sort(key=lambda x: x[0], reverse=True)
        _, row, prefix = rows[0]
        return row, prefix

    home_row, home_prefix = latest_stats(home_team, all_home, all_home_as_away, "HOME", "AWAY")
    away_row, away_prefix = latest_stats(away_team, all_away_as_home, all_away, "HOME", "AWAY")

    # Map column names to what the model expects
    stat_keys = [
        "ROLL_PTS", "ROLL_REB", "ROLL_AST", "ROLL_TOV",
        "ROLL_FG_PCT", "ROLL_FG3_PCT", "ROLL_FT_PCT",
        "ROLL_PLUS_MINUS", "ROLL_WIN_RATE", "REST_DAYS"
    ]

    home_stats = {k: home_row[f"{home_prefix}_{k}"] for k in stat_keys}
    away_stats = {k: away_row[f"{away_prefix}_{k}"] for k in stat_keys}

    vec = {}
    for k in stat_keys:
        vec[f"DIFF_{k}"] = home_stats[k] - away_stats[k]
    vec["HOME_ROLL_WIN_RATE"] = home_stats["ROLL_WIN_RATE"]
    vec["AWAY_ROLL_WIN_RATE"] = away_stats["ROLL_WIN_RATE"]
    vec["HOME_REST_DAYS"] = home_stats["REST_DAYS"]
    vec["AWAY_REST_DAYS"] = away_stats["REST_DAYS"]

    return pd.DataFrame([vec])[feature_cols]

# ── Main prediction function ───────────────────────────────────────────────────
def predict_game(home_team: str, away_team: str) -> dict:
    """
    Returns a dict with:
        home_team, away_team,
        home_win_prob, away_win_prob,
        predicted_winner,
        shap_values (array), feature_cols (list)
    """
    model, explainer, feature_cols = load_artifacts()
    X = build_feature_vector(home_team, away_team, feature_cols)

    home_win_prob = float(model.predict_proba(X)[0][1])
    away_win_prob = 1.0 - home_win_prob
    predicted_winner = home_team if home_win_prob >= 0.5 else away_team

    shap_values = explainer.shap_values(X.values)[0]

    return {
        "home_team": home_team,
        "away_team": away_team,
        "home_win_prob": round(home_win_prob, 4),
        "away_win_prob": round(away_win_prob, 4),
        "predicted_winner": predicted_winner,
        "confidence": round(max(home_win_prob, away_win_prob), 4),
        "shap_values": shap_values,
        "feature_cols": feature_cols,
        "feature_values": X.values[0],
    }

# ── CLI ────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict NBA game outcome.")
    parser.add_argument("--home", required=True, help="Home team abbreviation (e.g. BOS)")
    parser.add_argument("--away", required=True, help="Away team abbreviation (e.g. MIA)")
    args = parser.parse_args()

    result = predict_game(args.home.upper(), args.away.upper())

    print(f"\n{'='*40}")
    print(f"  {result['home_team']} (Home) vs {result['away_team']} (Away)")
    print(f"{'='*40}")
    print(f"  Predicted winner : {result['predicted_winner']}")
    print(f"  Home win prob    : {result['home_win_prob']:.1%}")
    print(f"  Away win prob    : {result['away_win_prob']:.1%}")
    print(f"  Confidence       : {result['confidence']:.1%}")
    print(f"{'='*40}")
    print("\nTop SHAP factors:")
    pairs = sorted(zip(result["shap_values"], result["feature_cols"]), key=lambda x: abs(x[0]), reverse=True)
    for val, feat in pairs[:5]:
        direction = "▲ favors home" if val > 0 else "▼ favors away"
        print(f"  {feat:35s} {val:+.4f}  {direction}")