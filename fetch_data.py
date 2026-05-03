"""
fetch_data.py
-------------
Pulls current season game logs and builds a feature-engineered dataset
ready for model training.

Usage:
    python data/fetch_data.py

Output:
    data/games_raw.csv        — raw game logs
    data/features.csv         — model-ready feature matrix
"""

import time
import pandas as pd
from nba_api.stats.endpoints import leaguegamelog, teamgamelogs
from nba_api.stats.static import teams

# ── Config ────────────────────────────────────────────────────────────────────
SEASON = "2024-25"
ROLLING_WINDOW = 5          # rolling average over last N games
OUTPUT_RAW = "games_raw.csv"
OUTPUT_FEATURES = "features.csv"

# ── Fetch raw game logs ───────────────────────────────────────────────────────
def fetch_game_logs(season: str) -> pd.DataFrame:
    print(f"Fetching game logs for {season}...")
    log = leaguegamelog.LeagueGameLog(season=season, season_type_all_star="Regular Season")
    df = log.get_data_frames()[0]
    df.to_csv(OUTPUT_RAW, index=False)
    print(f"  Saved {len(df)} rows to {OUTPUT_RAW}")
    return df

# ── Feature engineering ───────────────────────────────────────────────────────
def build_features(df: pd.DataFrame) -> pd.DataFrame:
    print("Engineering features...")

    df = df.copy()
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])
    df = df.sort_values(["TEAM_ID", "GAME_DATE"]).reset_index(drop=True)

    # Win flag (target)
    df["WIN"] = (df["WL"] == "W").astype(int)

    # Home/away from MATCHUP string (e.g. "BOS vs. MIA" = home, "BOS @ MIA" = away)
    df["HOME"] = df["MATCHUP"].apply(lambda x: 1 if "vs." in x else 0)

    # Rest days since last game
    df["PREV_DATE"] = df.groupby("TEAM_ID")["GAME_DATE"].shift(1)
    df["REST_DAYS"] = (df["GAME_DATE"] - df["PREV_DATE"]).dt.days.fillna(3)

    # Rolling averages for key stats (computed BEFORE the current game to avoid leakage)
    stat_cols = ["PTS", "REB", "AST", "TOV", "FG_PCT", "FG3_PCT", "FT_PCT", "PLUS_MINUS"]
    for col in stat_cols:
        df[f"ROLL_{col}"] = (
            df.groupby("TEAM_ID")[col]
            .transform(lambda x: x.shift(1).rolling(ROLLING_WINDOW, min_periods=1).mean())
        )

    # Rolling win rate
    df["ROLL_WIN_RATE"] = (
        df.groupby("TEAM_ID")["WIN"]
        .transform(lambda x: x.shift(1).rolling(ROLLING_WINDOW, min_periods=1).mean())
    )

    # ── Pair home and away teams per game ────────────────────────────────────
    # Each GAME_ID appears twice (once per team). We merge them into one row.
    home = df[df["HOME"] == 1].copy()
    away = df[df["HOME"] == 0].copy()

    roll_feature_cols = [f"ROLL_{c}" for c in stat_cols] + ["ROLL_WIN_RATE", "REST_DAYS"]
    meta_cols = ["GAME_ID", "GAME_DATE", "TEAM_ID", "TEAM_ABBREVIATION", "WIN"] + roll_feature_cols

    home = home[meta_cols].rename(columns={
        "TEAM_ID": "HOME_TEAM_ID",
        "TEAM_ABBREVIATION": "HOME_TEAM",
        "WIN": "HOME_WIN",
        **{c: f"HOME_{c}" for c in roll_feature_cols}
    })
    away = away[meta_cols].rename(columns={
        "TEAM_ID": "AWAY_TEAM_ID",
        "TEAM_ABBREVIATION": "AWAY_TEAM",
        "WIN": "AWAY_WIN",
        **{c: f"AWAY_{c}" for c in roll_feature_cols}
    })

    games = home.merge(away, on=["GAME_ID", "GAME_DATE"])

    # Differential features (home minus away) — often more predictive than raw values
    for col in roll_feature_cols:
        games[f"DIFF_{col}"] = games[f"HOME_{col}"] - games[f"AWAY_{col}"]

    # Target: did the home team win?
    games["TARGET"] = games["HOME_WIN"]

    games.to_csv(OUTPUT_FEATURES, index=False)
    print(f"  Saved {len(games)} paired games to {OUTPUT_FEATURES}")
    return games


if __name__ == "__main__":
    raw = fetch_game_logs(SEASON)
    time.sleep(1)   # be kind to the API
    features = build_features(raw)
    print("\nFeature columns:")
    print([c for c in features.columns if c.startswith(("DIFF_", "HOME_ROLL", "AWAY_ROLL"))])
    print("\nDone! Run model/train.py next.")