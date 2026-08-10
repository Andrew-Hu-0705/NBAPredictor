"""
fetch_data.py
-------------
Pulls game logs for one or more seasons and builds a feature-engineered
dataset ready for model training.

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
# Multiple seasons give the model more rows to learn from than a single
# season (~1200 games) can support. Rolling/rest stats are computed
# separately per season below so nothing leaks across the off-season gap.
SEASONS = ["2022-23", "2023-24", "2024-25"]
ROLLING_WINDOW = 10         # rolling average over last N games — tuned via CV, see train.py notes
OUTPUT_RAW = "games_raw.csv"
OUTPUT_FEATURES = "features.csv"

# Elo config. K and home-court advantage follow the values FiveThirtyEight
# published for their NBA Elo model. SEASON_CARRYOVER controls how much of a
# team's rating survives the off-season (25% reversion to the mean, so a
# team's rating isn't permanently anchored to a roster that's since turned over).
ELO_INITIAL = 1500.0
ELO_MEAN = 1500.0
ELO_K = 20.0
ELO_HOME_ADVANTAGE = 100.0
ELO_SEASON_CARRYOVER = 0.75

# ── Fetch raw game logs ───────────────────────────────────────────────────────
def fetch_one_season(season: str) -> pd.DataFrame:
    print(f"Fetching basic game logs for {season}...")
    log = leaguegamelog.LeagueGameLog(season=season, season_type_all_star="Regular Season")
    df_base = log.get_data_frames()[0]

    print(f"Fetching advanced game logs for {season}...")
    time.sleep(1)  # be kind to the API
    adv_log = teamgamelogs.TeamGameLogs(season_nullable=season, measure_type_player_game_logs_nullable="Advanced")
    df_adv = adv_log.get_data_frames()[0]

    # Merge on GAME_ID and TEAM_ID
    cols_to_use = df_adv.columns.difference(df_base.columns).tolist() + ["GAME_ID", "TEAM_ID"]
    df = pd.merge(df_base, df_adv[cols_to_use], on=["GAME_ID", "TEAM_ID"], how="left")
    df["SEASON"] = season
    return df

def fetch_game_logs(seasons: list) -> pd.DataFrame:
    dfs = []
    for i, season in enumerate(seasons):
        dfs.append(fetch_one_season(season))
        if i < len(seasons) - 1:
            time.sleep(1)  # be kind to the API between seasons
    df = pd.concat(dfs, ignore_index=True)

    df.to_csv(OUTPUT_RAW, index=False)
    print(f"  Saved {len(df)} rows across {len(seasons)} season(s) to {OUTPUT_RAW}")
    return df

# ── Elo ratings ────────────────────────────────────────────────────────────────
def compute_elo(games: pd.DataFrame) -> pd.DataFrame:
    """
    Adds DIFF_ELO: the home team's Elo rating minus the away team's, as of
    *before* this game (so it can't leak this game's outcome). Ratings are
    updated after each game using a margin-of-victory multiplier (same shape
    as FiveThirtyEight's NBA Elo formula) and partially revert to the mean
    between seasons.
    """
    games = games.sort_values(["SEASON", "GAME_DATE", "GAME_ID"]).reset_index(drop=True)
    ratings = {}
    current_season = None
    home_elo_pre, away_elo_pre = [], []

    for row in games.itertuples():
        if row.SEASON != current_season:
            if current_season is not None:
                for team in ratings:
                    ratings[team] = ELO_MEAN + (ratings[team] - ELO_MEAN) * ELO_SEASON_CARRYOVER
            current_season = row.SEASON

        rh = ratings.get(row.HOME_TEAM_ID, ELO_INITIAL)
        ra = ratings.get(row.AWAY_TEAM_ID, ELO_INITIAL)
        home_elo_pre.append(rh)
        away_elo_pre.append(ra)

        rh_adj = rh + ELO_HOME_ADVANTAGE
        expected_home = 1.0 / (1.0 + 10 ** (-(rh_adj - ra) / 400.0))
        actual_home = row.HOME_WIN

        elo_diff_winner = (rh_adj - ra) if actual_home == 1 else (ra - rh_adj)
        mov_mult = ((abs(row.HOME_GAME_MARGIN) + 3) ** 0.8) / (7.5 + 0.006 * elo_diff_winner)

        delta = ELO_K * mov_mult * (actual_home - expected_home)
        ratings[row.HOME_TEAM_ID] = rh + delta
        ratings[row.AWAY_TEAM_ID] = ra - delta

    games["HOME_ELO_PRE"] = home_elo_pre
    games["AWAY_ELO_PRE"] = away_elo_pre
    games["DIFF_ELO"] = games["HOME_ELO_PRE"] - games["AWAY_ELO_PRE"]
    return games

# ── Feature engineering ───────────────────────────────────────────────────────
def build_features(df: pd.DataFrame, window: int = ROLLING_WINDOW, save: bool = True) -> pd.DataFrame:
    print(f"Engineering features (rolling window={window})...")

    df = df.copy()
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])
    df = df.sort_values(["TEAM_ID", "GAME_DATE"]).reset_index(drop=True)

    # Win flag (target)
    df["WIN"] = (df["WL"] == "W").astype(int)

    # Home/away from MATCHUP string (e.g. "BOS vs. MIA" = home, "BOS @ MIA" = away)
    df["HOME"] = df["MATCHUP"].apply(lambda x: 1 if "vs." in x else 0)

    # Rest days since last game. Grouped by SEASON too so a team's first game of a
    # new season doesn't get a multi-month "rest" value from the prior off-season.
    df["PREV_DATE"] = df.groupby(["TEAM_ID", "SEASON"])["GAME_DATE"].shift(1)
    df["REST_DAYS"] = (df["GAME_DATE"] - df["PREV_DATE"]).dt.days.fillna(3)

    # Rolling averages for key stats (computed BEFORE the current game to avoid leakage).
    # Grouped by SEASON too so rolling form doesn't carry over across a roster
    # turnover / off-season.
    stat_cols = [
        "PTS", "REB", "AST", "TOV", "FG_PCT", "FG3_PCT", "FT_PCT", "PLUS_MINUS",
        "OFF_RATING", "DEF_RATING", "PACE", "TS_PCT", "EFG_PCT", "OREB_PCT", "DREB_PCT", "TM_TOV_PCT"
    ]
    for col in stat_cols:
        df[f"ROLL_{col}"] = (
            df.groupby(["TEAM_ID", "SEASON"])[col]
            .transform(lambda x: x.shift(1).rolling(window, min_periods=1).mean())
        )

    # Rolling win rate
    df["ROLL_WIN_RATE"] = (
        df.groupby(["TEAM_ID", "SEASON"])["WIN"]
        .transform(lambda x: x.shift(1).rolling(window, min_periods=1).mean())
    )

    # ── Pair home and away teams per game ────────────────────────────────────
    # Each GAME_ID appears twice (once per team). We merge them into one row.
    home = df[df["HOME"] == 1].copy()
    away = df[df["HOME"] == 0].copy()

    roll_feature_cols = [f"ROLL_{c}" for c in stat_cols] + ["ROLL_WIN_RATE", "REST_DAYS"]
    # PLUS_MINUS (this game's actual point margin, not rolled) is carried through
    # only for the home side — needed to compute Elo's margin-of-victory multiplier.
    meta_cols = ["GAME_ID", "GAME_DATE", "SEASON", "TEAM_ID", "TEAM_ABBREVIATION", "WIN"] + roll_feature_cols

    home = home[meta_cols + ["PLUS_MINUS"]].rename(columns={
        "TEAM_ID": "HOME_TEAM_ID",
        "TEAM_ABBREVIATION": "HOME_TEAM",
        "WIN": "HOME_WIN",
        "PLUS_MINUS": "HOME_GAME_MARGIN",
        **{c: f"HOME_{c}" for c in roll_feature_cols}
    })
    away = away[meta_cols].rename(columns={
        "TEAM_ID": "AWAY_TEAM_ID",
        "TEAM_ABBREVIATION": "AWAY_TEAM",
        "WIN": "AWAY_WIN",
        **{c: f"AWAY_{c}" for c in roll_feature_cols}
    })

    games = home.merge(away, on=["GAME_ID", "GAME_DATE", "SEASON"])

    # Differential features (home minus away) — often more predictive than raw values
    for col in roll_feature_cols:
        games[f"DIFF_{col}"] = games[f"HOME_{col}"] - games[f"AWAY_{col}"]

    games = compute_elo(games)

    # Target: did the home team win?
    games["TARGET"] = games["HOME_WIN"]

    if save:
        games.to_csv(OUTPUT_FEATURES, index=False)
        print(f"  Saved {len(games)} paired games to {OUTPUT_FEATURES}")
    return games


if __name__ == "__main__":
    raw = fetch_game_logs(SEASONS)
    features = build_features(raw)
    print("\nFeature columns:")
    print([c for c in features.columns if c.startswith(("DIFF_", "HOME_ROLL", "AWAY_ROLL"))])
    print("\nDone! Run model/train.py next.")