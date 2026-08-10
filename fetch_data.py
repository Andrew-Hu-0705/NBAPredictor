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
import numpy as np
import pandas as pd
from nba_api.stats.endpoints import leaguegamelog, teamgamelogs, playergamelogs
from nba_api.stats.static import teams

# ── Config ────────────────────────────────────────────────────────────────────
# Multiple seasons give the model more rows to learn from than a single
# season (~1200 games) can support. Rolling/rest stats are computed
# separately per season below so nothing leaks across the off-season gap.
SEASONS = ["2022-23", "2023-24", "2024-25"]
ROLLING_WINDOW = 10         # rolling average over last N games — tuned via CV, see train.py notes
OUTPUT_RAW = "games_raw.csv"
OUTPUT_PLAYER_RAW = "player_logs_raw.csv"
OUTPUT_FEATURES = "features.csv"

# "Missing rotation minutes" config. nba_api doesn't expose a live/historical
# injury report, so this is inferred from who actually suited up: a player
# counts as part of a team's rotation if they appeared in at least
# ROTATION_MIN_APPEARANCES of the team's last ROTATION_LOOKBACK_GAMES games,
# capped at the ROTATION_SIZE players with the most minutes.
ROTATION_LOOKBACK_GAMES = 5
ROTATION_MIN_APPEARANCES = 3
ROTATION_SIZE = 8

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

def fetch_player_logs(seasons: list) -> pd.DataFrame:
    """Player-level game logs (who actually played, and how many minutes),
    used to infer which of a team's rotation players are missing tonight."""
    dfs = []
    for i, season in enumerate(seasons):
        print(f"Fetching player game logs for {season}...")
        log = playergamelogs.PlayerGameLogs(season_nullable=season, season_type_nullable="Regular Season")
        d = log.get_data_frames()[0]
        d["SEASON"] = season
        dfs.append(d)
        if i < len(seasons) - 1:
            time.sleep(1)  # be kind to the API between seasons
    df = pd.concat(dfs, ignore_index=True)
    df = df[["PLAYER_ID", "PLAYER_NAME", "TEAM_ID", "GAME_ID", "GAME_DATE", "SEASON", "MIN"]]

    df.to_csv(OUTPUT_PLAYER_RAW, index=False)
    print(f"  Saved {len(df)} player-game rows across {len(seasons)} season(s) to {OUTPUT_PLAYER_RAW}")
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

# ── Missing rotation minutes ─────────────────────────────────────────────────
def compute_missing_rotation_minutes(team_games: pd.DataFrame, player_logs: pd.DataFrame) -> pd.Series:
    """
    For each row in team_games (one row per team per game), estimates how many
    minutes of that team's recent rotation are absent from tonight's box score —
    a proxy for "key players out" (injury, rest, trade) that neither the rolling
    box-score stats nor Elo can see, since those only reflect performance from
    games the team's *actual* lineup played. Uses only games strictly before the
    one being scored, so it can't leak tonight's box score.
    """
    game_players = (
        player_logs.groupby(["TEAM_ID", "GAME_ID"])
        .apply(lambda g: dict(zip(g["PLAYER_ID"], g["MIN"])))
        .to_dict()
    )

    missing_minutes = {}
    for (team_id, season), team_rows in team_games.groupby(["TEAM_ID", "SEASON"]):
        recent_game_ids = []  # sliding window of this team's own past games, this season
        for idx, row in team_rows.sort_values("GAME_DATE").iterrows():
            appearances = {}
            for gid in recent_game_ids[-ROTATION_LOOKBACK_GAMES:]:
                for pid, mins in game_players.get((team_id, gid), {}).items():
                    appearances.setdefault(pid, []).append(mins)
            rotation = {
                pid: np.mean(mins_list) for pid, mins_list in appearances.items()
                if len(mins_list) >= ROTATION_MIN_APPEARANCES
            }
            rotation = dict(sorted(rotation.items(), key=lambda kv: -kv[1])[:ROTATION_SIZE])

            present = game_players.get((team_id, row["GAME_ID"]), {})
            missing_minutes[idx] = sum(avg_min for pid, avg_min in rotation.items() if pid not in present)

            recent_game_ids.append(row["GAME_ID"])

    return pd.Series(missing_minutes).reindex(team_games.index).fillna(0.0)

# ── Feature engineering ───────────────────────────────────────────────────────
def build_features(df: pd.DataFrame, player_logs: pd.DataFrame, window: int = ROLLING_WINDOW, save: bool = True) -> pd.DataFrame:
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

    # Strength-of-schedule adjustment for OFF/DEF rating. Raw OFF_RATING/DEF_RATING
    # don't account for opponent quality — scoring well against a bad defense looks
    # the same as scoring well against a great one. Adjust each game's rating by how
    # far the opponent's own pre-game rolling rating deviates from the league
    # average, then roll THAT. Uses only the opponent's rolling stats from strictly
    # before this game (same leak-free shift(1) pattern as everything else here).
    opp_ratings = df[["GAME_ID", "TEAM_ID", "ROLL_OFF_RATING", "ROLL_DEF_RATING"]].rename(columns={
        "TEAM_ID": "OPP_TEAM_ID", "ROLL_OFF_RATING": "OPP_ROLL_OFF_RATING", "ROLL_DEF_RATING": "OPP_ROLL_DEF_RATING"
    })
    df = df.merge(opp_ratings, on="GAME_ID")
    df = df[df["TEAM_ID"] != df["OPP_TEAM_ID"]].reset_index(drop=True)

    league_avg_off = df["OFF_RATING"].mean()
    league_avg_def = df["DEF_RATING"].mean()
    df["OFF_RATING_ADJ"] = df["OFF_RATING"] - (df["OPP_ROLL_DEF_RATING"] - league_avg_def)
    df["DEF_RATING_ADJ"] = df["DEF_RATING"] - (df["OPP_ROLL_OFF_RATING"] - league_avg_off)

    for col in ["OFF_RATING_ADJ", "DEF_RATING_ADJ"]:
        df[f"ROLL_{col}"] = (
            df.groupby(["TEAM_ID", "SEASON"])[col]
            .transform(lambda x: x.shift(1).rolling(window, min_periods=1).mean())
        )

    # Missing rotation minutes — see compute_missing_rotation_minutes docstring
    df["MISSING_ROTATION_MIN"] = compute_missing_rotation_minutes(df, player_logs)

    # ── Pair home and away teams per game ────────────────────────────────────
    # Each GAME_ID appears twice (once per team). We merge them into one row.
    home = df[df["HOME"] == 1].copy()
    away = df[df["HOME"] == 0].copy()

    roll_feature_cols = (
        [f"ROLL_{c}" for c in stat_cols]
        + ["ROLL_WIN_RATE", "REST_DAYS", "MISSING_ROTATION_MIN", "ROLL_OFF_RATING_ADJ", "ROLL_DEF_RATING_ADJ"]
    )
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
    player_logs = fetch_player_logs(SEASONS)
    features = build_features(raw, player_logs)
    print("\nFeature columns:")
    print([c for c in features.columns if c.startswith(("DIFF_", "HOME_ROLL", "AWAY_ROLL"))])
    print("\nDone! Run model/train.py next.")