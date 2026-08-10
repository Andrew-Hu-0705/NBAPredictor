# 🏀 NBA Game Outcome Predictor

An ML-powered NBA game predictor built with XGBoost and SHAP explainability. Predicts win probabilities for any matchup using rolling team performance stats pulled live from the NBA API.

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![XGBoost](https://img.shields.io/badge/Model-XGBoost-orange)
![Streamlit](https://img.shields.io/badge/UI-Streamlit-red)

---

## Features

- **Live data** — pulls current season game logs directly from `nba_api` (no API key needed)
- **Feature engineering** — rolling averages, rest days, home/away splits, differential features
- **XGBoost classifier** — trained with time-series cross-validation to prevent data leakage
- **SHAP explainability** — every prediction comes with a breakdown of which factors drove it
- **Streamlit UI** — interactive team picker with win probability bar chart and SHAP waterfall plot

---

## Results

Measured on 3 seasons of data (2022-23 through 2024-25, ~3,640 games) with 5-fold `TimeSeriesSplit` CV:

| Metric | XGBoost | XGBoost+sigmoid | XGBoost+isotonic | Logistic Regression | Ensemble |
|---|---|---|---|---|---|
| CV Accuracy | ~64.1% ± 3.0% | ~64.8% ± 2.7% | ~64.4% ± 3.3% | ~66.0% ± 2.9% | ~65.7% ± 2.4% |
| CV AUC | ~0.70 ± 0.04 | ~0.70 ± 0.04 | ~0.68 ± 0.05 | ~0.71 ± 0.04 | ~0.71 ± 0.04 |
| CV Log Loss | ~0.63 | ~0.63 | ~0.89 | ~0.62 | ~0.62 |
| Baseline (always pick home team) | ~55.7% | — | — | — | — |

> NBA games are inherently hard to predict — Vegas sits around 67–70% on heavy favorites. XGBoost is regularized (depth/estimator limits, L1/L2, early stopping) and clears the home-team baseline by ~9 points. Logistic regression and a simple probability-averaging ensemble are run through the same CV folds as comparison points; the ensemble's gain over solo XGBoost was small enough that it isn't worth the extra serving complexity, so **XGBoost remains the production model**. The single biggest lever was adding Elo ratings (`DIFF_ELO`) as a feature — it alone accounted for most of the jump from ~61% to ~65% CV accuracy over the earlier rolling-stats-only feature set. `HOME_/AWAY_MISSING_ROTATION_MIN` moved accuracy by roughly +0.1 point — within CV noise, not the leap the name might suggest. Strength-of-schedule adjustment (below) was roughly neutral for XGBoost specifically but gave logistic regression a genuine, if small, boost. Sigmoid probability calibration is applied to the saved model since it edged out the raw model on CV log loss; isotonic calibration was tested too and rejected — it overfits badly with the amount of held-out data available per fold (as few as ~90 games in fold 1) and made log loss much worse. `train.py` re-runs this comparison and picks automatically each time. Re-run `python train.py` to reproduce these numbers.

---

## Project Structure

```
nba-predictor/
├── data/
│   ├── fetch_data.py       # Pull game logs + build feature matrix
│   ├── games_raw.csv       # Raw team game logs (generated)
│   ├── player_logs_raw.csv # Raw player game logs, for missing-rotation-minutes (generated)
│   └── features.csv        # Feature-engineered dataset (generated)
├── model/
│   ├── train.py            # Train XGBoost, evaluate, save artifacts
│   ├── nba_model.joblib    # Saved model (generated)
│   ├── shap_explainer.joblib
│   ├── feature_cols.txt
│   └── shap_summary.png    # SHAP summary plot (generated)
├── app/
│   └── app.py              # Streamlit UI
├── predict.py              # Prediction logic + CLI
├── requirements.txt
└── README.md
```

---

## Quickstart

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Fetch data and build features
python data/fetch_data.py

# 3. Train the model
python model/train.py

# 4. Predict from the CLI
python predict.py --home BOS --away MIA

# 5. Launch the Streamlit app
streamlit run app/app.py
```

---

## How It Works

### Data
Game logs are fetched via `nba_api` for one or more seasons (configured as `SEASONS` in `fetch_data.py`, default: 2022-23 through 2024-25). Each team's last N games (default: 10, tuned via CV — see below) are used to compute rolling averages for points, rebounds, assists, turnovers, shooting percentages, plus/minus, and win rate. Rest days between games are also included. Rolling stats and rest days are computed per-season so form doesn't carry over across an off-season.

### Features
Rather than using raw stats, the model is fed **differential features** (home team stat minus away team stat), which are more predictive than raw values and reduce dimensionality. Rolling stats are always computed from games *prior* to the current one to prevent data leakage. A handful of candidate features (e.g. raw points, raw turnovers, FG%, EFG%) were dropped from the final feature set for being highly redundant (>0.75 correlation) with another feature already kept — see the comment above `FEATURE_COLS` in `train.py` for the full list and reasoning.

**Elo ratings** (`DIFF_ELO`) are the strongest feature in the model by a wide margin. `compute_elo()` in `fetch_data.py` maintains a running per-team rating updated after every game (margin-of-victory-weighted, home-court adjusted, partially reverting to the mean between seasons), and the pre-game rating differential is fed to the model — pre-game, so it can't leak the outcome it's predicting.

The rolling window size (5 vs. 10 vs. 20 games, etc.) was swept via the same CV setup and turned out to barely matter once Elo was added (accuracy varied by about 1 point across window sizes 3–20) — 10 was picked as a reasonable middle value, not because it was a clear CV winner.

**Missing rotation minutes** (`HOME_/AWAY_MISSING_ROTATION_MIN`) is a proxy for "who's out tonight." `nba_api` doesn't expose a live or historical injury report, so `compute_missing_rotation_minutes()` in `fetch_data.py` infers it instead: a player counts as part of a team's rotation if they appeared in most of that team's last 5 games, and "missing" if they don't show up in this game's box score at all (whether from injury, rest, or a trade). The value is the recent average minutes of whoever's missing, so losing a 35-min/game starter counts for more than losing a bench player. It's leak-free for training (each row only uses that team's own games strictly before it), and the values check out — they spike hardest in the final week of each season, when contenders rest their starters. It moved CV accuracy by only about +0.1 point, well within noise — smaller than expected. One likely reason: the CLI/app can only use a team's *most recent played game* as of prediction time, not tonight's actual box score, so if that last game happened to be a end-of-season rest game, the predictor can carry a stale "missing rotation" signal into a hypothetical matchup where those players would actually be available. A live injury-report feed would fix this; this box-score-inferred version is a lagging indicator, not a live one.

**Strength-of-schedule adjustment** (`DIFF_ROLL_OFF_RATING_ADJ` / `DIFF_ROLL_DEF_RATING_ADJ`) replaces the raw rolling offensive/defensive rating features. A team's raw `OFF_RATING` looks the same whether it came against a great defense or a terrible one; `build_features()` adjusts each game's rating by how far that opponent's own pre-game rolling rating deviates from the league average (using only the opponent's rating from strictly before this game, so it can't leak), then rolls the adjusted value the same way as everything else. Measured effect: roughly neutral for XGBoost (trees can already learn opponent-conditioned interactions from the raw stats plus Elo), but a genuine small improvement for the logistic regression baseline, which can't capture that nonlinearity on its own.

### Model
XGBoost was chosen for its performance on tabular data, native feature importance, and compatibility with SHAP. Time-series cross-validation is used during evaluation to ensure no future games leak into training folds. The model is regularized (shallow trees, L1/L2 penalties, `min_child_weight`); early stopping and probability calibration are both fit against a held-out slice of the most recent games *within* each fold's training data — never the fold's evaluation set — so neither the tree count nor the probability calibration can peek at the data being scored. Two calibration methods (Platt/sigmoid, isotonic) are tried and compared against the raw model on every run; whichever wins on CV log loss is used, and calibration is skipped entirely if neither beats the raw model. A plain logistic regression, and a simple probability-averaging ensemble, are run through the same folds as comparison points (see [Results](#results)).

### Explainability
SHAP (SHapley Additive exPlanations) TreeExplainer is used to produce per-prediction breakdowns. For every matchup, you can see exactly which features pushed the model toward a home win or away win — e.g., "the home team's 4-day rest advantage was the biggest factor."

---

## Tech Stack

| Tool | Purpose |
|---|---|
| `nba_api` | Live NBA data |
| `pandas` | Data wrangling |
| `xgboost` | Classification model |
| `scikit-learn` | Pipelines, CV, metrics |
| `shap` | Model explainability |
| `streamlit` | Interactive UI |
| `matplotlib` | SHAP plots |

---

## Ideas for Extension

- Wire in a real (live) injury report source — the current `MISSING_ROTATION_MIN` feature is inferred from box scores after the fact, so it lags at actual prediction time; a live feed would make it much more useful for the CLI/app
- Predict point spread instead of win/loss
- Add more historical seasons for a larger training set
- Deploy to Streamlit Cloud

---

## License

MIT