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

| Metric | XGBoost | Logistic Regression | Ensemble (avg of both) |
|---|---|---|---|
| CV Accuracy | ~64.6% ± 3.2% | ~65.2% ± 3.3% | ~65.3% ± 3.5% |
| CV AUC | ~0.69 ± 0.04 | ~0.71 ± 0.04 | ~0.71 ± 0.04 |
| CV Log Loss | ~0.63 | ~0.62 | ~0.62 |
| Baseline (always pick home team) | ~55.7% | — | — |

> NBA games are inherently hard to predict — Vegas sits around 67–70% on heavy favorites. XGBoost is regularized (depth/estimator limits, L1/L2, early stopping) and clears the home-team baseline by ~9 points. Logistic regression and a simple probability-averaging ensemble are run through the same CV folds as comparison points; the ensemble's gain over solo XGBoost was small enough (<1 point) that it isn't worth the extra serving complexity, so **XGBoost remains the production model**. The single biggest lever was adding Elo ratings (`DIFF_ELO`) as a feature — it alone accounted for most of the jump from ~61% to ~65% CV accuracy over the previous rolling-stats-only feature set. Re-run `python train.py` to reproduce these numbers.

---

## Project Structure

```
nba-predictor/
├── data/
│   ├── fetch_data.py       # Pull game logs + build feature matrix
│   ├── games_raw.csv       # Raw game logs (generated)
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

### Model
XGBoost was chosen for its performance on tabular data, native feature importance, and compatibility with SHAP. Time-series cross-validation is used during evaluation to ensure no future games leak into training folds. The model is regularized (shallow trees, L1/L2 penalties, `min_child_weight`) and uses early stopping per CV fold to keep it from overfitting the training set — a plain logistic regression, and a simple probability-averaging ensemble of the two, are run through the same folds as comparison points on every training run (see [Results](#results)).

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

- Pull injury reports and factor out key players (rolling stats and Elo can't see who's out tonight — likely the next-biggest accuracy lever)
- Predict point spread instead of win/loss
- Add more historical seasons for a larger training set
- Deploy to Streamlit Cloud

---

## License

MIT