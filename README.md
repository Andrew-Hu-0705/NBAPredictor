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

| Metric | Score |
|---|---|
| CV Accuracy (5-fold TimeSeriesSplit) | ~64% |
| CV AUC | ~0.69 |
| Baseline (always pick home team) | ~58% |

> NBA games are inherently hard to predict — Vegas sits around 67–70% on heavy favorites. A 64% CV accuracy meaningfully beats the home-team baseline.

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
Game logs are fetched via `nba_api` for the current season. Each team's last N games (default: 5) are used to compute rolling averages for points, rebounds, assists, turnovers, shooting percentages, plus/minus, and win rate. Rest days between games are also included.

### Features
Rather than using raw stats, the model is fed **differential features** (home team stat minus away team stat), which are more predictive than raw values and reduce dimensionality. Rolling stats are always computed from games *prior* to the current one to prevent data leakage.

### Model
XGBoost was chosen for its performance on tabular data, native feature importance, and compatibility with SHAP. Time-series cross-validation is used during evaluation to ensure no future games leak into training folds.

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

- Add Elo ratings as a feature
- Pull injury reports and factor out key players
- Predict point spread instead of win/loss
- Add historical seasons for a larger training set
- Deploy to Streamlit Cloud

---

## License

MIT