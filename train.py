"""
train.py
--------
Trains an XGBoost classifier on the feature matrix produced by fetch_data.py.
Outputs a saved model, a SHAP explainer, and prints evaluation metrics.

Usage:
    python model/train.py

Output:
    model/nba_model.joblib     — trained XGBoost pipeline
    model/shap_explainer.joblib — SHAP TreeExplainer
    model/feature_cols.txt      — ordered list of features used
"""

import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score, classification_report
from xgboost import XGBClassifier

# ── Config ────────────────────────────────────────────────────────────────────
FEATURES_PATH = "features.csv"
MODEL_OUT = "nba_model.joblib"
EXPLAINER_OUT = "shap_explainer.joblib"
FEATURE_COLS_OUT = "feature_cols.txt"
SHAP_PLOT_OUT = "shap_summary.png"

# Features fed to the model (differentials + raw rolling stats)
FEATURE_COLS = [
    "DIFF_ROLL_PTS", "DIFF_ROLL_REB", "DIFF_ROLL_AST", "DIFF_ROLL_TOV",
    "DIFF_ROLL_FG_PCT", "DIFF_ROLL_FG3_PCT", "DIFF_ROLL_FT_PCT",
    "DIFF_ROLL_PLUS_MINUS", "DIFF_ROLL_WIN_RATE",
    "DIFF_ROLL_OFF_RATING", "DIFF_ROLL_DEF_RATING", "DIFF_ROLL_PACE",
    "DIFF_ROLL_TS_PCT", "DIFF_ROLL_EFG_PCT", "DIFF_ROLL_OREB_PCT", 
    "DIFF_ROLL_DREB_PCT", "DIFF_ROLL_TM_TOV_PCT",
    "DIFF_REST_DAYS",
    "HOME_ROLL_WIN_RATE", "AWAY_ROLL_WIN_RATE",
    "HOME_REST_DAYS", "AWAY_REST_DAYS",
]
TARGET_COL = "TARGET"

# ── Load data ─────────────────────────────────────────────────────────────────
def load_data():
    df = pd.read_csv(FEATURES_PATH, parse_dates=["GAME_DATE"])
    df = df.sort_values("GAME_DATE").reset_index(drop=True)
    # Drop rows where rolling stats aren't yet populated (first few games of season)
    df = df.dropna(subset=FEATURE_COLS + [TARGET_COL])
    print(f"Loaded {len(df)} games after dropping NaNs.")
    return df

# ── Train ──────────────────────────────────────────────────────────────────────
def train(df: pd.DataFrame):
    X = df[FEATURE_COLS].values
    y = df[TARGET_COL].values.astype(int)

    # Time-series cross-validation (never train on future games)
    tscv = TimeSeriesSplit(n_splits=5)
    cv_accs, cv_aucs = [], []

    xgb_params = dict(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        use_label_encoder=False,
        eval_metric="logloss",
        random_state=42,
    )

    print("\nRunning time-series cross-validation...")
    for fold, (train_idx, val_idx) in enumerate(tscv.split(X), 1):
        X_tr, X_val = X[train_idx], X[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]

        model = XGBClassifier(**xgb_params)
        model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)

        preds = model.predict(X_val)
        probs = model.predict_proba(X_val)[:, 1]
        acc = accuracy_score(y_val, preds)
        auc = roc_auc_score(y_val, probs)
        cv_accs.append(acc)
        cv_aucs.append(auc)
        print(f"  Fold {fold}: Accuracy={acc:.3f}  AUC={auc:.3f}")

    print(f"\nCV Accuracy: {np.mean(cv_accs):.3f} ± {np.std(cv_accs):.3f}")
    print(f"CV AUC:      {np.mean(cv_aucs):.3f} ± {np.std(cv_aucs):.3f}")

    # Final model trained on all data
    print("\nTraining final model on full dataset...")
    final_model = XGBClassifier(**xgb_params)
    final_model.fit(X, y, verbose=False)

    # Final in-sample report (for sanity check only)
    preds_all = final_model.predict(X)
    print("\nFull-data classification report (in-sample, for reference):")
    print(classification_report(y, preds_all, target_names=["Away Win", "Home Win"]))

    return final_model

# ── SHAP ───────────────────────────────────────────────────────────────────────
def compute_shap(model, X: np.ndarray, feature_names: list):
    print("Computing SHAP values...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    plt.figure()
    shap.summary_plot(
        shap_values, X,
        feature_names=feature_names,
        show=False,
        plot_size=(10, 6)
    )
    plt.tight_layout()
    plt.savefig(SHAP_PLOT_OUT, dpi=150)
    plt.close()
    print(f"  SHAP summary plot saved to {SHAP_PLOT_OUT}")

    return explainer

# ── Save ───────────────────────────────────────────────────────────────────────
def save_artifacts(model, explainer):
    joblib.dump(model, MODEL_OUT)
    joblib.dump(explainer, EXPLAINER_OUT)
    with open(FEATURE_COLS_OUT, "w") as f:
        f.write("\n".join(FEATURE_COLS))
    print(f"\nSaved model → {MODEL_OUT}")
    print(f"Saved SHAP explainer → {EXPLAINER_OUT}")
    print(f"Saved feature list → {FEATURE_COLS_OUT}")


if __name__ == "__main__":
    df = load_data()
    model = train(df)
    explainer = compute_shap(model, df[FEATURE_COLS].values, FEATURE_COLS)
    save_artifacts(model, explainer)
    print("\nDone! Run app/app.py to launch the Streamlit UI.")