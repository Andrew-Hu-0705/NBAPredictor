"""
train.py
--------
Trains an XGBoost classifier on the feature matrix produced by fetch_data.py.
Outputs a saved model, a SHAP explainer, and prints evaluation metrics
(including a logistic-regression baseline run through the same CV folds,
for comparison).

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
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, log_loss, roc_auc_score, brier_score_loss, classification_report,
)
from xgboost import XGBClassifier

# ── Config ────────────────────────────────────────────────────────────────────
FEATURES_PATH = "features.csv"
MODEL_OUT = "nba_model.joblib"
EXPLAINER_OUT = "shap_explainer.joblib"
FEATURE_COLS_OUT = "feature_cols.txt"
SHAP_PLOT_OUT = "shap_summary.png"

# Features fed to the model (differentials + raw rolling stats).
# A number of the original candidate features were dropped for being
# redundant with another feature already in this list (|corr| > 0.75, or in
# the case of DIFF_ROLL_WIN_RATE / DIFF_REST_DAYS, an exact linear
# combination of the HOME_/AWAY_ columns already included below):
#   DIFF_ROLL_PTS      -> redundant with DIFF_ROLL_OFF_RATING (pace-adjusted)
#   DIFF_ROLL_FG_PCT    -> redundant with DIFF_ROLL_TS_PCT (also captures FT/3PT)
#   DIFF_ROLL_EFG_PCT   -> redundant with DIFF_ROLL_TS_PCT
#   DIFF_ROLL_TOV       -> redundant with DIFF_ROLL_TM_TOV_PCT (possession-normalized)
#   DIFF_ROLL_WIN_RATE  -> = HOME_ROLL_WIN_RATE - AWAY_ROLL_WIN_RATE, both kept below
#   DIFF_REST_DAYS      -> = HOME_REST_DAYS - AWAY_REST_DAYS, both kept below
FEATURE_COLS = [
    "DIFF_ROLL_REB", "DIFF_ROLL_AST",
    "DIFF_ROLL_FG3_PCT", "DIFF_ROLL_FT_PCT",
    "DIFF_ROLL_PLUS_MINUS",
    "DIFF_ROLL_OFF_RATING", "DIFF_ROLL_DEF_RATING", "DIFF_ROLL_PACE",
    "DIFF_ROLL_TS_PCT", "DIFF_ROLL_OREB_PCT",
    "DIFF_ROLL_DREB_PCT", "DIFF_ROLL_TM_TOV_PCT",
    "HOME_ROLL_WIN_RATE", "AWAY_ROLL_WIN_RATE",
    "HOME_REST_DAYS", "AWAY_REST_DAYS",
    "DIFF_ELO",   # pre-game Elo differential, computed in fetch_data.py — by far the strongest single feature
]
TARGET_COL = "TARGET"

XGB_PARAMS = dict(
    n_estimators=500,       # upper bound; early stopping picks the real number per fold
    max_depth=3,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=5,
    reg_alpha=1.0,
    reg_lambda=2.0,
    eval_metric="logloss",
    early_stopping_rounds=30,
    random_state=42,
)

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
    xgb_accs, xgb_aucs, xgb_loglosses, xgb_briers, best_iters = [], [], [], [], []
    lr_accs, lr_aucs, lr_loglosses = [], [], []
    ens_accs, ens_aucs, ens_loglosses = [], [], []

    print("\nRunning time-series cross-validation (XGBoost vs. logistic regression baseline vs. ensemble)...")
    for fold, (train_idx, val_idx) in enumerate(tscv.split(X), 1):
        X_tr, X_val = X[train_idx], X[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]

        # XGBoost, with early stopping against this fold's validation split
        model = XGBClassifier(**XGB_PARAMS)
        model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
        best_iters.append(model.best_iteration)

        probs = model.predict_proba(X_val)[:, 1]
        acc = accuracy_score(y_val, probs > 0.5)
        auc = roc_auc_score(y_val, probs)
        ll = log_loss(y_val, probs)
        brier = brier_score_loss(y_val, probs)
        xgb_accs.append(acc); xgb_aucs.append(auc); xgb_loglosses.append(ll); xgb_briers.append(brier)

        # Logistic regression baseline, same fold
        scaler = StandardScaler().fit(X_tr)
        lr = LogisticRegression(max_iter=1000).fit(scaler.transform(X_tr), y_tr)
        lr_probs = lr.predict_proba(scaler.transform(X_val))[:, 1]
        lr_acc = accuracy_score(y_val, lr_probs > 0.5)
        lr_auc = roc_auc_score(y_val, lr_probs)
        lr_ll = log_loss(y_val, lr_probs)
        lr_accs.append(lr_acc); lr_aucs.append(lr_auc); lr_loglosses.append(lr_ll)

        # Ensemble: simple average of the two models' probabilities
        ens_probs = (probs + lr_probs) / 2
        ens_acc = accuracy_score(y_val, ens_probs > 0.5)
        ens_auc = roc_auc_score(y_val, ens_probs)
        ens_ll = log_loss(y_val, ens_probs)
        ens_accs.append(ens_acc); ens_aucs.append(ens_auc); ens_loglosses.append(ens_ll)

        print(f"  Fold {fold} [n_val={len(val_idx)}]")
        print(f"    XGBoost  : acc={acc:.3f}  auc={auc:.3f}  logloss={ll:.3f}  brier={brier:.3f}  best_iter={model.best_iteration}")
        print(f"    LogReg   : acc={lr_acc:.3f}  auc={lr_auc:.3f}  logloss={lr_ll:.3f}")
        print(f"    Ensemble : acc={ens_acc:.3f}  auc={ens_auc:.3f}  logloss={ens_ll:.3f}")

    print(f"\n{'Model':<10} {'Accuracy':<18} {'AUC':<18} {'LogLoss':<10}")
    print(f"{'XGBoost':<10} {np.mean(xgb_accs):.3f} ± {np.std(xgb_accs):.3f}   {np.mean(xgb_aucs):.3f} ± {np.std(xgb_aucs):.3f}   {np.mean(xgb_loglosses):.3f}")
    print(f"{'LogReg':<10} {np.mean(lr_accs):.3f} ± {np.std(lr_accs):.3f}   {np.mean(lr_aucs):.3f} ± {np.std(lr_aucs):.3f}   {np.mean(lr_loglosses):.3f}")
    print(f"{'Ensemble':<10} {np.mean(ens_accs):.3f} ± {np.std(ens_accs):.3f}   {np.mean(ens_aucs):.3f} ± {np.std(ens_aucs):.3f}   {np.mean(ens_loglosses):.3f}")

    # Final model trained on all data. Use the average best_iteration found
    # via early stopping across folds instead of re-enabling early stopping
    # on the full dataset (which would need its own held-out slice and just
    # re-introduce the leakage/overfitting risk this was meant to avoid).
    final_n_estimators = max(10, int(round(np.mean(best_iters))))
    print(f"\nTraining final model on full dataset (n_estimators={final_n_estimators}, from avg CV best_iteration)...")
    final_params = {**XGB_PARAMS, "n_estimators": final_n_estimators}
    final_params.pop("early_stopping_rounds")
    final_model = XGBClassifier(**final_params)
    final_model.fit(X, y, verbose=False)

    # Final in-sample report (for sanity check only — expect this to look
    # much closer to the CV numbers now that capacity is regularized)
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
