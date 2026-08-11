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

import json
import os

import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
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
FEATURE_IMPORTANCE_OUT = "feature_importance.json"
DRIFT_REFERENCE_OUT = "drift_reference.json"
DRIFT_N_BINS = 10

MLFLOW_EXPERIMENT = "nba-game-predictor"
REGISTERED_MODEL_NAME = "nba-predictor"

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
#   DIFF_MISSING_ROTATION_MIN -> = HOME_MISSING_ROTATION_MIN - AWAY_MISSING_ROTATION_MIN, both kept below
#   DIFF_ROLL_OFF_RATING / DIFF_ROLL_DEF_RATING -> superseded by the opponent-adjusted
#     versions below (same stat, adjusted for strength of schedule — see fetch_data.py)
FEATURE_COLS = [
    "DIFF_ROLL_REB", "DIFF_ROLL_AST",
    "DIFF_ROLL_FG3_PCT", "DIFF_ROLL_FT_PCT",
    "DIFF_ROLL_PLUS_MINUS",
    "DIFF_ROLL_OFF_RATING_ADJ", "DIFF_ROLL_DEF_RATING_ADJ", "DIFF_ROLL_PACE",
    "DIFF_ROLL_TS_PCT", "DIFF_ROLL_OREB_PCT",
    "DIFF_ROLL_DREB_PCT", "DIFF_ROLL_TM_TOV_PCT",
    "HOME_ROLL_WIN_RATE", "AWAY_ROLL_WIN_RATE",
    "HOME_REST_DAYS", "AWAY_REST_DAYS",
    "DIFF_ELO",   # pre-game Elo differential, computed in fetch_data.py — by far the strongest single feature
    # Minutes of each team's recent rotation missing from tonight's box score (injury/rest/trade
    # proxy — see compute_missing_rotation_minutes in fetch_data.py). Neither rolling box-score
    # stats nor Elo can see that tonight's star is out; this is the closest proxy available
    # without a live injury-report data source.
    "HOME_MISSING_ROTATION_MIN", "AWAY_MISSING_ROTATION_MIN",
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

# Fraction of each fold's training data (most recent games) held out purely for
# early stopping + probability calibration — never used to fit tree splits.
CALIB_FRAC = 0.15
MIN_CALIB = 60

# ── Load data ─────────────────────────────────────────────────────────────────
def load_data():
    df = pd.read_csv(FEATURES_PATH, parse_dates=["GAME_DATE"])
    df = df.sort_values("GAME_DATE").reset_index(drop=True)
    # Drop rows where rolling stats aren't yet populated (first few games of season)
    df = df.dropna(subset=FEATURE_COLS + [TARGET_COL])
    print(f"Loaded {len(df)} games after dropping NaNs.")
    return df

def _split_fit_calib(idx: np.ndarray):
    """Carves the most recent slice off a chronologically-ordered index array,
    for early stopping / probability calibration — never used to fit tree splits."""
    n_calib = max(MIN_CALIB, int(len(idx) * CALIB_FRAC))
    return idx[:-n_calib], idx[-n_calib:]

# ── Train ──────────────────────────────────────────────────────────────────────
def train(df: pd.DataFrame):
    X = df[FEATURE_COLS].values
    y = df[TARGET_COL].values.astype(int)

    # Time-series cross-validation (never train on future games)
    tscv = TimeSeriesSplit(n_splits=5)
    xgb_accs, xgb_aucs, xgb_loglosses, xgb_briers, best_iters = [], [], [], [], []
    sig_accs, sig_aucs, sig_loglosses = [], [], []
    iso_accs, iso_aucs, iso_loglosses = [], [], []
    lr_accs, lr_aucs, lr_loglosses = [], [], []
    ens_accs, ens_aucs, ens_loglosses = [], [], []

    print("\nRunning time-series cross-validation (XGBoost, calibrated variants, logistic regression, ensemble)...")
    for fold, (train_idx, val_idx) in enumerate(tscv.split(X), 1):
        X_val, y_val = X[val_idx], y[val_idx]

        # Within this fold's training data, carve off the most recent slice
        # purely for early stopping + calibration — never seen by tree fitting.
        fit_idx, calib_idx = _split_fit_calib(train_idx)
        X_fit, y_fit = X[fit_idx], y[fit_idx]
        X_calib, y_calib = X[calib_idx], y[calib_idx]

        model = XGBClassifier(**XGB_PARAMS)
        model.fit(X_fit, y_fit, eval_set=[(X_calib, y_calib)], verbose=False)
        best_iters.append(model.best_iteration)

        probs = model.predict_proba(X_val)[:, 1]
        acc = accuracy_score(y_val, probs > 0.5)
        auc = roc_auc_score(y_val, probs)
        ll = log_loss(y_val, probs)
        brier = brier_score_loss(y_val, probs)
        xgb_accs.append(acc); xgb_aucs.append(auc); xgb_loglosses.append(ll); xgb_briers.append(brier)

        # Calibrate the same fitted model on the held-out calibration slice —
        # two standard methods, compared against each other and the raw model.
        sig_model = CalibratedClassifierCV(model, method="sigmoid", cv="prefit").fit(X_calib, y_calib)
        sig_probs = sig_model.predict_proba(X_val)[:, 1]
        sig_accs.append(accuracy_score(y_val, sig_probs > 0.5))
        sig_aucs.append(roc_auc_score(y_val, sig_probs))
        sig_loglosses.append(log_loss(y_val, sig_probs))

        iso_model = CalibratedClassifierCV(model, method="isotonic", cv="prefit").fit(X_calib, y_calib)
        iso_probs = iso_model.predict_proba(X_val)[:, 1]
        iso_accs.append(accuracy_score(y_val, iso_probs > 0.5))
        iso_aucs.append(roc_auc_score(y_val, iso_probs))
        iso_loglosses.append(log_loss(y_val, iso_probs))

        # Logistic regression baseline, fit on the full fold training set (it
        # doesn't need a calibration slice — it's already well-calibrated by
        # construction, and doesn't use early stopping)
        X_tr, y_tr = X[train_idx], y[train_idx]
        scaler = StandardScaler().fit(X_tr)
        lr = LogisticRegression(max_iter=1000).fit(scaler.transform(X_tr), y_tr)
        lr_probs = lr.predict_proba(scaler.transform(X_val))[:, 1]
        lr_acc = accuracy_score(y_val, lr_probs > 0.5)
        lr_auc = roc_auc_score(y_val, lr_probs)
        lr_ll = log_loss(y_val, lr_probs)
        lr_accs.append(lr_acc); lr_aucs.append(lr_auc); lr_loglosses.append(lr_ll)

        # Ensemble: simple average of raw XGBoost + logistic regression probabilities
        ens_probs = (probs + lr_probs) / 2
        ens_acc = accuracy_score(y_val, ens_probs > 0.5)
        ens_auc = roc_auc_score(y_val, ens_probs)
        ens_ll = log_loss(y_val, ens_probs)
        ens_accs.append(ens_acc); ens_aucs.append(ens_auc); ens_loglosses.append(ens_ll)

        print(f"  Fold {fold} [n_fit={len(fit_idx)} n_calib={len(calib_idx)} n_val={len(val_idx)}]")
        print(f"    XGBoost         : acc={acc:.3f}  auc={auc:.3f}  logloss={ll:.3f}  brier={brier:.3f}  best_iter={model.best_iteration}")
        print(f"    XGBoost+sigmoid : acc={sig_accs[-1]:.3f}  auc={sig_aucs[-1]:.3f}  logloss={sig_loglosses[-1]:.3f}")
        print(f"    XGBoost+isotonic: acc={iso_accs[-1]:.3f}  auc={iso_aucs[-1]:.3f}  logloss={iso_loglosses[-1]:.3f}")
        print(f"    LogReg          : acc={lr_acc:.3f}  auc={lr_auc:.3f}  logloss={lr_ll:.3f}")
        print(f"    Ensemble        : acc={ens_acc:.3f}  auc={ens_auc:.3f}  logloss={ens_ll:.3f}")

    print(f"\n{'Model':<18} {'Accuracy':<18} {'AUC':<18} {'LogLoss':<10}")
    print(f"{'XGBoost':<18} {np.mean(xgb_accs):.3f} ± {np.std(xgb_accs):.3f}   {np.mean(xgb_aucs):.3f} ± {np.std(xgb_aucs):.3f}   {np.mean(xgb_loglosses):.3f}")
    print(f"{'XGBoost+sigmoid':<18} {np.mean(sig_accs):.3f} ± {np.std(sig_accs):.3f}   {np.mean(sig_aucs):.3f} ± {np.std(sig_aucs):.3f}   {np.mean(sig_loglosses):.3f}")
    print(f"{'XGBoost+isotonic':<18} {np.mean(iso_accs):.3f} ± {np.std(iso_accs):.3f}   {np.mean(iso_aucs):.3f} ± {np.std(iso_aucs):.3f}   {np.mean(iso_loglosses):.3f}")
    print(f"{'LogReg':<18} {np.mean(lr_accs):.3f} ± {np.std(lr_accs):.3f}   {np.mean(lr_aucs):.3f} ± {np.std(lr_aucs):.3f}   {np.mean(lr_loglosses):.3f}")
    print(f"{'Ensemble':<18} {np.mean(ens_accs):.3f} ± {np.std(ens_accs):.3f}   {np.mean(ens_aucs):.3f} ± {np.std(ens_aucs):.3f}   {np.mean(ens_loglosses):.3f}")

    # Pick whichever calibration method scored the lower mean CV log loss, but
    # only use it if it actually beats the raw model — a bad calibration slice
    # can make things worse, especially in early folds with few games to calibrate on.
    calib_choice = "sigmoid" if np.mean(sig_loglosses) <= np.mean(iso_loglosses) else "isotonic"
    calib_ll = min(np.mean(sig_loglosses), np.mean(iso_loglosses))
    use_calibration = calib_ll < np.mean(xgb_loglosses)
    print(f"\nCalibration: {calib_choice} scored best (logloss={calib_ll:.3f} vs raw {np.mean(xgb_loglosses):.3f}); "
          f"{'applying' if use_calibration else 'skipping — raw model is already as good or better'} it in the saved model.")

    # Final model: hold out the most recent slice of the full dataset the same
    # way each fold did, both for early stopping and (if it won above) calibration.
    n = len(X)
    fit_idx, calib_idx = _split_fit_calib(np.arange(n))
    print(f"\nTraining final model on full dataset (n_fit={len(fit_idx)}, n_calib={len(calib_idx)}, early-stopped against the held-out slice)...")
    final_model = XGBClassifier(**XGB_PARAMS)
    final_model.fit(X[fit_idx], y[fit_idx], eval_set=[(X[calib_idx], y[calib_idx])], verbose=False)
    print(f"  best_iteration={final_model.best_iteration}")

    if use_calibration:
        production_model = CalibratedClassifierCV(final_model, method=calib_choice, cv="prefit")
        production_model.fit(X[calib_idx], y[calib_idx])
    else:
        production_model = final_model

    # Final in-sample report (for sanity check only — expect this to look
    # much closer to the CV numbers now that capacity is regularized)
    preds_all = final_model.predict(X)
    print("\nFull-data classification report (in-sample, for reference):")
    print(classification_report(y, preds_all, target_names=["Away Win", "Home Win"]))

    if use_calibration and calib_choice == "sigmoid":
        prod_accs, prod_aucs = sig_accs, sig_aucs
    elif use_calibration:
        prod_accs, prod_aucs = iso_accs, iso_aucs
    else:
        prod_accs, prod_aucs = xgb_accs, xgb_aucs

    metrics = {
        "n_games": float(n),
        "cv_xgb_accuracy_mean": float(np.mean(xgb_accs)), "cv_xgb_accuracy_std": float(np.std(xgb_accs)),
        "cv_xgb_auc_mean": float(np.mean(xgb_aucs)), "cv_xgb_auc_std": float(np.std(xgb_aucs)),
        "cv_xgb_logloss_mean": float(np.mean(xgb_loglosses)),
        "cv_xgb_brier_mean": float(np.mean(xgb_briers)),
        "cv_sigmoid_accuracy_mean": float(np.mean(sig_accs)), "cv_sigmoid_auc_mean": float(np.mean(sig_aucs)),
        "cv_sigmoid_logloss_mean": float(np.mean(sig_loglosses)),
        "cv_isotonic_accuracy_mean": float(np.mean(iso_accs)), "cv_isotonic_auc_mean": float(np.mean(iso_aucs)),
        "cv_isotonic_logloss_mean": float(np.mean(iso_loglosses)),
        "cv_logreg_accuracy_mean": float(np.mean(lr_accs)), "cv_logreg_auc_mean": float(np.mean(lr_aucs)),
        "cv_logreg_logloss_mean": float(np.mean(lr_loglosses)),
        "cv_ensemble_accuracy_mean": float(np.mean(ens_accs)), "cv_ensemble_auc_mean": float(np.mean(ens_aucs)),
        "cv_ensemble_logloss_mean": float(np.mean(ens_loglosses)),
        "production_cv_accuracy_mean": float(np.mean(prod_accs)),
        "production_cv_auc_mean": float(np.mean(prod_aucs)),
        "final_best_iteration": float(final_model.best_iteration),
    }

    return production_model, final_model, metrics, (calib_choice if use_calibration else "none")

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

# ── Data drift reference ────────────────────────────────────────────────────────
def compute_drift_reference(df: pd.DataFrame, feature_cols: list, n_bins: int = DRIFT_N_BINS) -> dict:
    """Per-feature histogram of the training distribution, for comparing
    incoming production requests against at serving time (see api/drift.py).
    Bin edges are quantiles of the training data itself, so each bin holds
    ~1/n_bins of the reference mass by construction."""
    bins, reference_proportions = {}, {}
    for col in feature_cols:
        values = df[col].values.astype(float)
        edges = np.unique(np.quantile(values, np.linspace(0, 1, n_bins + 1)))
        if len(edges) < 3:
            # Degenerate/near-constant feature — not enough distinct quantiles
            # for a meaningful histogram; skip it rather than divide by zero.
            continue
        edges[0], edges[-1] = -np.inf, np.inf  # catch any future value, not just the training range
        counts, _ = np.histogram(values, bins=edges)
        bins[col] = edges.tolist()
        reference_proportions[col] = (counts / counts.sum()).tolist()

    return {
        "feature_cols": feature_cols,
        "n_reference_samples": int(len(df)),
        "bins": bins,
        "reference_proportions": reference_proportions,
    }


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
    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "file:./mlruns"))
    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    df = load_data()
    with mlflow.start_run() as run:
        production_model, raw_xgb_model, metrics, calib_choice = train(df)
        # SHAP needs the raw XGBoost model — a CalibratedClassifierCV wrapper isn't a
        # tree model TreeExplainer can read. Calibration is a monotonic post-hoc
        # remapping of probabilities, so the raw model's SHAP attributions still
        # correctly explain what's driving the (pre-calibration) prediction.
        explainer = compute_shap(raw_xgb_model, df[FEATURE_COLS].values, FEATURE_COLS)
        save_artifacts(production_model, explainer)

        feature_importance = dict(sorted(
            zip(FEATURE_COLS, (float(v) for v in raw_xgb_model.feature_importances_)),
            key=lambda kv: kv[1], reverse=True,
        ))
        with open(FEATURE_IMPORTANCE_OUT, "w") as f:
            json.dump(feature_importance, f, indent=2)

        drift_reference = compute_drift_reference(df, FEATURE_COLS)
        with open(DRIFT_REFERENCE_OUT, "w") as f:
            json.dump(drift_reference, f, indent=2)
        print(f"Saved drift reference ({drift_reference['n_reference_samples']} samples) → {DRIFT_REFERENCE_OUT}")

        mlflow.log_params({
            **XGB_PARAMS,
            "calib_frac": CALIB_FRAC,
            "min_calib": MIN_CALIB,
            "n_features": len(FEATURE_COLS),
            "calibration_applied": calib_choice,
        })
        mlflow.log_metrics(metrics)
        mlflow.log_artifact(SHAP_PLOT_OUT)
        mlflow.log_artifact(FEATURE_IMPORTANCE_OUT)
        mlflow.log_artifact(EXPLAINER_OUT, artifact_path="explainer")
        mlflow.log_artifact(FEATURE_COLS_OUT)
        mlflow.log_artifact(DRIFT_REFERENCE_OUT)
        mlflow.sklearn.log_model(
            production_model, artifact_path="model", registered_model_name=REGISTERED_MODEL_NAME,
        )

        print(f"\nMLflow run '{run.info.run_id}' logged under experiment '{MLFLOW_EXPERIMENT}'.")
        print(f"Registered as a new version of '{REGISTERED_MODEL_NAME}' in the model registry.")
        print("This version is NOT automatically served — promote it explicitly with:")
        print(f"  python promote_model.py <version>")

    print("\nDone! Run 'python app.py' for the Flask UI or 'uvicorn api.main:app' for the FastAPI service.")
