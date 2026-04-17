"""
src/export_model.py
───────────────────
Train the CVD risk pipeline on the integrated dataset and export:
  models/model.pkl
  models/scaler.pkl
  models/feature_names.json
  models/feature_importance.csv

Usage (from project root):
    python src/export_model.py
    python src/export_model.py --data outputs/cvd_integrated_dataset.csv
    python src/export_model.py --model xgboost
"""

import argparse
import json
import os
import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, classification_report, f1_score,
    precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

warnings.filterwarnings("ignore")

# ── Paths ──────────────────────────────────────────────────────────────────
ROOT       = Path(__file__).parent.parent
DATA_DIR   = ROOT / "outputs"
MODELS_DIR = ROOT / "models"
MODELS_DIR.mkdir(exist_ok=True)

SEED = 42
np.random.seed(SEED)

# ── Feature definitions ────────────────────────────────────────────────────
CLINICAL   = ["age","sex","chest_pain","resting_bp","cholesterol","fasting_bs",
              "rest_ecg","max_hr","exercise_angina","oldpeak","slope","ca","thal"]
WEARABLE   = ["resting_hr","daily_steps","sleep_duration","sleep_quality",
              "hrv_ms","stress_level","activity_idx"]
ENGINEERED = ["lifestyle_score","stress_index","cardio_fitness",
              "sleep_hr_recovery","pulse_pressure","chol_hr_ratio"]
ALL_FEATS  = CLINICAL + WEARABLE + ENGINEERED
TARGET     = "target"

# ── Binary cols (should not be scaled) ────────────────────────────────────
BINARY_COLS = ["sex","fasting_bs","exercise_angina","rest_ecg"]


# ═══════════════════════════════════════════════════════════════════════════
#  DATA LOADING & VALIDATION
# ═══════════════════════════════════════════════════════════════════════════

def load_data(path: Path) -> pd.DataFrame:
    print(f"\n[1] Loading data from: {path}")
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {path}\n"
            "Run the integration pipeline first to generate the dataset."
        )
    df = pd.read_csv(path)
    print(f"    Shape: {df.shape}")
    print(f"    Columns: {list(df.columns)}")
    print(f"    Missing: {df.isnull().sum().sum()}")
    return df


def validate_and_align(df: pd.DataFrame) -> tuple[pd.DataFrame, list]:
    """
    Ensure the dataset has the expected columns.
    Returns aligned df and the final feature list.
    """
    print("\n[2] Validating columns …")
    available = [f for f in ALL_FEATS if f in df.columns]
    missing   = [f for f in ALL_FEATS if f not in df.columns]

    if missing:
        print(f"    ⚠ Missing features (will be dropped from training): {missing}")
    print(f"    ✓ Using {len(available)} features: {available}")

    if TARGET not in df.columns:
        raise ValueError(f"Target column '{TARGET}' not found. Available: {list(df.columns)}")

    # Drop any internal pipeline columns
    drop_cols = [c for c in df.columns if c.startswith("_")]
    df = df.drop(columns=drop_cols, errors="ignore")

    # Impute residual NaNs
    X = df[available].copy()
    imp = SimpleImputer(strategy="median")
    X   = pd.DataFrame(imp.fit_transform(X), columns=available)
    y   = df[TARGET].astype(int)

    print(f"    Target distribution: {y.value_counts().to_dict()}")
    print(f"    CVD rate: {y.mean():.1%}")
    return X, y, available


# ═══════════════════════════════════════════════════════════════════════════
#  FEATURE ENGINEERING (compute from scratch if pre-built cols absent)
# ═══════════════════════════════════════════════════════════════════════════

def add_engineered_features(X: pd.DataFrame) -> pd.DataFrame:
    """Add engineered cols if not already present."""
    def mn(s, lo, hi): return (s.clip(lo, hi) - lo) / max(hi - lo, 1e-9)

    def _col(name): return X[name] if name in X.columns else pd.Series(0, index=X.index)

    rhr   = pd.to_numeric(_col("resting_hr"),    errors="coerce").fillna(68)
    steps = pd.to_numeric(_col("daily_steps"),   errors="coerce").fillna(7000)
    slp   = pd.to_numeric(_col("sleep_duration"),errors="coerce").fillna(6.8)
    slq   = pd.to_numeric(_col("sleep_quality"), errors="coerce").fillna(68)
    hrv   = pd.to_numeric(_col("hrv_ms"),        errors="coerce").fillna(44)
    stress= pd.to_numeric(_col("stress_level"),  errors="coerce").fillna(2.7)
    bp    = pd.to_numeric(_col("resting_bp"),    errors="coerce").fillna(130)
    chol  = pd.to_numeric(_col("cholesterol"),   errors="coerce").fillna(240)

    if "lifestyle_score" not in X.columns:
        X = X.copy()
        X["lifestyle_score"] = (
            0.25 * mn(steps, 200, 20000)  * 100 +
            0.22 * slq                          +
            0.20 * mn(hrv, 5, 120)        * 100 +
            0.15 * mn(slp, 2.5, 11)       * 100 +
            0.10 * (1 - mn(stress, 1, 5)) * 100 +
            0.08 * (1 - mn(rhr, 35, 110)) * 100
        ).clip(0, 100)

    if "stress_index" not in X.columns:
        X["stress_index"] = (
            0.40 * (1 - mn(hrv, 5, 120)) +
            0.35 * mn(rhr, 35, 110)      +
            0.25 * mn(stress, 1, 5)
        )

    if "cardio_fitness" not in X.columns:
        raw = steps / (rhr + 1e-8)
        X["cardio_fitness"] = (mn(raw, 0, 300) * 100).clip(0, 100)

    if "sleep_hr_recovery" not in X.columns:
        X["sleep_hr_recovery"] = mn(slq, 10, 100) * (1 - mn(rhr, 35, 110))

    if "pulse_pressure" not in X.columns:
        X["pulse_pressure"] = bp * 0.40

    if "chol_hr_ratio" not in X.columns:
        X["chol_hr_ratio"] = chol / (rhr + 1e-8)

    return X


# ═══════════════════════════════════════════════════════════════════════════
#  MODEL TRAINING
# ═══════════════════════════════════════════════════════════════════════════

def get_candidates(n_samples: int) -> dict:
    """Return candidate models appropriate for dataset size."""
    base = {
        "LogisticRegression": LogisticRegression(
            max_iter=2000, random_state=SEED, class_weight="balanced", C=1.0),
        "RandomForest": RandomForestClassifier(
            n_estimators=200, max_depth=8, random_state=SEED,
            class_weight="balanced", n_jobs=-1),
        "GradientBoosting": GradientBoostingClassifier(
            n_estimators=150, learning_rate=0.08, max_depth=4,
            random_state=SEED, subsample=0.85),
    }
    # Add XGBoost if available
    try:
        from xgboost import XGBClassifier
        base["XGBoost"] = XGBClassifier(
            n_estimators=150, learning_rate=0.08, max_depth=5,
            random_state=SEED, scale_pos_weight=1,
            eval_metric="logloss", verbosity=0)
    except ImportError:
        print("    ⚠ XGBoost not installed — skipping")
    return base


def train_and_select(X_train, y_train, X_val, y_val) -> tuple:
    """Train all candidates; return best by F1 + dict of all results."""
    print("\n[4] Training candidate models …\n")
    candidates = get_candidates(len(X_train))
    results    = {}

    print(f"{'Model':<25} {'Val F1':>8} {'AUC':>8} {'Recall':>8} {'Prec':>8}")
    print("─" * 60)

    for name, mdl in candidates.items():
        mdl.fit(X_train, y_train)
        y_pred = mdl.predict(X_val)
        y_prob = (mdl.predict_proba(X_val)[:, 1]
                  if hasattr(mdl, "predict_proba") else y_pred.astype(float))

        f1  = f1_score(y_val, y_pred, zero_division=0)
        auc = roc_auc_score(y_val, y_prob)
        rec = recall_score(y_val, y_pred, zero_division=0)
        pre = precision_score(y_val, y_pred, zero_division=0)

        results[name] = {"model": mdl, "f1": f1, "auc": auc,
                         "recall": rec, "precision": pre}
        print(f"{name:<25} {f1:>8.4f} {auc:>8.4f} {rec:>8.4f} {pre:>8.4f}")

    best_name  = max(results, key=lambda k: results[k]["f1"])
    best_model = results[best_name]["model"]
    print(f"\n✓ Best model: {best_name}  (F1 = {results[best_name]['f1']:.4f})")
    return best_model, best_name, results


# ═══════════════════════════════════════════════════════════════════════════
#  FEATURE IMPORTANCE EXTRACTION
# ═══════════════════════════════════════════════════════════════════════════

def extract_importance(model, feature_names: list) -> pd.DataFrame:
    """Extract feature importance regardless of model type."""
    if hasattr(model, "feature_importances_"):
        imp = model.feature_importances_
        method = "tree feature_importances_"
    elif hasattr(model, "coef_"):
        imp = np.abs(model.coef_[0])
        method = "|logistic coef_|"
    else:
        # Permutation importance fallback
        print("    Using permutation importance (slower) …")
        from sklearn.inspection import permutation_importance as pi_fn
        pi = pi_fn(model, X_test, y_test, n_repeats=10, random_state=SEED)
        imp = pi.importances_mean
        method = "permutation"

    df_imp = pd.DataFrame({
        "feature":    feature_names,
        "importance": imp,
    }).sort_values("importance", ascending=False).reset_index(drop=True)

    print(f"    Importance method: {method}")
    return df_imp


# ═══════════════════════════════════════════════════════════════════════════
#  EXPORT
# ═══════════════════════════════════════════════════════════════════════════

def export_artefacts(model, scaler, feature_names: list, feat_imp: pd.DataFrame,
                     best_name: str, eval_metrics: dict):
    print("\n[6] Exporting artefacts …")

    # model.pkl
    with open(MODELS_DIR / "model.pkl", "wb") as f:
        pickle.dump(model, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"    ✓ model.pkl  ({os.path.getsize(MODELS_DIR/'model.pkl')//1024} KB)")

    # scaler.pkl
    with open(MODELS_DIR / "scaler.pkl", "wb") as f:
        pickle.dump(scaler, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"    ✓ scaler.pkl")

    # feature_names.json
    with open(MODELS_DIR / "feature_names.json", "w") as f:
        json.dump(feature_names, f, indent=2)
    print(f"    ✓ feature_names.json  ({len(feature_names)} features)")

    # feature_importance.csv
    feat_imp.to_csv(MODELS_DIR / "feature_importance.csv", index=False)
    print(f"    ✓ feature_importance.csv")

    # model_metadata.json
    meta = {
        "model_type":    best_name,
        "model_class":   type(model).__name__,
        "n_features":    len(feature_names),
        "feature_names": feature_names,
        "seed":          SEED,
        "eval_metrics":  {k: round(float(v), 4) for k, v in eval_metrics.items()},
    }
    with open(MODELS_DIR / "model_metadata.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"    ✓ model_metadata.json")

    print(f"\n✅ All artefacts saved to: {MODELS_DIR}/")


# ═══════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main(data_path: Path, chosen_model: str = "auto"):
    print("=" * 60)
    print("  CVD Risk Model — Export Pipeline")
    print("=" * 60)

    # 1. Load
    df = load_data(data_path)

    # 2. Validate & align
    X, y, feat_list = validate_and_align(df)

    # 3. Add engineered features if missing
    X = add_engineered_features(X)
    feat_list = list(X.columns)

    # 4. Split
    print(f"\n[3] Splitting data (70/15/15) …")
    X_tv, X_test, y_tv, y_test = train_test_split(
        X, y, test_size=0.15, random_state=SEED, stratify=y)
    X_train, X_val, y_train, y_val = train_test_split(
        X_tv, y_tv, test_size=0.176, random_state=SEED, stratify=y_tv)
    print(f"    Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")

    # 5. Scale (continuous features only)
    cont_cols = [c for c in feat_list if c not in BINARY_COLS]
    scaler    = StandardScaler()
    X_train_s = X_train.copy()
    X_val_s   = X_val.copy()
    X_test_s  = X_test.copy()
    X_train_s[cont_cols] = scaler.fit_transform(X_train[cont_cols])
    X_val_s[cont_cols]   = scaler.transform(X_val[cont_cols])
    X_test_s[cont_cols]  = scaler.transform(X_test[cont_cols])
    # Store feature names in scaler for app.py compatibility check
    scaler.feature_names_in_ = np.array(feat_list)

    # 6. Train & select
    best_model, best_name, results = train_and_select(
        X_train_s, y_train, X_val_s, y_val)

    # 7. Evaluate on test set
    print("\n[5] Final test-set evaluation …")
    y_pred = best_model.predict(X_test_s)
    y_prob = (best_model.predict_proba(X_test_s)[:, 1]
              if hasattr(best_model, "predict_proba") else y_pred.astype(float))
    eval_metrics = {
        "accuracy":  accuracy_score(y_test, y_pred),
        "f1":        f1_score(y_test, y_pred, zero_division=0),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall":    recall_score(y_test, y_pred, zero_division=0),
        "auc_roc":   roc_auc_score(y_test, y_prob),
    }
    print("\nTest-set metrics:")
    for k, v in eval_metrics.items():
        print(f"    {k:<12} {v:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=["No CVD","CVD"]))

    # 8. Feature importance
    global X_test, y_test  # needed for permutation fallback
    feat_imp = extract_importance(best_model, feat_list)

    # 9. Export
    export_artefacts(best_model, scaler, feat_list, feat_imp, best_name, eval_metrics)

    print("\n" + "=" * 60)
    print("  Run the app:  streamlit run app.py")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export CVD risk model artefacts")
    parser.add_argument(
        "--data",
        type=Path,
        default=DATA_DIR / "cvd_integrated_dataset.csv",
        help="Path to integrated dataset CSV",
    )
    parser.add_argument(
        "--model",
        default="auto",
        choices=["auto","lr","rf","gb","xgb"],
        help="Model to use (auto = best F1)",
    )
    args = parser.parse_args()
    main(args.data, args.model)
