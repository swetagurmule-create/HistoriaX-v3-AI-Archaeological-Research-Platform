"""
HistoriaX v3 — Improved Archaeological Site Prediction Trainer
===============================================================
BUGS FIXED from original geo/train_model.py:
  BUG-1: Single LabelEncoder reused across 3 columns → all 3 now separate
  BUG-2: Encoders never saved → soil/veg/target encoders now saved to disk
  BUG-3: No class_weight → balanced weighting for High(1.9%)/Medium/Low
  BUG-4: Default RF hyperparams → tuned 300-tree ensemble
  BUG-5: No stratified split → stratify=y added

NEW: Feature engineering (6 derived features)
NEW: VotingClassifier ensemble (RF + GradientBoosting + optional XGBoost)
NEW: StandardScaler persisted for consistent inference
NEW: 5-fold stratified cross-validation reported
NEW: Feature importance saved to JSON
"""
import sys
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import (RandomForestClassifier, GradientBoostingClassifier,
                               VotingClassifier)
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score, classification_report

log = logging.getLogger("historiaX.gis.trainer")

HERE     = Path(__file__).parent
DATA_CSV = HERE / "gis" / "global_archaeology_gis_dataset.csv"
OUT_DIR  = HERE / "gis"

FEATURE_BASE = [
    "Elevation_m", "Distance_to_River_km", "Distance_to_Coast_km",
    "Rainfall_mm", "Temperature_C", "Soil_Type", "Vegetation_Type"
]


# ── 1. Feature Engineering ────────────────────────────────────────────────
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Domain-driven derived features. Archaeology research shows:
    - Sites cluster within 5 km of rivers (alluvial settlement)
    - Low-to-medium elevations (0–800 m) dominate
    - Mild climates (10–30°C, 400–2000 mm rain) favour habitation
    - Soil×Vegetation interaction matters
    """
    d = df.copy()

    # River proximity score (nonlinear: exponential decay)
    d["River_Prox_Score"] = np.exp(-d["Distance_to_River_km"] / 10.0)

    # Coast proximity score
    d["Coast_Prox_Score"] = np.exp(-d["Distance_to_Coast_km"] / 200.0)

    # Combined water access (rivers weighted 3× coast for inland sites)
    d["Water_Access"] = d["River_Prox_Score"] * 3 + d["Coast_Prox_Score"]

    # Elevation suitability (archaeological sites peak at 50–500 m)
    d["Elev_Suit"] = np.where(d["Elevation_m"] < 50, 0.5,
                    np.where(d["Elevation_m"] < 500, 1.0,
                    np.where(d["Elevation_m"] < 1500, 0.6,
                    np.where(d["Elevation_m"] < 2500, 0.3, 0.1))))

    # Climate habitability score
    temp_opt = 1 - abs(d["Temperature_C"] - 20) / 40.0  # optimal ~20°C
    rain_opt = np.where(d["Rainfall_mm"] < 300, d["Rainfall_mm"] / 300,
               np.where(d["Rainfall_mm"] < 1500, 1.0,
                        1.0 - (d["Rainfall_mm"] - 1500) / 2500))
    d["Climate_Score"] = (temp_opt.clip(0, 1) + rain_opt.clip(0, 1)) / 2

    # Log-transforms for skewed features
    d["Log_River"]  = np.log1p(d["Distance_to_River_km"])
    d["Log_Coast"]  = np.log1p(d["Distance_to_Coast_km"])
    d["Log_Rain"]   = np.log1p(d["Rainfall_mm"])

    return d


def get_all_features(df: pd.DataFrame) -> list:
    eng = ["River_Prox_Score", "Coast_Prox_Score", "Water_Access",
           "Elev_Suit", "Climate_Score", "Log_River", "Log_Coast", "Log_Rain"]
    return FEATURE_BASE + [c for c in eng if c in df.columns]


# ── 2. Data Loading & Encoding ────────────────────────────────────────────
def load_and_encode(path: Path):
    df = pd.read_csv(str(path)).drop_duplicates().dropna()
    df = engineer_features(df)

    # FIX-1: Separate encoder per column
    soil_enc   = LabelEncoder()
    veg_enc    = LabelEncoder()
    target_enc = LabelEncoder()

    df["Soil_Type"]               = soil_enc.fit_transform(df["Soil_Type"])
    df["Vegetation_Type"]         = veg_enc.fit_transform(df["Vegetation_Type"])
    df["Archaeological_Potential"] = target_enc.fit_transform(df["Archaeological_Potential"])

    return df, soil_enc, veg_enc, target_enc


# ── 3. SMOTE helper ────────────────────────────────────────────────────────
def try_smote(X, y):
    try:
        from imblearn.over_sampling import SMOTE
        k = min(5, int(pd.Series(y).value_counts().min()) - 1)
        if k < 1:
            return X, y
        sm = SMOTE(random_state=42, k_neighbors=k)
        Xr, yr = sm.fit_resample(X, y)
        log.info(f"SMOTE: {len(y)} → {len(yr)} samples")
        return Xr, yr
    except ImportError:
        log.info("imbalanced-learn not installed; using class_weight='balanced'")
        return X, y


# ── 4. XGBoost helper ─────────────────────────────────────────────────────
def make_xgb():
    try:
        from xgboost import XGBClassifier
        return XGBClassifier(
            n_estimators=200, max_depth=6, learning_rate=0.08,
            subsample=0.8, colsample_bytree=0.8,
            eval_metric="mlogloss", random_state=42, verbosity=0
        )
    except ImportError:
        return None


# ── 5. Main training function ─────────────────────────────────────────────
def train(verbose: bool = True) -> dict:
    log.info("="*60)
    log.info("HistoriaX — GIS Model Training Pipeline v3")
    log.info("="*60)

    df, soil_enc, veg_enc, target_enc = load_and_encode(DATA_CSV)
    feat_cols = get_all_features(df)
    X = df[feat_cols].values
    y = df["Archaeological_Potential"].values

    classes = target_enc.classes_
    counts  = dict(zip(classes, np.bincount(y)))
    log.info(f"Dataset: {len(df)} rows | Features: {len(feat_cols)}")
    log.info(f"Classes: {counts}")

    # FIX-3: StandardScaler for numeric consistency
    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # FIX-4: Stratified split
    X_tr, X_te, y_tr, y_te = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y)

    X_tr, y_tr = try_smote(X_tr, y_tr)

    # ── Ensemble ────────────────────────────────────────────────────────
    rf = RandomForestClassifier(
        n_estimators=300, max_depth=None, min_samples_leaf=3,
        class_weight="balanced", random_state=42, n_jobs=-1)  # FIX-3
    gb = GradientBoostingClassifier(
        n_estimators=200, max_depth=5, learning_rate=0.08,
        subsample=0.8, random_state=42)

    estimators = [("rf", rf), ("gb", gb)]
    xgb = make_xgb()
    if xgb:
        estimators.append(("xgb", xgb))
        log.info("Ensemble: RF + GB + XGBoost")
    else:
        log.info("Ensemble: RF + GB (install xgboost for +XGB)")

    ensemble = VotingClassifier(estimators=estimators, voting="soft", n_jobs=1)

    # Cross-validation (fast: RF only)
    log.info("Running 5-fold CV on RF...")
    rf_quick = RandomForestClassifier(
        n_estimators=100, class_weight="balanced", random_state=42, n_jobs=-1)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(rf_quick, X_scaled, y, cv=cv, scoring="accuracy")
    log.info(f"RF CV Accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    log.info(f"Training ensemble on {len(X_tr)} samples...")
    ensemble.fit(X_tr, y_tr)

    y_pred = ensemble.predict(X_te)
    acc    = accuracy_score(y_te, y_pred)
    report = classification_report(y_te, y_pred, target_names=classes, output_dict=True)
    log.info(f"Ensemble Test Accuracy: {acc:.4f}")
    if verbose:
        print(classification_report(y_te, y_pred, target_names=classes))

    # Feature importances from RF component
    rf.fit(X_tr, y_tr)
    imps = pd.Series(rf.feature_importances_, index=feat_cols).sort_values(ascending=False)

    # ── FIX-2: Save ALL encoders ──────────────────────────────────────
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(ensemble,    str(OUT_DIR / "archaeology_model.pkl"))
    joblib.dump(scaler,      str(OUT_DIR / "feature_scaler.pkl"))
    joblib.dump(soil_enc,    str(OUT_DIR / "soil_encoder.pkl"))
    joblib.dump(veg_enc,     str(OUT_DIR / "veg_encoder.pkl"))
    joblib.dump(target_enc,  str(OUT_DIR / "target_encoder.pkl"))
    joblib.dump(feat_cols,   str(OUT_DIR / "feature_cols.pkl"))

    meta = {
        "accuracy":          float(acc),
        "cv_mean":           float(cv_scores.mean()),
        "cv_std":            float(cv_scores.std()),
        "classes":           list(classes),
        "soil_types":        list(soil_enc.classes_),
        "vegetation_types":  list(veg_enc.classes_),
        "feature_cols":      feat_cols,
        "feature_importance": imps.to_dict(),
        "per_class_metrics": {
            c: {
                "precision": round(report[c]["precision"], 4),
                "recall":    round(report[c]["recall"], 4),
                "f1":        round(report[c]["f1-score"], 4),
            }
            for c in classes if c in report
        },
        "ensemble":          [e[0] for e in estimators],
        "n_train":           int(len(X_tr)),
        "n_test":            int(len(X_te)),
    }
    with open(str(OUT_DIR / "model_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    log.info(f"✓ All artifacts saved → {OUT_DIR}")
    return meta


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    train(verbose=True)
