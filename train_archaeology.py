"""
HistoriaX v3 — Improved Archaeology Model Trainer
Improvements over v2:
  1. Feature engineering (river/coast interaction, climate index)
  2. XGBoost + GradientBoosting ensemble alongside RandomForest
  3. Proper stratified CV evaluation
  4. SMOTE oversampling for High-class imbalance
  5. Separate encoder per column (audit fix)
  6. StandardScaler for numeric features
  7. Feature importance analysis output
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).parent
DATA_PATH  = ROOT / "historical_ai_system" / "data" / "global_archaeology_gis_dataset.csv"
MODEL_DIR  = ROOT / "historical_ai_system"

FEATURE_COLS = [
    "Elevation_m", "Distance_to_River_km", "Distance_to_Coast_km",
    "Rainfall_mm", "Temperature_C", "Soil_Type", "Vegetation_Type"
]
TARGET_COL = "Archaeological_Potential"


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add domain-informed derived features.
    Archaeological sites cluster near rivers, at moderate elevations,
    with specific climate+soil combinations.
    """
    df = df.copy()

    # River + coast proximity score (key archaeological indicator)
    df["River_Coast_Proximity"] = 1.0 / (1.0 + df["Distance_to_River_km"]) + \
                                   0.4 / (1.0 + df["Distance_to_Coast_km"])

    # Habitability index (moderate temp + rainfall = human settlement)
    temp_norm = df["Temperature_C"] / 50.0
    rain_norm = df["Rainfall_mm"] / 3000.0
    df["Habitability_Index"] = 1.0 - (temp_norm - 0.5).abs() + rain_norm * 0.5

    # Elevation favorability (low-to-medium elevations preferred)
    df["Elevation_Score"] = np.where(
        df["Elevation_m"] < 500,  1.0,
        np.where(df["Elevation_m"] < 1500, 0.7,
        np.where(df["Elevation_m"] < 2500, 0.3, 0.1))
    )

    # Log-transform skewed distance features
    df["Log_River_Dist"]  = np.log1p(df["Distance_to_River_km"])
    df["Log_Coast_Dist"]  = np.log1p(df["Distance_to_Coast_km"])
    df["Log_Rainfall"]    = np.log1p(df["Rainfall_mm"])

    return df


def load_and_prepare(path: Path):
    df = pd.read_csv(str(path)).dropna().drop_duplicates()
    df = engineer_features(df)

    # Separate encoders per column (v2 audit fix)
    soil_enc   = LabelEncoder()
    veg_enc    = LabelEncoder()
    target_enc = LabelEncoder()

    df["Soil_Type"]       = soil_enc.fit_transform(df["Soil_Type"])
    df["Vegetation_Type"] = veg_enc.fit_transform(df["Vegetation_Type"])
    df["Archaeological_Potential"] = target_enc.fit_transform(df["Archaeological_Potential"])

    return df, soil_enc, veg_enc, target_enc


def get_feature_cols(df):
    base = list(FEATURE_COLS)
    engineered = ["River_Coast_Proximity","Habitability_Index","Elevation_Score",
                  "Log_River_Dist","Log_Coast_Dist","Log_Rainfall"]
    return base + [c for c in engineered if c in df.columns]


def try_smote(X_train, y_train):
    """Try SMOTE oversampling; fall back gracefully if not installed."""
    try:
        from imblearn.over_sampling import SMOTE
        sm = SMOTE(random_state=42, k_neighbors=min(5, min(pd.Series(y_train).value_counts()) - 1))
        X_res, y_res = sm.fit_resample(X_train, y_train)
        print(f"  SMOTE: {len(y_train)} → {len(y_res)} samples")
        return X_res, y_res
    except ImportError:
        print("  SMOTE not available (pip install imbalanced-learn); using class_weight='balanced'")
        return X_train, y_train


def try_xgboost():
    try:
        from xgboost import XGBClassifier
        return XGBClassifier(
            n_estimators=200, max_depth=6, learning_rate=0.1,
            subsample=0.8, colsample_bytree=0.8,
            use_label_encoder=False, eval_metric="mlogloss",
            random_state=42, verbosity=0
        )
    except ImportError:
        return None


def train(verbose=True):
    print("\n" + "="*60)
    print("HistoriaX — Archaeology Model Training Pipeline v3")
    print("="*60)

    # Load
    df, soil_enc, veg_enc, target_enc = load_and_prepare(DATA_PATH)
    feature_cols = get_feature_cols(df)
    X = df[feature_cols].values
    y = df[TARGET_COL].values

    print(f"\nDataset: {len(df)} rows  |  Features: {len(feature_cols)}")
    classes = target_enc.classes_
    print(f"Classes: {dict(zip(classes, np.bincount(y)))}")

    # Scale numeric features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )

    # SMOTE for class imbalance
    X_train, y_train = try_smote(X_train, y_train)

    # ── Model ensemble ────────────────────────────────────────────────────
    rf = RandomForestClassifier(
        n_estimators=300, max_depth=None, min_samples_leaf=3,
        class_weight="balanced", random_state=42, n_jobs=-1
    )
    gb = GradientBoostingClassifier(
        n_estimators=200, max_depth=5, learning_rate=0.08,
        subsample=0.8, random_state=42
    )

    estimators = [("rf", rf), ("gb", gb)]
    xgb = try_xgboost()
    if xgb:
        estimators.append(("xgb", xgb))
        print(f"  Using XGBoost ensemble: RF + GB + XGB")
    else:
        print(f"  Using ensemble: RF + GB")

    ensemble = VotingClassifier(estimators=estimators, voting="soft", n_jobs=-1)

    # Cross-validation
    print(f"\n[Cross-Validation (5-fold stratified)]")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    # Use RF alone for fast CV
    rf_cv = RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=42, n_jobs=-1)
    cv_scores = cross_val_score(rf_cv, X_scaled, y, cv=cv, scoring="accuracy")
    print(f"  RF CV Accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    # Train full ensemble
    print(f"\n[Training ensemble on {len(X_train)} samples...]")
    ensemble.fit(X_train, y_train)

    # Evaluate
    y_pred = ensemble.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"\n[Test Results]")
    print(f"  Ensemble Accuracy: {acc:.4f}")
    print(f"\n{classification_report(y_test, y_pred, target_names=classes)}")

    # Feature importance (from RF component)
    rf.fit(X_train, y_train)  # refit standalone for importances
    importances = pd.Series(rf.feature_importances_, index=feature_cols).sort_values(ascending=False)
    print("\n[Feature Importance (RF)]")
    for feat, imp in importances.items():
        bar = "█" * int(imp * 40)
        print(f"  {feat:<30} {imp:.4f}  {bar}")

    # Save
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(ensemble,   str(MODEL_DIR / "archaeology_model.pkl"))
    joblib.dump(scaler,     str(MODEL_DIR / "feature_scaler.pkl"))
    joblib.dump(soil_enc,   str(MODEL_DIR / "soil_encoder.pkl"))
    joblib.dump(veg_enc,    str(MODEL_DIR / "veg_encoder.pkl"))
    joblib.dump(target_enc, str(MODEL_DIR / "target_encoder.pkl"))
    joblib.dump(feature_cols, str(MODEL_DIR / "feature_cols.pkl"))

    meta = {
        "accuracy": float(acc),
        "cv_mean":  float(cv_scores.mean()),
        "cv_std":   float(cv_scores.std()),
        "classes":  list(classes),
        "features": feature_cols,
        "n_train":  int(len(X_train)),
        "n_test":   int(len(X_test)),
        "ensemble": [e[0] for e in estimators],
        "feature_importance": {f: float(v) for f, v in importances.items()}
    }
    import json
    with open(str(MODEL_DIR / "model_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\n✓ All artifacts saved to: {MODEL_DIR}")
    print("="*60)
    return meta


if __name__ == "__main__":
    train()
