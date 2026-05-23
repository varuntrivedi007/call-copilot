import json
import pickle
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    fbeta_score,
    precision_recall_curve,
    roc_auc_score,
)
from sklearn.preprocessing import LabelEncoder

from features import CATEGORICAL, FEATURES, NUMERIC, drop_duration, load_processed, split

ART = Path("artifacts")
WEIGHTS = [8, 10, 12, 15, 20]


def encode_categoricals(X_train, X_val, X_test):
    encoders = {}
    for col in CATEGORICAL:
        le = LabelEncoder()
        combined = pd.concat([X_train[col], X_val[col], X_test[col]]).astype(str)
        le.fit(combined)
        X_train[col] = le.transform(X_train[col].astype(str))
        X_val[col] = le.transform(X_val[col].astype(str))
        X_test[col] = le.transform(X_test[col].astype(str))
        encoders[col] = le
    return X_train, X_val, X_test, encoders


def best_f2_threshold(y_true, proba):
    precision, recall, thresholds = precision_recall_curve(y_true, proba)
    f2 = (1 + 4) * precision * recall / (4 * precision + recall + 1e-12)
    f2 = f2[:-1]
    best_idx = int(np.nanargmax(f2))
    return float(thresholds[best_idx]), float(f2[best_idx])


def train_one(X_train, y_train, X_val, y_val, spw):
    model = lgb.LGBMClassifier(
        n_estimators=800,
        learning_rate=0.05,
        num_leaves=63,
        min_child_samples=40,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        scale_pos_weight=spw,
        random_state=42,
        n_jobs=-1,
        verbose=-1,
    )
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        eval_metric="average_precision",
        callbacks=[lgb.early_stopping(50, verbose=False)],
        categorical_feature=CATEGORICAL,
    )
    return model


def main():
    df = load_processed()
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = split(df)
    X_train = drop_duration(X_train)
    X_val = drop_duration(X_val)
    X_test = drop_duration(X_test)
    X_train, X_val, X_test, encoders = encode_categoricals(X_train, X_val, X_test)

    results = []
    best_f2 = -1
    best_spw = None
    best_model = None

    for spw in WEIGHTS:
        print(f"\n--- training with scale_pos_weight={spw} ---")
        model = train_one(X_train, y_train, X_val, y_val, spw)
        
        calibrated = CalibratedClassifierCV(model, method="isotonic", cv="prefit")
        calibrated.fit(X_val, y_val)
        val_proba = calibrated.predict_proba(X_val)[:, 1]
        threshold, f2 = best_f2_threshold(y_val, val_proba)
        pr_auc = average_precision_score(y_val, val_proba)
        print(f"  val F2={f2:.4f}  PR-AUC={pr_auc:.4f}  threshold={threshold:.4f}")
        results.append({
            "scale_pos_weight": spw,
            "val_f2": f2,
            "val_pr_auc": pr_auc,
            "threshold": threshold,
        })
        if f2 > best_f2:
            best_f2 = f2
            best_spw = spw
            best_model = model
            best_calibrator = calibrated
            best_threshold = threshold

    print(f"\n=== winner: scale_pos_weight={best_spw} (val F2={best_f2:.4f}) ===")

    
    test_proba = best_calibrator.predict_proba(X_test)[:, 1]
    val_proba = best_calibrator.predict_proba(X_val)[:, 1]
    final = {}
    for name, y_true, proba in [("val", y_val, val_proba), ("test", y_test, test_proba)]:
        pred = (proba >= best_threshold).astype(int)
        final[name] = {
            "pr_auc": float(average_precision_score(y_true, proba)),
            "roc_auc": float(roc_auc_score(y_true, proba)),
            "f1": float(f1_score(y_true, pred)),
            "f2": float(fbeta_score(y_true, pred, beta=2)),
            "confusion": confusion_matrix(y_true, pred).tolist(),
        }
    print(json.dumps(final, indent=2))

   
    with open(ART / "model.pkl", "wb") as f:
        pickle.dump(best_model, f)
    with open(ART / "calibrator.pkl", "wb") as f:
        pickle.dump(best_calibrator, f)
    with open(ART / "encoders.pkl", "wb") as f:
        pickle.dump(encoders, f)
    with open(ART / "metrics.json", "w") as f:
        json.dump({
            "threshold": best_threshold,
            "scale_pos_weight": best_spw,
            **final,
        }, f, indent=2)
    with open(ART / "spw_sweep.json", "w") as f:
        json.dump({"sweep": results, "winner": best_spw}, f, indent=2)
    print("artifacts saved")


if __name__ == "__main__":
    main()
