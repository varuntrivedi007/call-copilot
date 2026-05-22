"""Train LightGBM, calibrate probabilities, pick F2-optimal threshold.

Saves the calibrated model, raw LightGBM, encoders, threshold, and metrics to
artifacts/ so the Streamlit UI and SHAP layer can reuse them.
"""

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
ART.mkdir(exist_ok=True)


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
    # F-beta with beta=2
    beta = 2
    f2 = (1 + beta**2) * precision * recall / (beta**2 * precision + recall + 1e-12)
    # precision/recall arrays are length N+1 vs thresholds length N
    f2 = f2[:-1]
    best_idx = int(np.nanargmax(f2))
    return float(thresholds[best_idx]), float(f2[best_idx])


def main():
    df = load_processed()
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = split(df)
    X_train = drop_duration(X_train)
    X_val = drop_duration(X_val)
    X_test = drop_duration(X_test)

    X_train, X_val, X_test, encoders = encode_categoricals(X_train, X_val, X_test)

    pos = int((y_train == 1).sum())
    neg = int((y_train == 0).sum())
    scale_pos_weight = neg / pos
    print(f"scale_pos_weight = {scale_pos_weight:.3f} (neg={neg}, pos={pos})")

    model = lgb.LGBMClassifier(
        n_estimators=800,
        learning_rate=0.05,
        num_leaves=63,
        min_child_samples=40,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        scale_pos_weight=scale_pos_weight,
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

    val_proba_raw = model.predict_proba(X_val)[:, 1]
    test_proba_raw = model.predict_proba(X_test)[:, 1]

    # Calibrate on validation set via isotonic regression (prefit avoids retraining)
    calibrated = CalibratedClassifierCV(model, method="isotonic", cv="prefit")
    calibrated.fit(X_val, y_val)
    val_proba = calibrated.predict_proba(X_val)[:, 1]
    test_proba = calibrated.predict_proba(X_test)[:, 1]

    threshold, f2_at = best_f2_threshold(y_val, val_proba)
    print(f"chosen F2 threshold: {threshold:.4f} (val F2={f2_at:.4f})")

    metrics = {}
    for name, y_true, proba in [("val", y_val, val_proba), ("test", y_test, test_proba)]:
        pred = (proba >= threshold).astype(int)
        metrics[name] = {
            "pr_auc": float(average_precision_score(y_true, proba)),
            "roc_auc": float(roc_auc_score(y_true, proba)),
            "f1": float(f1_score(y_true, pred)),
            "f2": float(fbeta_score(y_true, pred, beta=2)),
            "confusion": confusion_matrix(y_true, pred).tolist(),
        }
    print(json.dumps(metrics, indent=2))

    raw_metrics = {
        "val_pr_auc_raw": float(average_precision_score(y_val, val_proba_raw)),
        "test_pr_auc_raw": float(average_precision_score(y_test, test_proba_raw)),
    }
    metrics["raw"] = raw_metrics

    with open(ART / "model.pkl", "wb") as f:
        pickle.dump(model, f)
    with open(ART / "calibrator.pkl", "wb") as f:
        pickle.dump(calibrated, f)
    with open(ART / "encoders.pkl", "wb") as f:
        pickle.dump(encoders, f)
    with open(ART / "metrics.json", "w") as f:
        json.dump({"threshold": threshold, **metrics}, f, indent=2)
    with open(ART / "feature_list.json", "w") as f:
        json.dump({"features": FEATURES, "categorical": CATEGORICAL, "numeric": NUMERIC}, f, indent=2)
    print("artifacts saved")


if __name__ == "__main__":
    main()
