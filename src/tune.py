import json
import pickle
from pathlib import Path

import lightgbm as lgb
import numpy as np
import optuna
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
    beta = 2
    f2 = (1 + beta**2) * precision * recall / (beta**2 * precision + recall + 1e-12)
    f2 = f2[:-1]
    best_idx = int(np.nanargmax(f2))
    return float(thresholds[best_idx]), float(f2[best_idx])


def make_model(params, scale_pos_weight):
    return lgb.LGBMClassifier(
        n_estimators=1500,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        n_jobs=-1,
        verbose=-1,
        **params,
    )


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
    print(f"scale_pos_weight = {scale_pos_weight:.3f}")

    def objective(trial: optuna.trial.Trial) -> float:
        params = {
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.12, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 15, 255, log=True),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 200, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
            "max_depth": trial.suggest_int("max_depth", -1, 12),
        }
        model = make_model(params, scale_pos_weight)
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            eval_metric="average_precision",
            callbacks=[lgb.early_stopping(50, verbose=False)],
            categorical_feature=CATEGORICAL,
        )
        proba = model.predict_proba(X_val)[:, 1]
        return average_precision_score(y_val, proba)

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=10),
    )
    print("running 50 Optuna trials...")
    study.optimize(objective, n_trials=50, show_progress_bar=False)
    print(f"best val PR-AUC: {study.best_value:.4f}")
    print(f"best params: {json.dumps(study.best_params, indent=2)}")

    
    best_model = make_model(study.best_params, scale_pos_weight)
    best_model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        eval_metric="average_precision",
        callbacks=[lgb.early_stopping(50, verbose=False)],
        categorical_feature=CATEGORICAL,
    )

    
    calibrated = CalibratedClassifierCV(best_model, method="isotonic", cv="prefit")
    calibrated.fit(X_val, y_val)
    val_proba = calibrated.predict_proba(X_val)[:, 1]
    test_proba = calibrated.predict_proba(X_test)[:, 1]

    threshold, f2_val = best_f2_threshold(y_val, val_proba)
    print(f"chosen threshold: {threshold:.4f} (val F2={f2_val:.4f})")

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

    with open(ART / "model.pkl", "wb") as f:
        pickle.dump(best_model, f)
    with open(ART / "calibrator.pkl", "wb") as f:
        pickle.dump(calibrated, f)
    with open(ART / "encoders.pkl", "wb") as f:
        pickle.dump(encoders, f)
    with open(ART / "metrics.json", "w") as f:
        json.dump({"threshold": threshold, **metrics, "best_optuna_value": study.best_value}, f, indent=2)
    with open(ART / "optuna_best.json", "w") as f:
        json.dump({
            "best_params": study.best_params,
            "best_val_pr_auc": study.best_value,
            "n_trials": len(study.trials),
            "trial_history": [
                {"number": t.number, "value": t.value, "params": t.params}
                for t in study.trials if t.value is not None
            ],
        }, f, indent=2)
    print("artifacts saved")


if __name__ == "__main__":
    main()
