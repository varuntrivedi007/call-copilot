"""Baselines: dummy classifier + logistic regression with class weights."""

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    fbeta_score,
    roc_auc_score,
)
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from features import CATEGORICAL, NUMERIC, drop_duration, load_processed, split


def evaluate(name: str, y_true, y_pred, y_proba) -> dict:
    return {
        "model": name,
        "pr_auc": average_precision_score(y_true, y_proba),
        "roc_auc": roc_auc_score(y_true, y_proba),
        "f1": f1_score(y_true, y_pred),
        "f2": fbeta_score(y_true, y_pred, beta=2),
        "tn_fp_fn_tp": confusion_matrix(y_true, y_pred).ravel().tolist(),
    }


def main() -> None:
    df = load_processed()
    (X_train, y_train), (X_val, y_val), _ = split(df)
    X_train = drop_duration(X_train)
    X_val = drop_duration(X_val)

    # Dummy: always predicts majority class
    dummy = DummyClassifier(strategy="most_frequent")
    dummy.fit(X_train, y_train)
    dummy_pred = dummy.predict(X_val)
    dummy_proba = dummy.predict_proba(X_val)[:, 1]

    # Logistic regression with class weights
    numeric_pipe = Pipeline(
        [("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]
    )
    preprocessor = ColumnTransformer(
        [
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
            ("num", numeric_pipe, NUMERIC),
        ]
    )
    logreg = Pipeline(
        [
            ("prep", preprocessor),
            (
                "model",
                LogisticRegression(
                    class_weight="balanced", max_iter=2000, n_jobs=-1
                ),
            ),
        ]
    )
    logreg.fit(X_train, y_train)
    logreg_proba = logreg.predict_proba(X_val)[:, 1]
    logreg_pred = (logreg_proba >= 0.5).astype(int)

    results = pd.DataFrame(
        [
            evaluate("dummy_majority", y_val, dummy_pred, dummy_proba),
            evaluate("logreg_balanced", y_val, logreg_pred, logreg_proba),
        ]
    )
    print(results.to_string(index=False))
    results.to_csv("artifacts/baseline_results.csv", index=False)


if __name__ == "__main__":
    main()
