import pickle
from pathlib import Path

import pandas as pd
from sklearn.preprocessing import LabelEncoder

from features import CATEGORICAL, drop_duration, load_processed, split

ART = Path("artifacts")
OUT = Path("predictions.csv")


def main():
    df = load_processed()
    (_, _), (_, _), (X_test, y_test) = split(df)
    test_indices = X_test.index.tolist()
    X_test = drop_duration(X_test).copy()

    with open(ART / "model.pkl", "rb") as f:
        model = pickle.load(f)
    with open(ART / "calibrator.pkl", "rb") as f:
        calibrator = pickle.load(f)
    with open(ART / "encoders.pkl", "rb") as f:
        encoders = pickle.load(f)
    import json
    with open(ART / "metrics.json") as f:
        metrics = json.load(f)
    threshold = metrics["threshold"]

    
    for col in CATEGORICAL:
        le: LabelEncoder = encoders[col]
        X_test[col] = X_test[col].astype(str).map(
            lambda v: le.transform([v])[0] if v in le.classes_ else le.transform([le.classes_[0]])[0]
        )

    proba = calibrator.predict_proba(X_test)[:, 1]
    pred = (proba >= threshold).astype(int)

    out = pd.DataFrame({
        "row_id": test_indices,
        "probability": proba.round(6),
        "predicted_label": pred,
        "true_label": y_test.values,
        "threshold": threshold,
    })
    out.to_csv(OUT, index=False)
    print(f"wrote {OUT} · rows={len(out)}")
    print(f"predicted positives: {pred.sum()} ({pred.mean()*100:.1f}%)")
    print(f"actual positives: {y_test.sum()} ({y_test.mean()*100:.1f}%)")
    print(f"sample:\n{out.head().to_string(index=False)}")


if __name__ == "__main__":
    main()
