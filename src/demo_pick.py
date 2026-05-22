"""Pick three demo customers from the validation set: high / mid / low probability.

Writes the row indices and metadata to artifacts/demo_customers.json so the
Streamlit UI and rehearsal script can jump straight to them.
"""

import json
from pathlib import Path

import numpy as np

from features import drop_duration, load_processed, split
from inference import Copilot

ART = Path("artifacts")


def main():
    df = load_processed()
    (_, _), (X_val, y_val), _ = split(df)
    val_indices = X_val.index.tolist()

    copilot = Copilot()
    probs = []
    for idx in val_indices:
        row = df.loc[idx].to_dict()
        out = copilot.predict(row)
        probs.append((idx, out["probability"], int(y_val.loc[idx])))

    probs_sorted_high = sorted(probs, key=lambda t: -t[1])
    high = next((p for p in probs_sorted_high if p[2] == 1), probs_sorted_high[0])

    mid_candidates = [p for p in probs if 0.35 <= p[1] <= 0.55]
    mid = mid_candidates[len(mid_candidates) // 2] if mid_candidates else probs_sorted_high[len(probs_sorted_high) // 2]

    low_candidates = sorted(probs, key=lambda t: t[1])
    low = next((p for p in low_candidates if p[2] == 0), low_candidates[0])

    picks = {
        "high": {"index": int(high[0]), "probability": float(high[1]), "true_y": int(high[2])},
        "mid": {"index": int(mid[0]), "probability": float(mid[1]), "true_y": int(mid[2])},
        "low": {"index": int(low[0]), "probability": float(low[1]), "true_y": int(low[2])},
    }
    print(json.dumps(picks, indent=2))
    with open(ART / "demo_customers.json", "w") as f:
        json.dump(picks, f, indent=2)


if __name__ == "__main__":
    main()
