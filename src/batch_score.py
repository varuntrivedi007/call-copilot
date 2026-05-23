"""Bulk score a CSV of customers.

Fills missing columns with sensible defaults (macro features = recent median,
campaign history = first-contact). Output keeps the original columns plus
probability, label, bucket, and top drivers.

Pillar 01 note: the scorer does not sort or rank. Sorting/filtering is left to
the consumer (UI table). The UI is responsible for not displaying a
probability-descending list.
"""

import argparse
from pathlib import Path
from typing import Optional

import pandas as pd

from features import CATEGORICAL, load_processed


def _recent_macro_defaults(reference: pd.DataFrame) -> dict:
    recent = reference.tail(1000)
    return {
        col: float(recent[col].median())
        for col in ["emp_var_rate", "cons_price_idx", "cons_conf_idx", "euribor3m", "nr_employed"]
        if col in reference.columns
    }


def normalize_uploaded(df: pd.DataFrame, reference: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """Bring an uploaded CSV up to the model's feature schema."""
    if reference is None:
        reference = load_processed()

    df = df.copy()

    # Categorical defaults
    for col in CATEGORICAL:
        if col not in df.columns:
            df[col] = "unknown"
        df[col] = df[col].astype(str).fillna("unknown")

    # Engineered pdays fields
    if "pdays" in df.columns and "was_contacted_before" not in df.columns:
        df["was_contacted_before"] = (df["pdays"] != 999).astype(int)
        df["pdays_clean"] = df["pdays"].where(df["pdays"] != 999)
    if "was_contacted_before" not in df.columns:
        df["was_contacted_before"] = 0
    if "pdays_clean" not in df.columns:
        df["pdays_clean"] = float("nan")

    # Campaign history defaults
    if "previous" not in df.columns:
        df["previous"] = 0
    if "campaign" not in df.columns:
        df["campaign"] = 1
    if "age" not in df.columns:
        df["age"] = int(reference["age"].median())

    # Macro defaults
    for col, val in _recent_macro_defaults(reference).items():
        if col not in df.columns:
            df[col] = val
        else:
            df[col] = df[col].fillna(val)

    # Seasonality defaults from last reference row
    last = reference.iloc[-1]
    if "month" not in df.columns:
        df["month"] = str(last.get("month", "may"))
    if "day_of_week" not in df.columns:
        df["day_of_week"] = str(last.get("day_of_week", "mon"))

    return df


def score_dataframe(df: pd.DataFrame, copilot) -> pd.DataFrame:
    """Return per-row probability + drivers. Does not sort."""
    rows = []
    for orig_idx, row in df.iterrows():
        row_dict = row.to_dict()
        result = copilot.predict(row_dict)
        flow = result.get("flow", {})
        rows.append({
            "row_index": int(orig_idx),
            "name": row_dict.get("name") or row_dict.get("customer_id") or f"Customer {orig_idx}",
            "age": row_dict.get("age"),
            "job": row_dict.get("job"),
            "marital": row_dict.get("marital"),
            "education": row_dict.get("education"),
            "probability": float(result["probability"]),
            "bucket": flow.get("confidence_bucket", "low"),
            "label": result["label"],
            "top_positive_drivers": "; ".join(
                d["driver"] for d in result["positive_drivers"][:2]
            ) or "—",
            "top_negative_drivers": "; ".join(
                d["driver"] for d in result["negative_drivers"][:2]
            ) or "—",
        })
    return pd.DataFrame(rows)


def main():
    from inference import Copilot  # local import to avoid Streamlit load order issues

    parser = argparse.ArgumentParser(description="Bulk score customers from CSV")
    parser.add_argument("--input", required=True, help="Path to input CSV")
    parser.add_argument("--output", required=True, help="Path to output scored CSV")
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    df_norm = normalize_uploaded(df)
    copilot = Copilot()
    scored = score_dataframe(df_norm, copilot)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    scored.to_csv(args.output, index=False)
    print(f"wrote {args.output} · rows={len(scored)}")


if __name__ == "__main__":
    main()
