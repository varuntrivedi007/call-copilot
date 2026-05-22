"""Inference + explanation helpers shared by Streamlit app and CLI demos.

`predict_one` returns probability, calibrated probability, top SHAP drivers
mapped to talking points, and a structured 'flow' the agent can read.
"""

import json
import pickle
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import shap

from features import CATEGORICAL, FEATURES, NUMERIC, load_processed
from talking_points import driver_for

ART = Path("artifacts")


class Copilot:
    def __init__(self):
        with open(ART / "model.pkl", "rb") as f:
            self.model = pickle.load(f)
        with open(ART / "calibrator.pkl", "rb") as f:
            self.calibrator = pickle.load(f)
        with open(ART / "encoders.pkl", "rb") as f:
            self.encoders = pickle.load(f)
        with open(ART / "metrics.json") as f:
            self.metrics = json.load(f)
        self.threshold = self.metrics["threshold"]
        self.explainer = shap.TreeExplainer(self.model)

    def _encode_row(self, row: dict) -> pd.DataFrame:
        x = {f: row.get(f) for f in FEATURES}
        df = pd.DataFrame([x])
        for col in CATEGORICAL:
            le = self.encoders[col]
            val = str(df.at[0, col])
            if val in le.classes_:
                df[col] = le.transform([val])
            else:
                # Unknown category at inference -> fall back to most common during training
                df[col] = le.transform([le.classes_[0]])
        return df[FEATURES]

    def predict(self, row: dict) -> dict:
        encoded = self._encode_row(row)
        raw_proba = float(self.model.predict_proba(encoded)[0, 1])
        cal_proba = float(self.calibrator.predict_proba(encoded)[0, 1])

        shap_values = self.explainer.shap_values(encoded)
        if isinstance(shap_values, list):
            shap_row = shap_values[1][0]
        else:
            shap_row = shap_values[0]

        contributions = sorted(
            [
                (feat, float(shap_row[i]))
                for i, feat in enumerate(FEATURES)
            ],
            key=lambda t: abs(t[1]),
            reverse=True,
        )

        drivers = []
        for feat, contrib in contributions[:6]:
            raw_value = row.get(feat)
            d = driver_for(feat, raw_value, contrib)
            if d is not None:
                drivers.append(d)

        positive_drivers = [d for d in drivers if d["direction"] == "+"][:3]
        negative_drivers = [d for d in drivers if d["direction"] == "-"][:2]

        flow = build_flow(cal_proba, positive_drivers, negative_drivers, customer=row)

        return {
            "probability": cal_proba,
            "raw_probability": raw_proba,
            "threshold": self.threshold,
            "label": "likely_yes" if cal_proba >= self.threshold else "likely_no",
            "positive_drivers": positive_drivers,
            "negative_drivers": negative_drivers,
            "flow": flow,
        }


def confidence_bucket(p: float) -> str:
    if p >= 0.5:
        return "high"
    if p >= 0.2:
        return "medium"
    return "low"


def build_flow(p: float, pos: list, neg: list, customer: Optional[dict] = None) -> dict:
    bucket = confidence_bucket(p)
    customer = customer or {}
    job = str(customer.get("job", "")).replace("_", " ")
    age = customer.get("age")
    contacted_before = int(customer.get("was_contacted_before", 0))

    top_pos_feature = pos[0]["feature"] if pos else None
    top_pos_value = str(pos[0]["value"]) if pos else None

    if bucket == "high":
        if top_pos_feature == "poutcome" and top_pos_value == "success":
            opener_script = (
                "\"Hi, this is [Agent] from the bank. I'm following up because you signed up "
                "with us in a previous campaign, and I wanted to share a new term-deposit "
                "option I think fits what you liked last time. Do you have two minutes?\""
            )
        elif top_pos_feature == "was_contacted_before" or contacted_before == 1:
            opener_script = (
                "\"Hi, this is [Agent] from the bank. We spoke on a previous campaign — "
                "I'm calling back with an updated offer that I think will be more relevant "
                "for you now. Is this a good time?\""
            )
        else:
            opener_script = (
                "\"Hi, this is [Agent] from the bank. I'm calling because we have a "
                "term-deposit option that's been a strong fit for customers in a similar "
                "profile to yours. Do you have a couple of minutes to hear how it works?\""
            )
    elif bucket == "medium":
        opener_script = (
            "\"Hi, this is [Agent] from the bank. Before I take any of your time — "
            "are you currently looking at ways to grow your savings, or is this not "
            "something on your radar right now?\""
        )
    else:
        opener_script = (
            "\"Hi, this is [Agent] from the bank — I won't take much of your time. "
            "I'm just checking in to see how things are going and whether there's "
            "anything we can help with on your accounts. Is now a bad moment?\""
        )

    talking_points = []
    for d in pos:
        if d.get("hint"):
            talking_points.append({"driver": d["driver"], "say": d["hint"]})
    if not talking_points:
        talking_points.append({
            "driver": "Generic value",
            "say": "\"It's a fixed-rate term deposit, principal is protected, and you can pick the term length.\"",
        })

    objection_preempts = []
    for d in neg:
        if d.get("hint"):
            objection_preempts.append({"driver": d["driver"], "say": d["hint"]})
    if not objection_preempts:
        objection_preempts.append({
            "driver": "No major risk surfaced",
            "say": "Listen for objections, don't pre-empt. If they hesitate, ask what's holding them back.",
        })

    closing_script = {
        "high": (
            "\"Based on what we discussed, I can set you up today. Would you like to start "
            "with the 6-month term or the 12-month term?\""
        ),
        "medium": (
            "\"I don't want to push this if the timing isn't right. Can I send you a short "
            "summary by email and follow up next week?\""
        ),
        "low": (
            "\"Thanks for taking my call — I won't keep you. I'll make a note not to call "
            "again on this campaign. Have a good day.\""
        ),
    }[bucket]

    diagnostic_questions = {
        "high": [
            "\"What's the time horizon you'd want to lock these funds for?\"",
            "\"Are you comparing this against any other savings options right now?\"",
        ],
        "medium": [
            "\"What would make this a clear yes or no for you?\"",
            "\"Is there a savings goal you're working toward in the next 12 months?\"",
        ],
        "low": [
            "\"Is there a better channel or time for us to reach you in future?\"",
        ],
    }[bucket]

    profile_summary_bits = []
    if age:
        profile_summary_bits.append(f"age {int(age)}")
    if job:
        profile_summary_bits.append(job)
    if contacted_before:
        profile_summary_bits.append("prior contact on file")
    else:
        profile_summary_bits.append("no prior contact")
    profile_summary = ", ".join(profile_summary_bits)

    return {
        "confidence_bucket": bucket,
        "profile_summary": profile_summary,
        "stage_1_opener": opener_script,
        "stage_2_talking_points": talking_points,
        "stage_3_diagnostic_questions": diagnostic_questions,
        "stage_4_objection_preempts": objection_preempts,
        "stage_5_close": closing_script,
    }


def sample_customer(idx: int = 0) -> dict:
    df = load_processed()
    row = df.iloc[idx].to_dict()
    return row


if __name__ == "__main__":
    copilot = Copilot()
    for i in [3, 100, 5000]:
        cust = sample_customer(i)
        out = copilot.predict(cust)
        print(f"--- customer {i} (true y={cust['y']}) ---")
        print(json.dumps({k: v for k, v in out.items() if k != "flow"}, indent=2, default=str))
        print("flow:")
        print(json.dumps(out["flow"], indent=2))
