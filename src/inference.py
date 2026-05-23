import json
import pickle
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import shap

from features import CATEGORICAL, FEATURES, NUMERIC, load_processed
from talking_points import driver_for

try:
    from llm import generate_full_flow as llm_full_flow
    from llm import generate_pitch as llm_generate_pitch
    from llm import llm_available
except ImportError:
    llm_full_flow = None
    llm_generate_pitch = None
    llm_available = lambda: False  # noqa: E731

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


def _life_stage(age) -> str:
    try:
        a = int(age)
    except (TypeError, ValueError):
        return "your stage of life"
    if a < 25:
        return "early-career"
    if a < 35:
        return "career-building"
    if a < 50:
        return "mid-career"
    if a < 60:
        return "pre-retirement"
    return "retirement"


def _job_hook(job: str) -> str:
    job = (job or "").lower()
    hooks = {
        "retired": "given you're retired, predictable income matters more than upside",
        "student": "since you're studying, even small amounts grow meaningfully over time",
        "blue-collar": "for someone who works hard for their money, no-surprise returns matter",
        "admin.": "I know admins value clear paperwork — I can send the full terms straight after",
        "technician": "technicians I speak with usually want the numbers up front, so let me do that",
        "services": "in services work the income can swing month-to-month, this gives a stable anchor",
        "management": "for a management role, you'll appreciate that the math is straightforward",
        "entrepreneur": "for entrepreneurs, parking surplus cash with a guaranteed return helps balance risk elsewhere",
        "self-employed": "for self-employed folks, this can smooth out income variability",
        "housemaid": "for budgeting at home, this is a way to set aside funds you won't accidentally dip into",
        "unemployed": "I won't push this — if it's not the right time financially, I'd rather know now",
    }
    return hooks.get(job, "for someone in your line of work, the predictability tends to be the appeal")


def build_flow(p: float, pos: list, neg: list, customer: Optional[dict] = None) -> dict:
    bucket = confidence_bucket(p)
    customer = customer or {}
    name = (customer.get("name") or "").strip() or None
    salutation = f"{name}" if name else "there"
    job_raw = str(customer.get("job", ""))
    job = job_raw.replace(".", "").replace("_", " ")
    age = customer.get("age")
    marital = str(customer.get("marital", "")).lower()
    education = str(customer.get("education", "")).replace("_", " ")
    housing = str(customer.get("housing", "")).lower()
    loan = str(customer.get("loan", "")).lower()
    poutcome = str(customer.get("poutcome", "")).lower()
    contact = str(customer.get("contact", "")).lower()
    campaign_count = customer.get("campaign", 1)
    previous = customer.get("previous", 0)
    contacted_before = int(customer.get("was_contacted_before", 0))
    stage = _life_stage(age)
    job_hook = _job_hook(job_raw)

    top_pos_feature = pos[0]["feature"] if pos else None
    top_pos_value = str(pos[0]["value"]) if pos else None

    if bucket == "high":
        if poutcome == "success":
            opener_script = (
                f"\"Hi {salutation}, this is [Agent] from the bank. I'm calling because you signed up "
                f"with us in our previous campaign — I wanted to follow up with a new term-deposit "
                f"option that fits well for someone in the {stage} stage like you. Do you have two minutes?\""
            )
        elif contacted_before == 1:
            opener_script = (
                f"\"Hi {salutation}, this is [Agent] from the bank. We spoke during an earlier campaign — "
                f"I'm circling back because we have an updated term-deposit option, and {job_hook}. "
                f"Is this a good moment for a couple of minutes?\""
            )
        else:
            opener_script = (
                f"\"Hi {salutation}, this is [Agent] from the bank. I'm reaching out because, "
                f"{job_hook}, and customers at the {stage} stage have been particularly happy with "
                f"our current term-deposit option. Do you have two minutes to hear how it works?\""
            )
    elif bucket == "medium":
        opener_script = (
            f"\"Hi {salutation}, this is [Agent] from the bank. Before I take much of your time — "
            f"I'm calling people in the {stage} stage about a savings option. Is growing your savings "
            f"something on your radar right now, or is this not a great time?\""
        )
    else:
        if int(campaign_count or 0) >= 3:
            opener_script = (
                f"\"Hi {salutation}, this is [Agent] from the bank. I know we've reached out a few times "
                f"already on this campaign — I won't keep you. Is there a better channel or time we should "
                f"use, or would you prefer we hold off entirely?\""
            )
        else:
            opener_script = (
                f"\"Hi {salutation}, this is [Agent] from the bank — quick courtesy call, "
                f"I won't take much of your time. Just checking in on how things are with your accounts. "
                f"Is now a bad moment?\""
            )

    sentences = []
    drivers_used = []
    job_lower = str(job_raw).lower()

    if poutcome == "success":
        sentences.append(
            f"So the reason I picked up the phone to you specifically, {salutation}, "
            f"is that we worked together on a deposit before and it went well on your end. "
            f"I didn't want to call you cold about something new without that context."
        )
        sentences.append(
            "What's different this time is the rate we're able to offer — it sits a notch above "
            "what you had with us last time, and the structure is otherwise the same one you "
            "already know."
        )
        drivers_used.append(f"Prior campaign success (previous contacts: {int(previous or 0)})")
    elif contacted_before == 1:
        sentences.append(
            "We had a conversation in an earlier round of calls, so I won't start from scratch — "
            "I just want to walk you through what's actually changed since we last spoke, and "
            "see whether it lands differently now."
        )
        drivers_used.append("Prior contact on file")

    if housing == "yes":
        sentences.append(
            "One thing I should flag before anything else — because you already hold a mortgage "
            "with us, you sit inside our relationship tier, and the rate I can quote you is not "
            "the same rate we'd quote a new customer walking in off the street."
        )
        drivers_used.append("Existing mortgage — relationship customer")

    if "university" in education.lower():
        sentences.append(
            "I'll walk you through how the product actually works rather than reading marketing at you. "
            "You agree a fixed interest rate with me on the call. Your money goes in and stays untouched "
            "for whatever term we agree — anywhere from three months out to two years. At the end of that "
            "window you take out exactly what you put in plus the interest, calculated on the rate we set today. "
            "It doesn't move with the market in between."
        )
        drivers_used.append("University-educated — detail-tolerant framing")
    elif "basic" in education.lower():
        sentences.append(
            "Here's how it works in real life. You decide an amount that you're comfortable setting aside — "
            "could be small, doesn't have to be large. You hand it over, we agree the interest rate today, "
            "and after a period you choose, you get your money back with the interest added on. "
            "There's nothing happening in between, no fees taken out, no rate changing on you."
        )
        drivers_used.append("Basic education — jargon-free framing")
    else:
        sentences.append(
            "Mechanically it's straightforward. We agree a rate today, you put an amount in, and you "
            "leave it alone for however long suits you. At the end you take it out with the interest "
            "added — the rate doesn't move on you in between."
        )

    if marital == "married":
        sentences.append(
            "I know this is the kind of thing most couples want to talk through together before "
            "committing, so I'm not asking you to decide anything in this call. If it sounds like "
            "it could be a fit, I'd rather send the terms across and you can sit down with your "
            "partner tonight and look at it properly."
        )
        drivers_used.append("Married — joint planning angle")

    if job_lower == "retired":
        sentences.append(
            "Where I think this lands for you, being retired, is the predictability side. You don't "
            "want to be checking markets or wondering whether your savings rate just got cut. "
            "Once we agree the rate today, that's the rate — nothing else has to happen."
        )
        drivers_used.append("Retired — capital preservation framing")
    elif job_lower == "blue-collar":
        sentences.append(
            "I'm not going to oversell this to you. It's a place to put money you don't need to touch "
            "for a while, with a return you know in advance. No tricks, nothing that changes on you "
            "halfway through. The reason people in your line of work tend to like it is exactly that."
        )
        drivers_used.append("Blue-collar — plain-spoken framing")
    elif job_lower == "student":
        sentences.append(
            "I know money's probably tight while you're studying, so I'll say upfront the amount you'd "
            "need to start with is lower than people usually assume. The reason to do it now rather than "
            "later is just that the interest compounds — time does most of the work, not the amount."
        )
        drivers_used.append("Student — small-starter framing")
    elif job_lower in ("management", "admin"):
        sentences.append(
            "What usually matters most to people in your role is that the terms are clear and there's "
            "no admin overhead — you don't have to manage this once it's set up, you don't have to "
            "rebalance anything, you just check in at the end of the term."
        )
        drivers_used.append("Management/admin — low-overhead framing")

    if "low interest" in " ".join(d.get("driver", "").lower() for d in pos):
        sentences.append(
            "One thing I'd be transparent about — rates across the market are sitting unusually low "
            "right now, which actually works in your favour here, because the rate we lock in today "
            "doesn't move if things drop further from where they are."
        )
        drivers_used.append("Low rate environment — time-sensitive angle")

    if not sentences:
        sentences.append(
            "Where I'd land if I were on your side of the call is this — your principal is protected, "
            "your rate is agreed up front, and you know exactly what you're getting at the end. "
            "Most of the people I talk to compare it against just leaving the money sitting in a "
            "current account, and once they see the difference written down the choice usually makes itself."
        )

    pitch_paragraph = " ".join(sentences)
    pitch_block = {
        "drivers": drivers_used,
        "paragraph": pitch_paragraph,
        "source": "rules",
    }

    objection_preempts = []
    for d in neg:
        if d.get("hint"):
            objection_preempts.append({"driver": d["driver"], "say": d["hint"]})
    if not objection_preempts:
        objection_preempts.append({
            "driver": "No major risk surfaced",
            "say": "Listen for objections, don't pre-empt. If they hesitate, ask what's holding them back.",
        })

    name_for_close = name if name else "and have a good rest of your day"
    if bucket == "high":
        closing_script = (
            f"\"Based on what we've talked through{', ' + name if name else ''}, I can set this up today. "
            f"Would you like to start with the 6-month term or the 12-month term?\""
        )
    elif bucket == "medium":
        closing_script = (
            f"\"I don't want to push if the timing isn't right{', ' + name if name else ''}. "
            f"Can I send a one-page summary by email and follow up next week — same time work?\""
        )
    else:
        closing_script = (
            f"\"Thanks for taking my call{', ' + name if name else ''} — I won't keep you. "
            f"I'll make a note not to reach out again on this campaign. Take care.\""
        )

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

    flow_source = "rules"
    if llm_full_flow is not None and llm_available():
        llm_flow = llm_full_flow(customer, p, pos, neg, bucket)
        if llm_flow:
            flow_source = "groq-llama-3.3-70b"
            opener_script = llm_flow.get("opener", opener_script)
            llm_pitch = llm_flow.get("pitch")
            if llm_pitch:
                pitch_block = {
                    "drivers": drivers_used,
                    "paragraph": llm_pitch,
                    "source": "groq-llama-3.3-70b",
                }
            llm_dq = llm_flow.get("diagnostic_questions") or []
            if isinstance(llm_dq, list) and llm_dq:
                diagnostic_questions = [str(q) for q in llm_dq][:4]
            llm_op = llm_flow.get("objection_preempts") or []
            if isinstance(llm_op, list) and llm_op:
                objection_preempts = [
                    {"driver": "LLM-surfaced risk", "say": str(s)} for s in llm_op[:4]
                ]
            closing_script = llm_flow.get("close", closing_script)

    return {
        "confidence_bucket": bucket,
        "profile_summary": profile_summary,
        "flow_source": flow_source,
        "stage_1_opener": opener_script,
        "stage_2_pitch": pitch_block,
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
