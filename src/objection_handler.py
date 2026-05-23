"""Live objection handler.

Agent types or selects what the customer just said. We keyword-match against
common objection categories and return a tailored counter that pulls in
relevant customer context (job, marital status, prior contact, etc.).
"""

from dataclasses import dataclass
from typing import Optional

try:
    from llm import generate_objection_response, llm_available
except ImportError:
    generate_objection_response = None
    llm_available = lambda: False  # noqa: E731


@dataclass
class Objection:
    category: str
    keywords: tuple
    counter_template: str
    follow_up: str


OBJECTIONS = [
    Objection(
        category="no_time",
        keywords=("no time", "busy", "in a meeting", "can't talk", "bad time", "later"),
        counter_template=(
            "\"Understood — I'll keep it under 60 seconds. The short version: it's a fixed-rate "
            "term deposit, principal protected. Worth me sending a one-pager so you can read it "
            "when it's quieter?\""
        ),
        follow_up="If they still say no — ask for a better callback time and end the call.",
    ),
    Objection(
        category="not_interested",
        keywords=("not interested", "don't want", "no thanks", "not for me"),
        counter_template=(
            "\"That's fair — can I ask, is it the product itself or just bad timing? "
            "If it's timing I'll note it and try again later; if it's the product I'd like to know why "
            "so I don't waste your time again.\""
        ),
        follow_up="Listen carefully. Don't argue. Use answer to decide whether to re-queue or drop.",
    ),
    Objection(
        category="too_risky",
        keywords=("risky", "risk", "lose money", "scared", "unsafe"),
        counter_template=(
            "\"That's actually the part most people misunderstand — your principal is fully protected. "
            "The rate is fixed when you open it, so even if markets move, the return you signed up for "
            "doesn't change. You can't end up with less than you put in.\""
        ),
        follow_up="Then offer to walk through the FDIC-equivalent guarantee in writing.",
    ),
    Objection(
        category="rate_too_low",
        keywords=("rate", "interest", "too low", "better elsewhere", "not enough"),
        counter_template=(
            "\"Honest answer: you can find higher headline rates, but most have variable terms or "
            "lock-up penalties. The trade-off here is a fixed rate you know in advance with no surprises. "
            "Would you like me to compare it side-by-side with the alternative you're looking at?\""
        ),
        follow_up="Pull up rate comparison sheet. Be specific, not vague.",
    ),
    Objection(
        category="no_money",
        keywords=("no money", "can't afford", "tight", "short", "not enough cash"),
        counter_template=(
            "\"Totally understand. The minimum is lower than most people assume — you can open one "
            "with a small amount and add to it later. Would it help if I told you the minimum and you "
            "decide if it's worth it?\""
        ),
        follow_up="If still no — do NOT push. Move to soft-close.",
    ),
    Objection(
        category="need_spouse",
        keywords=("wife", "husband", "spouse", "partner", "talk to", "check with"),
        counter_template=(
            "\"Of course — most couples decide this together. Let me send you the one-page summary "
            "so you both can look at the same numbers. When would be a good time to follow up — "
            "later this week?\""
        ),
        follow_up="Lock in a specific callback time. Don't leave it open-ended.",
    ),
    Objection(
        category="dont_trust_bank",
        keywords=("don't trust", "scam", "fraud", "is this real", "suspicious"),
        counter_template=(
            "\"That's a completely reasonable concern — there are a lot of fake calls out there. "
            "You can verify me by hanging up and calling the bank's main number on the back of your card. "
            "Ask for [Agent], and I can pick up from there.\""
        ),
        follow_up="Encourage them to hang up and call back. Builds trust.",
    ),
    Objection(
        category="already_have",
        keywords=("already have", "got one", "another bank", "current account"),
        counter_template=(
            "\"Good — then you already understand how these work. The reason I'm calling is our "
            "current rate is competitive vs what most customers have. If you'd like, share roughly "
            "what rate you're getting and I'll tell you straight if we beat it or not.\""
        ),
        follow_up="Be honest if you can't beat it. Trust > one sale.",
    ),
    Objection(
        category="how_did_you_get_number",
        keywords=("how did you get", "where did you get my number", "remove me", "do not call"),
        counter_template=(
            "\"You're on our customer list because you have an account with us. If you'd like to be "
            "removed from marketing calls, I can do that right now — no questions asked.\""
        ),
        follow_up="If they ask to opt out — opt them out immediately and confirm in writing.",
    ),
]


def find_objection(customer_said: str) -> Optional[Objection]:
    text = (customer_said or "").lower()
    if not text:
        return None
    best = None
    best_score = 0
    for obj in OBJECTIONS:
        score = sum(1 for kw in obj.keywords if kw in text)
        if score > best_score:
            best = obj
            best_score = score
    return best if best_score > 0 else None


def respond(customer_said: str, customer: dict) -> dict:
    obj = find_objection(customer_said)
    if obj is None:
        counter = (
            "\"Can you tell me a bit more about that? I want to make sure I'm answering "
            "what you're actually asking, not a generic version of it.\""
        )
        source = "rules"
        if generate_objection_response is not None and llm_available() and customer_said:
            llm_text = generate_objection_response(customer_said, customer, "unmatched")
            if llm_text:
                counter = llm_text
                source = "groq-llama-3.3-70b"
        return {
            "category": "unmatched",
            "counter": counter,
            "follow_up": "Let the customer talk. Take notes. Then pick the closest objection category.",
            "personalization_note": None,
            "source": source,
        }

    personalization = None
    job = str(customer.get("job", "")).replace("_", " ")
    marital = str(customer.get("marital", "")).lower()
    housing = str(customer.get("housing", "")).lower()
    poutcome = str(customer.get("poutcome", "")).lower()

    if obj.category == "too_risky" and poutcome == "success":
        personalization = (
            "Reminder: they said yes to us before — reference that prior trust."
        )
    if obj.category == "need_spouse" and marital != "married":
        personalization = (
            "They're not married per our records — gently clarify who they need to consult, "
            "could be a parent or financial advisor."
        )
    if obj.category == "dont_trust_bank" and housing == "yes":
        personalization = (
            "They have a mortgage with us — surface that relationship to anchor trust: "
            "\"You actually already bank with us — your mortgage is with our branch.\""
        )
    if obj.category == "no_money" and job in ("retired", "unemployed", "student"):
        personalization = (
            f"Their job profile ({job}) means budget pressure is plausible — back off if they push twice."
        )

    counter = obj.counter_template
    source = "rules"
    if generate_objection_response is not None and llm_available():
        llm_text = generate_objection_response(customer_said, customer, obj.category)
        if llm_text:
            counter = llm_text
            source = "groq-llama-3.3-70b"

    return {
        "category": obj.category,
        "counter": counter,
        "follow_up": obj.follow_up,
        "personalization_note": personalization,
        "source": source,
    }


def list_categories() -> list:
    return [o.category for o in OBJECTIONS]
