import json
import os
from functools import lru_cache
from typing import Optional

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

try:
    from groq import Groq
except ImportError:
    Groq = None  # noqa: N816


MODEL = "llama-3.3-70b-versatile"
TIMEOUT_S = 12
MAX_TOKENS = 600


def _client() -> Optional["Groq"]:
    key = os.environ.get("GROQ_API_KEY")
    if not key or Groq is None:
        return None
    try:
        return Groq(api_key=key, timeout=TIMEOUT_S)
    except Exception:
        return None


def _format_drivers(pos: list, neg: list) -> str:
    lines = []
    for d in pos:
        lines.append(f"+ {d['driver']} (value={d['value']}, impact={d['shap']:+.3f})")
    for d in neg:
        lines.append(f"- {d['driver']} (value={d['value']}, impact={d['shap']:+.3f})")
    return "\n".join(lines) if lines else "(no notable drivers)"


def _customer_card(customer: dict) -> str:
    fields = [
        ("Name", customer.get("name")),
        ("Age", customer.get("age")),
        ("Job", customer.get("job")),
        ("Marital", customer.get("marital")),
        ("Education", customer.get("education")),
        ("Housing loan", customer.get("housing")),
        ("Personal loan", customer.get("loan")),
        ("Contact channel", customer.get("contact")),
        ("Prior outcome", customer.get("poutcome")),
        ("Times contacted this campaign", customer.get("campaign")),
        ("Prior contacts", customer.get("previous")),
    ]
    return "\n".join(f"{k}: {v}" for k, v in fields if v not in (None, "", "unknown"))


_RESPONSE_CACHE: dict = {}


def _call(prompt_key: str, system: str, user: str, max_tokens: int = MAX_TOKENS) -> Optional[str]:
    if prompt_key in _RESPONSE_CACHE:
        return _RESPONSE_CACHE[prompt_key]
    client = _client()
    if client is None:
        return None
    try:
        resp = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            max_tokens=max_tokens,
            temperature=0.6,
        )
        text = resp.choices[0].message.content.strip()
        if text:
            _RESPONSE_CACHE[prompt_key] = text
        return text or None
    except Exception as e:
        print(f"[llm] call failed: {e}")
        return None


def _cached_call(prompt_key: str, system: str, user: str) -> Optional[str]:
    return _call(prompt_key, system, user)


def generate_pitch(customer: dict, probability: float, pos: list, neg: list, bucket: str) -> Optional[str]:
    system = (
        "You are a senior call-center coach writing a single conversational pitch paragraph "
        "for a human agent to read aloud after their opener. The agent is selling a fixed-rate "
        "term-deposit product to an existing bank customer.\n\n"
        "Rules:\n"
        "- One flowing paragraph (4-7 sentences). No bullet points, no headings.\n"
        "- Sound like a real human salesperson, not a chatbot. Use contractions.\n"
        "- Never say 'short version', 'in simple terms', 'plain English', 'simply put'.\n"
        "- Refer to the customer by name if provided. Otherwise no salutation.\n"
        "- Lead with the strongest positive driver as the reason for the call.\n"
        "- Weave in one or two more drivers naturally. Don't list them mechanically.\n"
        "- Acknowledge a negative driver only if it materially shapes the pitch.\n"
        "- Do not invent facts about the product. Stick to: fixed rate, principal protected, "
        "term length 3-24 months, agree rate up front."
    )
    user = (
        f"Customer profile:\n{_customer_card(customer)}\n\n"
        f"Confidence score: {int(round(probability*100))}% (bucket={bucket})\n\n"
        f"Top drivers from the model:\n{_format_drivers(pos, neg)}\n\n"
        f"Write the pitch paragraph now. Output only the paragraph text, no preamble."
    )
    key = json.dumps({"c": customer.get("name", ""), "p": int(probability * 1000),
                      "drivers": [d["driver"] for d in pos + neg]}, sort_keys=True)
    return _cached_call(key, system, user)


def generate_objection_response(customer_said: str, customer: dict, category: str) -> Optional[str]:
    system = (
        "You are a senior call-center coach. The agent is mid-call. The customer just said "
        "something. Write the agent's reply.\n\n"
        "Rules:\n"
        "- One or two sentences max. Conversational.\n"
        "- No 'I understand your concern' or other call-center cliches.\n"
        "- Do not promise anything not in the product spec: fixed rate, principal protected, "
        "3-24 month term.\n"
        "- Output only the agent's spoken reply, in quotes."
    )
    user = (
        f"Customer profile:\n{_customer_card(customer)}\n\n"
        f"Customer just said: \"{customer_said}\"\n"
        f"Detected objection category: {category}\n\n"
        f"Write the agent's reply."
    )
    key = json.dumps({"said": customer_said, "cat": category,
                      "name": customer.get("name", "")}, sort_keys=True)
    return _cached_call(key, system, user)


def generate_full_flow(customer: dict, probability: float, pos: list, neg: list, bucket: str) -> Optional[dict]:
    """One LLM call returns the full conversation flow as structured JSON."""
    system = (
        "You are a senior call-center coach. You output strict JSON only, no prose.\n"
        "Generate a tailored conversation flow for a bank call-center agent calling a customer "
        "about a fixed-rate term-deposit product.\n\n"
        "Rules:\n"
        "- Output JSON with keys: opener (string), pitch (string), diagnostic_questions (array of 2-3 strings), "
        "objection_preempts (array of 2-3 strings), close (string).\n"
        "- Each spoken line must be in double quotes inside the string (so the agent reads it verbatim).\n"
        "- No bullet points, no headings, no markdown.\n"
        "- Sound like a real human salesperson. Use contractions. No 'short version' / 'in simple terms' / 'plain English'.\n"
        "- Address customer by name if provided.\n"
        "- Tailor tone to the confidence bucket: high=warm direct, medium=exploratory, low=soft no-pressure.\n"
        "- Use the model drivers as evidence baked naturally into the pitch and objection pre-empts.\n"
        "- Product facts allowed: fixed rate locked in today, principal protected, term length 3-24 months. No other product facts.\n"
        "- Output ONLY valid JSON. Do not wrap in ```json or any text."
    )
    user = (
        f"Customer profile:\n{_customer_card(customer)}\n\n"
        f"Confidence: {int(round(probability*100))}% (bucket={bucket})\n\n"
        f"Drivers:\n{_format_drivers(pos, neg)}\n\n"
        "Return the JSON now."
    )
    key = json.dumps({
        "type": "full_flow",
        "name": customer.get("name", ""),
        "p_bin": int(probability * 20),
        "bucket": bucket,
        "drivers": sorted([d["driver"] for d in pos + neg]),
        "job": customer.get("job"),
        "marital": customer.get("marital"),
        "edu": customer.get("education"),
        "housing": customer.get("housing"),
        "poutcome": customer.get("poutcome"),
    }, sort_keys=True)

    raw = _call(key, system, user, max_tokens=900)
    if not raw:
        return None
    
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("```", 2)[1]
        if cleaned.startswith("json"):
            cleaned = cleaned[4:]
        cleaned = cleaned.strip("`\n ")
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        return None
    required = {"opener", "pitch", "diagnostic_questions", "objection_preempts", "close"}
    if not required.issubset(data.keys()):
        return None
    return data


def llm_available() -> bool:
    return _client() is not None
