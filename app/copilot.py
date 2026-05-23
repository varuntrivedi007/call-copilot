"""Subscription Companion — agent-facing call copilot.

Pillar 01 rule: agent picks the customer (or call comes in inbound). Tool
reacts with confidence, drivers, conversation flow. No ranked lead lists.
"""

import sys
import textwrap
from pathlib import Path

import pandas as pd
import streamlit as st


def md(html: str):
    st.markdown(textwrap.dedent(html).strip(), unsafe_allow_html=True)

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from batch_score import normalize_uploaded, score_dataframe  # noqa: E402
from features import CATEGORICAL, NUMERIC, load_processed  # noqa: E402
from inference import Copilot  # noqa: E402
from llm import llm_available  # noqa: E402
from objection_handler import list_categories, respond  # noqa: E402


st.set_page_config(
    page_title="Subscription Companion",
    layout="wide",
    page_icon="•",
    initial_sidebar_state="expanded",
)


# ---- Design tokens ---------------------------------------------------------
COLOR_PRIMARY = "#4F46E5"          # indigo-600
COLOR_PRIMARY_SOFT = "#EEF2FF"     # indigo-50
COLOR_ACCENT = "#7C3AED"           # violet-600
COLOR_TEXT = "#0B1220"
COLOR_TEXT_SOFT = "#1E293B"
COLOR_MUTED = "#64748B"
COLOR_BORDER = "#E5E7EB"
COLOR_BORDER_SOFT = "#F1F5F9"
COLOR_CARD = "#FFFFFF"
COLOR_SURFACE = "#F7F8FB"
COLOR_HIGH = "#059669"
COLOR_HIGH_SOFT = "#ECFDF5"
COLOR_HIGH_BORDER = "#A7F3D0"
COLOR_MID = "#B45309"
COLOR_MID_SOFT = "#FFFBEB"
COLOR_MID_BORDER = "#FDE68A"
COLOR_LOW = "#475569"
COLOR_LOW_SOFT = "#F1F5F9"
COLOR_NEG = "#DC2626"


def inject_css():
    st.markdown(
        f"""
        <link rel="preconnect" href="https://fonts.googleapis.com">
        <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
        <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=Instrument+Serif&display=swap" rel="stylesheet">
        <style>
        :root {{
            --sc-primary: {COLOR_PRIMARY};
            --sc-accent: {COLOR_ACCENT};
            --sc-text: {COLOR_TEXT};
            --sc-muted: {COLOR_MUTED};
            --sc-border: {COLOR_BORDER};
            --sc-card: {COLOR_CARD};
            --sc-surface: {COLOR_SURFACE};
        }}
        html, body, [class*="css"], .stApp, .stMarkdown, .stText {{
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, "SF Pro Text", "Segoe UI", Helvetica, Arial, sans-serif;
            -webkit-font-smoothing: antialiased;
            -moz-osx-font-smoothing: grayscale;
        }}
        .stApp {{
            background:
                radial-gradient(900px 500px at 8% -10%, #EEF2FF 0%, rgba(238,242,255,0) 60%),
                radial-gradient(900px 600px at 92% 0%, #F5F3FF 0%, rgba(245,243,255,0) 55%),
                radial-gradient(700px 500px at 100% 100%, #ECFDF5 0%, rgba(236,253,245,0) 60%),
                {COLOR_SURFACE};
            background-attachment: fixed;
        }}
        .block-container {{
            padding-top: 1.6rem;
            padding-bottom: 5rem;
            max-width: 1320px;
        }}
        h1, h2, h3, h4 {{
            color: {COLOR_TEXT};
            letter-spacing: -0.018em;
            font-weight: 700;
        }}
        h1 {{ font-weight: 800; }}
        /* Hero */
        .sc-hero {{
            position: relative;
            overflow: hidden;
            background:
                radial-gradient(600px 300px at 0% 0%, rgba(79, 70, 229, 0.10), rgba(255,255,255,0) 60%),
                radial-gradient(500px 280px at 100% 100%, rgba(124, 58, 237, 0.10), rgba(255,255,255,0) 60%),
                linear-gradient(180deg, rgba(255,255,255,0.95) 0%, rgba(255,255,255,0.80) 100%);
            backdrop-filter: blur(10px);
            -webkit-backdrop-filter: blur(10px);
            border: 1px solid rgba(229, 231, 235, 0.7);
            border-radius: 20px;
            padding: 28px 32px;
            margin-bottom: 22px;
            box-shadow: 0 10px 30px -12px rgba(15, 23, 42, 0.08), 0 1px 2px rgba(15, 23, 42, 0.04);
        }}
        .sc-hero .sc-hero-row {{
            display: flex;
            align-items: center;
            gap: 18px;
        }}
        .sc-avatar {{
            flex: 0 0 auto;
            width: 56px;
            height: 56px;
            border-radius: 50%;
            background: linear-gradient(135deg, {COLOR_PRIMARY} 0%, {COLOR_ACCENT} 100%);
            color: #fff;
            display: flex;
            align-items: center;
            justify-content: center;
            font-weight: 700;
            font-size: 20px;
            letter-spacing: 0.02em;
            box-shadow: 0 6px 16px -6px rgba(79, 70, 229, 0.45);
        }}
        .sc-hero h1 {{
            margin: 0;
            font-size: 28px;
            color: {COLOR_TEXT};
            font-weight: 700;
            letter-spacing: -0.02em;
        }}
        .sc-hero .sc-hero-meta {{
            margin-top: 4px;
            color: {COLOR_MUTED};
            font-size: 14px;
        }}
        /* Cards */
        .sc-card {{
            background: rgba(255, 255, 255, 0.78);
            backdrop-filter: blur(8px);
            -webkit-backdrop-filter: blur(8px);
            border: 1px solid rgba(229, 231, 235, 0.8);
            border-radius: 16px;
            padding: 22px 24px;
            box-shadow: 0 4px 14px -8px rgba(15, 23, 42, 0.10), 0 1px 2px rgba(15, 23, 42, 0.03);
            margin-bottom: 16px;
            transition: box-shadow 200ms ease, transform 200ms ease;
        }}
        .sc-card:hover {{
            box-shadow: 0 10px 30px -14px rgba(15, 23, 42, 0.14);
        }}
        .sc-card-title {{
            font-size: 11px;
            font-weight: 600;
            color: {COLOR_MUTED};
            text-transform: uppercase;
            letter-spacing: 0.10em;
            margin-bottom: 10px;
        }}
        /* Confidence ring gauge */
        .sc-ring-wrap {{
            display: flex;
            align-items: center;
            gap: 18px;
        }}
        .sc-ring-text {{
            display: flex;
            flex-direction: column;
            line-height: 1.15;
        }}
        .sc-ring-pct {{
            font-size: 36px;
            font-weight: 800;
            letter-spacing: -0.02em;
            color: {COLOR_TEXT};
        }}
        .sc-ring-sub {{
            font-size: 12px;
            color: {COLOR_MUTED};
            margin-top: 2px;
        }}
        /* Pills */
        .sc-pill {{
            display: inline-flex;
            align-items: center;
            font-size: 11px;
            font-weight: 600;
            letter-spacing: 0.06em;
            text-transform: uppercase;
            padding: 5px 11px;
            border-radius: 999px;
            white-space: nowrap;
        }}
        .sc-pill::before {{
            content: '';
            width: 6px; height: 6px; border-radius: 50%;
            margin-right: 7px;
            background: currentColor;
            opacity: 0.85;
        }}
        .sc-pill-high {{ background: {COLOR_HIGH_SOFT}; color: {COLOR_HIGH}; border: 1px solid {COLOR_HIGH_BORDER}; }}
        .sc-pill-mid  {{ background: {COLOR_MID_SOFT}; color: {COLOR_MID}; border: 1px solid {COLOR_MID_BORDER}; }}
        .sc-pill-low  {{ background: {COLOR_LOW_SOFT}; color: {COLOR_LOW}; border: 1px solid {COLOR_BORDER}; }}
        /* Drivers */
        .sc-driver {{
            background: rgba(255, 255, 255, 0.92);
            border: 1px solid {COLOR_BORDER};
            border-left: 3px solid {COLOR_BORDER};
            border-radius: 12px;
            padding: 12px 14px;
            margin-bottom: 8px;
            transition: transform 150ms ease, border-color 150ms ease;
        }}
        .sc-driver:hover {{ transform: translateY(-1px); }}
        .sc-driver.pos {{ border-left-color: {COLOR_HIGH}; }}
        .sc-driver.neg {{ border-left-color: {COLOR_NEG}; }}
        .sc-driver .label {{ font-weight: 600; color: {COLOR_TEXT}; font-size: 14px; }}
        .sc-driver .meta  {{ color: {COLOR_MUTED}; font-size: 12px; margin-top: 4px; }}
        /* Stages */
        .sc-stage {{
            background: rgba(255, 255, 255, 0.82);
            border: 1px solid rgba(229, 231, 235, 0.8);
            border-radius: 16px;
            padding: 20px 22px;
            margin-bottom: 14px;
            box-shadow: 0 6px 18px -14px rgba(15, 23, 42, 0.10);
        }}
        .sc-stage h4 {{
            margin: 0 0 14px 0;
            font-size: 12px;
            font-weight: 700;
            color: {COLOR_PRIMARY};
            text-transform: uppercase;
            letter-spacing: 0.10em;
            display: flex; align-items: center; gap: 8px;
        }}
        .sc-script {{
            background: linear-gradient(135deg, #ECFDF5 0%, #F0FDF4 100%);
            border: 1px solid {COLOR_HIGH_BORDER};
            border-radius: 12px;
            padding: 16px 18px;
            font-size: 15.5px;
            line-height: 1.6;
            color: #064E3B;
            font-weight: 500;
        }}
        .sc-bullet {{
            background: {COLOR_PRIMARY_SOFT};
            border: 1px solid #C7D2FE;
            border-radius: 10px;
            padding: 12px 14px;
            font-size: 14px;
            line-height: 1.55;
            color: #312E81;
            margin-bottom: 8px;
        }}
        .sc-objection {{
            background: linear-gradient(135deg, {COLOR_MID_SOFT} 0%, #FEF3C7 100%);
            border: 1px solid {COLOR_MID_BORDER};
            border-radius: 10px;
            padding: 12px 14px;
            font-size: 14px;
            line-height: 1.55;
            color: #78350F;
            margin-bottom: 8px;
        }}
        .sc-meta-row {{
            color: {COLOR_MUTED};
            font-size: 12.5px;
            margin-bottom: 6px;
        }}
        .sc-divider {{
            height: 1px;
            background: linear-gradient(90deg, rgba(0,0,0,0) 0%, {COLOR_BORDER} 50%, rgba(0,0,0,0) 100%);
            margin: 28px 0;
            border: 0;
        }}
        /* Sidebar polish */
        section[data-testid="stSidebar"] {{
            background: linear-gradient(180deg, #FFFFFF 0%, #FAFAFE 100%);
            border-right: 1px solid {COLOR_BORDER};
        }}
        section[data-testid="stSidebar"] .stRadio, section[data-testid="stSidebar"] .stSelectbox, section[data-testid="stSidebar"] .stTextInput {{
            margin-top: 8px;
        }}
        /* Buttons */
        .stButton > button {{
            border-radius: 10px;
            border: 1px solid {COLOR_BORDER};
            background: #fff;
            color: {COLOR_TEXT};
            font-weight: 600;
            transition: all 150ms ease;
        }}
        .stButton > button:hover {{
            border-color: {COLOR_PRIMARY};
            color: {COLOR_PRIMARY};
            transform: translateY(-1px);
            box-shadow: 0 6px 14px -8px rgba(79, 70, 229, 0.35);
        }}
        .stButton > button[kind="primary"] {{
            background: linear-gradient(135deg, {COLOR_PRIMARY} 0%, {COLOR_ACCENT} 100%);
            color: #FFFFFF;
            border-color: transparent;
            box-shadow: 0 8px 20px -10px rgba(79, 70, 229, 0.55);
        }}
        .stButton > button[kind="primary"]:hover {{
            color: #FFFFFF;
            transform: translateY(-1px);
            box-shadow: 0 12px 28px -10px rgba(124, 58, 237, 0.55);
        }}
        /* Inputs */
        .stTextInput input, .stNumberInput input, .stSelectbox div[data-baseweb="select"] > div {{
            border-radius: 10px !important;
        }}
        /* Tables */
        [data-testid="stDataFrame"] {{
            border-radius: 14px;
            overflow: hidden;
            border: 1px solid {COLOR_BORDER};
            box-shadow: 0 4px 14px -8px rgba(15, 23, 42, 0.06);
        }}
        /* Expander */
        details[data-testid="stExpander"] {{
            background: rgba(255, 255, 255, 0.72);
            border: 1px solid {COLOR_BORDER};
            border-radius: 12px;
        }}
        /* Section header */
        .sc-section-title {{
            font-size: 22px;
            font-weight: 700;
            color: {COLOR_TEXT};
            letter-spacing: -0.018em;
            margin: 0 0 14px 0;
            display: flex; align-items: center; gap: 10px;
        }}
        .sc-section-title::before {{
            content: '';
            display: inline-block;
            width: 4px; height: 22px;
            border-radius: 4px;
            background: linear-gradient(180deg, {COLOR_PRIMARY} 0%, {COLOR_ACCENT} 100%);
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_resource
def get_copilot():
    return Copilot()


@st.cache_data
def get_data():
    return load_processed()


def bucket_for(p: float) -> str:
    if p >= 0.5:
        return "high"
    if p >= 0.2:
        return "medium"
    return "low"


def bucket_color(p: float) -> str:
    b = bucket_for(p)
    return {"high": COLOR_HIGH, "medium": COLOR_MID, "low": COLOR_LOW}[b]


def bucket_pill(p: float) -> str:
    b = bucket_for(p)
    klass = {"high": "sc-pill-high", "medium": "sc-pill-mid", "low": "sc-pill-low"}[b]
    label = {"high": "Strong signal", "medium": "Mixed signal", "low": "Faint signal"}[b]
    return f'<span class="sc-pill {klass}">{label}</span>'


def bucket_fit_label(p: float) -> str:
    b = bucket_for(p)
    return {
        "high": "Strong historical fit",
        "medium": "Mixed historical fit",
        "low": "Lower historical fit",
    }[b]


def _initials(name: str) -> str:
    parts = [p for p in name.strip().split() if p]
    if not parts:
        return "•"
    if len(parts) == 1:
        return parts[0][:2].upper()
    return (parts[0][0] + parts[-1][0]).upper()


def render_hero(cust: dict, result: dict):
    name = (cust.get("name") or "").strip()
    display_name = name if name else "Customer"
    initials = _initials(name) if name else "•"
    job = str(cust.get("job", "")).replace("_", " ").replace(".", "")
    age = cust.get("age")
    profile_bits = []
    if age:
        try:
            profile_bits.append(f"age {int(age)}")
        except (TypeError, ValueError):
            pass
    if job:
        profile_bits.append(job)
    marital = str(cust.get("marital", "")).lower()
    if marital and marital != "unknown":
        profile_bits.append(marital)
    education = str(cust.get("education", "")).replace("_", " ")
    if education and education.lower() != "unknown":
        profile_bits.append(education)
    profile_line = " · ".join(profile_bits) or "no profile data"

    md(
        f'<div class="sc-hero">'
        f'<div class="sc-hero-row">'
        f'<div class="sc-avatar">{initials}</div>'
        f'<div style="flex:1">'
        f'<h1>{display_name}</h1>'
        f'<div class="sc-hero-meta">{profile_line}</div>'
        f'</div>'
        f'<div>{bucket_pill(result["probability"])}</div>'
        f'</div>'
        f'</div>'
    )


def render_gauge(p: float, threshold: float):
    pct = int(round(p * 100))
    color = bucket_color(p)
    radius = 38
    circumference = 2 * 3.14159 * radius
    progress = max(0.0, min(1.0, p))
    dash = circumference * progress
    gap = circumference - dash
    svg = (
        f'<svg width="96" height="96" viewBox="0 0 96 96" style="flex:0 0 auto">'
        f'<circle cx="48" cy="48" r="{radius}" fill="none" stroke="#E5E7EB" stroke-width="8"/>'
        f'<circle cx="48" cy="48" r="{radius}" fill="none" stroke="{color}" stroke-width="8" '
        f'stroke-linecap="round" stroke-dasharray="{dash:.2f} {gap:.2f}" transform="rotate(-90 48 48)"/>'
        f'<text x="48" y="52" text-anchor="middle" font-size="20" font-weight="700" fill="{COLOR_TEXT}" '
        f'font-family="Inter, sans-serif">{pct}%</text>'
        f'</svg>'
    )
    md(
        f'<div class="sc-card">'
        f'<div class="sc-card-title">Confidence score</div>'
        f'<div class="sc-ring-wrap">'
        f'{svg}'
        f'<div class="sc-ring-text">'
        f'<div class="sc-ring-pct" style="color:{color}">{pct}%</div>'
        f'<div class="sc-ring-sub">Reference rate from past calls with similar profiles.</div>'
        f'</div></div></div>'
    )


def render_secondary_stats(result: dict):
    fit = bucket_fit_label(result["probability"])
    md(f"""
        <div class="sc-card">
          <div class="sc-card-title">Historical fit</div>
          <div style="font-size:20px;font-weight:600;color:{COLOR_TEXT}">{fit}</div>
          <div class="sc-ring-sub" style="margin-top:4px">Pattern from similar past customers — not a verdict on this person.</div>
        </div>
    """)


def render_drivers_panel(title: str, drivers: list, polarity: str):
    klass = "pos" if polarity == "+" else "neg"
    arrow = "▲" if polarity == "+" else "▼"
    if not drivers:
        items = '<div class="sc-meta-row">No drivers surfaced.</div>'
    else:
        items = "".join(
            f'<div class="sc-driver {klass}"><div class="label">{arrow} {d["driver"]}</div>'
            f'<div class="meta">value: {d["value"]} · shap impact: {d["shap"]:+.3f}</div></div>'
            for d in drivers
        )
    html = (
        f'<div class="sc-card">'
        f'<div class="sc-card-title">{title}</div>'
        f'{items}'
        f'</div>'
    )
    st.markdown(html, unsafe_allow_html=True)


def render_flow(flow: dict):
    src = flow.get("flow_source", "rules")
    if src.startswith("groq"):
        engine_badge = '<span class="sc-pill sc-pill-high">Generated by Groq Llama 3.3</span>'
    else:
        engine_badge = '<span class="sc-pill sc-pill-low">Rule-based fallback</span>'
    md(
        f'<div class="sc-meta-row" style="display:flex;align-items:center;gap:10px">'
        f'<span>Confidence: {flow["confidence_bucket"].title()} · Profile: {flow["profile_summary"]}</span>'
        f'{engine_badge}'
        f'</div>'
    )

    md(
        f'<div class="sc-stage">'
        f'<h4>Stage 1 · Opener</h4>'
        f'<div class="sc-script">{flow["stage_1_opener"]}</div>'
        f'</div>'
    )

    pitch = flow["stage_2_pitch"]
    flow_src = flow.get("flow_source", pitch.get("source", "rules"))
    drivers_html = ""
    if flow_src == "rules" and pitch.get("drivers"):
        chips = " ".join(
            f'<span class="sc-pill sc-pill-low" style="margin-right:6px">{d}</span>'
            for d in pitch["drivers"]
        )
        drivers_html = f'<div class="sc-meta-row" style="margin-bottom:10px">Built from: {chips}</div>'
    source_badge = (
        '<span class="sc-pill sc-pill-high" style="margin-left:8px">AI-generated</span>'
        if flow_src.startswith("groq")
        else '<span class="sc-pill sc-pill-low" style="margin-left:8px">Rule-based</span>'
    )
    md(
        f'<div class="sc-stage">'
        f'<h4>Stage 2 · Pitch (read top to bottom){source_badge}</h4>'
        f'{drivers_html}'
        f'<div class="sc-script">{pitch["paragraph"]}</div>'
        f'</div>'
    )

    q_html = "".join(
        f'<div class="sc-bullet">{q}</div>' for q in flow["stage_3_diagnostic_questions"]
    )
    md(
        f'<div class="sc-stage">'
        f'<h4>Stage 3 · Diagnostic questions</h4>'
        f'{q_html}'
        f'</div>'
    )

    op_html = "".join(
        f'<div style="margin-bottom:10px">'
        f'<div class="sc-meta-row"><strong>Risk:</strong> {op["driver"]}</div>'
        f'<div class="sc-objection">{op["say"]}</div>'
        f'</div>'
        for op in flow["stage_4_objection_preempts"]
    )
    md(
        f'<div class="sc-stage">'
        f'<h4>Stage 4 · Objection pre-empts</h4>'
        f'{op_html}'
        f'</div>'
    )

    md(
        f'<div class="sc-stage">'
        f'<h4>Stage 5 · Close</h4>'
        f'<div class="sc-script">{flow["stage_5_close"]}</div>'
        f'</div>'
    )


def render_objection_handler(customer: dict):
    st.markdown(
        '<div class="sc-card-title" style="font-size:13px;margin-bottom:4px">Live objection handler</div>',
        unsafe_allow_html=True,
    )
    st.caption("Mid-call: type or paste what the customer just said. Tool suggests a response.")

    col1, col2 = st.columns([3, 1])
    with col1:
        said = st.text_input(
            "Customer said",
            value="",
            placeholder="e.g. 'I don't have time right now' or 'sounds too risky'",
            key="objection_input",
            label_visibility="collapsed",
        )
    with col2:
        quick = st.selectbox(
            "Or pick category",
            ["(text input)"] + list_categories(),
            key="objection_quick",
            label_visibility="collapsed",
        )

    text_to_use = said
    if quick != "(text input)" and not said:
        text_to_use = quick.replace("_", " ")

    if not text_to_use:
        st.markdown(
            '<div class="sc-bullet">Waiting for input.</div>',
            unsafe_allow_html=True,
        )
        return

    result = respond(text_to_use, customer)
    extra = ""
    if result.get("personalization_note"):
        extra = (
            f'<div class="sc-objection" style="margin-top:8px">'
            f'<strong>Personalization:</strong> {result["personalization_note"]}</div>'
        )
    src = result.get("source", "rules")
    badge = (
        '<span class="sc-pill sc-pill-high" style="margin-left:8px">AI-generated</span>'
        if src.startswith("groq")
        else '<span class="sc-pill sc-pill-low" style="margin-left:8px">Rule-based</span>'
    )
    md(
        f'<div class="sc-stage">'
        f'<h4>Detected: {result["category"].replace("_", " ")}{badge}</h4>'
        f'<div class="sc-script">{result["counter"]}</div>'
        f'<div class="sc-meta-row" style="margin-top:10px"><strong>Follow-up:</strong> {result["follow_up"]}</div>'
        f'{extra}'
        f'</div>'
    )


def build_customer_form(df: pd.DataFrame) -> dict:
    st.markdown(
        '<div class="sc-card-title" style="font-size:13px">Manual customer entry (advanced)</div>',
        unsafe_allow_html=True,
    )
    cust = {}
    cols = st.columns(3)
    field_cols = list(CATEGORICAL + NUMERIC)
    for i, feat in enumerate(field_cols):
        with cols[i % 3]:
            if feat in CATEGORICAL:
                options = sorted(df[feat].dropna().astype(str).unique().tolist())
                cust[feat] = st.selectbox(feat, options, key=f"manual_{feat}")
            else:
                if feat in df.columns and df[feat].notna().any():
                    default = float(df[feat].median())
                else:
                    default = 0.0
                cust[feat] = st.number_input(feat, value=default, key=f"manual_{feat}")
    return cust


def new_customer_form(df: pd.DataFrame) -> dict:
    """Agent-friendly form for customers NOT in the dataset.

    Only asks for fields the agent would realistically know. Macro-economic
    features and campaign-history defaults are auto-filled.
    """
    st.markdown(
        '<div class="sc-card-title" style="font-size:13px">New customer details</div>',
        unsafe_allow_html=True,
    )
    st.caption("Fill what you know. The system auto-fills macro indicators and first-contact defaults.")

    cust = {}
    row1 = st.columns(3)
    with row1[0]:
        cust["age"] = st.number_input("Age", min_value=18, max_value=99, value=35, key="new_age")
    with row1[1]:
        jobs = sorted(df["job"].dropna().astype(str).unique().tolist())
        cust["job"] = st.selectbox("Job", jobs, index=0, key="new_job")
    with row1[2]:
        marital = sorted(df["marital"].dropna().astype(str).unique().tolist())
        cust["marital"] = st.selectbox("Marital status", marital, index=0, key="new_marital")

    row2 = st.columns(3)
    with row2[0]:
        edu = sorted(df["education"].dropna().astype(str).unique().tolist())
        cust["education"] = st.selectbox("Education", edu, index=0, key="new_education")
    with row2[1]:
        cust["housing"] = st.selectbox("Housing loan with us?", ["no", "yes", "unknown"], index=0, key="new_housing")
    with row2[2]:
        cust["loan"] = st.selectbox("Personal loan with us?", ["no", "yes", "unknown"], index=0, key="new_loan")

    row3 = st.columns(3)
    with row3[0]:
        cust["default"] = st.selectbox("Credit default history?", ["no", "yes", "unknown"], index=0, key="new_default")
    with row3[1]:
        cust["contact"] = st.selectbox("Contact channel", ["cellular", "telephone"], index=0, key="new_contact")
    with row3[2]:
        st.markdown(
            f'<div style="font-size:12px;color:{COLOR_MUTED};padding-top:28px">'
            f'Campaign + macro fields auto-filled.</div>',
            unsafe_allow_html=True,
        )

    # First-contact defaults
    cust["poutcome"] = "nonexistent"
    cust["previous"] = 0
    cust["campaign"] = 1
    cust["was_contacted_before"] = 0
    cust["pdays_clean"] = float("nan")

    # Seasonality from last known row
    last_row = df.iloc[-1]
    cust["month"] = str(last_row.get("month", "may"))
    cust["day_of_week"] = str(last_row.get("day_of_week", "mon"))

    # Macro features: median of last 1000 rows = recent-period proxy
    recent = df.tail(1000)
    for macro in ["emp_var_rate", "cons_price_idx", "cons_conf_idx", "euribor3m", "nr_employed"]:
        if macro in df.columns:
            cust[macro] = float(recent[macro].median())

    with st.expander("Auto-filled fields (review or override later)"):
        auto_view = {
            k: str(cust[k]) for k in
            ["poutcome", "previous", "campaign", "was_contacted_before", "pdays_clean",
             "month", "day_of_week", "emp_var_rate", "cons_price_idx",
             "cons_conf_idx", "euribor3m", "nr_employed"]
            if k in cust
        }
        st.dataframe(
            pd.DataFrame({"value": auto_view}).reset_index().rename(columns={"index": "field"}),
            hide_index=True,
            use_container_width=True,
        )

    return cust


def batch_upload_view(reference_df: pd.DataFrame, copilot):
    """Bulk customer list view.

    Upload a CSV → rows shown immediately (no scoring upfront).
    Click any row → that single customer is scored and the copilot opens.

    Pillar 01 compliant: rows are NOT sorted by probability (no probability
    column visible in the picker), so the tool does not present a ranked
    call list.
    """
    md(
        '<div class="sc-hero">'
        '<h1>Batch upload</h1>'
        '<p>Upload a CSV of customers. The full list loads instantly. '
        'Click a row to score that customer and open the copilot.</p>'
        '</div>'
    )

    uploaded = st.file_uploader("Upload customer CSV", type=["csv"], key="batch_upload")

    if "batch_raw" not in st.session_state:
        st.session_state.batch_raw = None
        st.session_state.batch_selected_idx = None

    if uploaded is not None:
        try:
            raw = pd.read_csv(uploaded)
        except Exception as e:
            st.error(f"Could not read CSV: {e}")
            return None
        normalized = normalize_uploaded(raw, reference_df)
        if "name" not in normalized.columns:
            normalized["name"] = [f"Customer {i}" for i in range(len(normalized))]
        st.session_state.batch_raw = normalized
        st.session_state.batch_selected_idx = None  # reset selection on new upload

    raw = st.session_state.batch_raw
    if raw is None:
        st.info("Drop a CSV above. Required-ish columns: age, job, marital, education, default, housing, loan, contact, month, day_of_week, campaign, pdays, previous, poutcome. Macro and missing columns are auto-filled.")
        return None

    # Filters
    filter_cols = st.columns([3, 1])
    with filter_cols[0]:
        query = st.text_input("Search by name or job", value="", key="batch_search")
    with filter_cols[1]:
        sort_choice = st.selectbox(
            "Sort by",
            ["name", "row order"],
            index=0,
            key="batch_sort",
        )

    view = raw.copy()
    view["_orig_idx"] = view.index
    if query.strip():
        q = query.strip().lower()
        view = view[
            view["name"].astype(str).str.lower().str.contains(q)
            | view["job"].astype(str).str.lower().str.contains(q)
        ]
    if sort_choice == "name":
        view = view.sort_values("name", kind="stable")
    else:
        view = view.sort_values("_orig_idx", kind="stable")

    view = view.reset_index(drop=True)

    md(
        '<div class="sc-meta-row" style="margin-top:6px">'
        'Tip: select a row in the table to score that customer and open the copilot. '
        'The list itself is not ranked by probability — score is only revealed on click.'
        '</div>'
    )

    display_cols = ["name", "age", "job", "marital", "education", "housing", "loan", "contact"]
    available_cols = [c for c in display_cols if c in view.columns]

    selection = st.dataframe(
        view[available_cols],
        hide_index=True,
        use_container_width=True,
        height=460,
        on_select="rerun",
        selection_mode="single-row",
        key="batch_table",
    )

    selected_rows = selection.get("selection", {}).get("rows", []) if isinstance(selection, dict) else []
    if not selected_rows:
        return None

    picked_view_idx = selected_rows[0]
    orig_idx = int(view.iloc[picked_view_idx]["_orig_idx"])
    cust = raw.iloc[orig_idx].to_dict()
    if not cust.get("name") or pd.isna(cust.get("name")):
        cust["name"] = f"Customer {orig_idx}"

    st.success(f"Opening copilot for {cust['name']}…")
    return cust


def main():
    inject_css()

    df = get_data()
    copilot = get_copilot()

    # Sidebar -----------------------------------------------------------------
    st.sidebar.markdown(
        f"""
        <div style="padding:6px 0 14px 0">
          <div style="font-weight:700;font-size:18px;color:{COLOR_TEXT}">Subscription Companion</div>
          <div style="font-size:12px;color:{COLOR_MUTED};margin-top:2px">Call-center copilot</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    mode = st.sidebar.radio(
        "Customer source",
        ["Pick from dataset", "New customer", "Batch upload", "Manual entry (advanced)"],
    )

    if mode == "Pick from dataset":
        if "row_idx_input" not in st.session_state:
            st.session_state.row_idx_input = 3

        st.sidebar.markdown("**Quick demo customers**")
        demo_col1, demo_col2, demo_col3 = st.sidebar.columns(3)
        if demo_col1.button("High 67%"):
            st.session_state.row_idx_input = 37234
            st.rerun()
        if demo_col2.button("Mid 35%"):
            st.session_state.row_idx_input = 37203
            st.rerun()
        if demo_col3.button("Low 0%"):
            st.session_state.row_idx_input = 10277
            st.rerun()

        idx = st.sidebar.number_input(
            "Row index",
            min_value=0,
            max_value=len(df) - 1,
            step=1,
            key="row_idx_input",
        )
        cust = df.iloc[int(idx)].to_dict()
        st.sidebar.caption(f"Active row: {int(idx)} · True label: {cust['y']}")
    elif mode == "New customer":
        cust = new_customer_form(df)
    elif mode == "Batch upload":
        cust = batch_upload_view(df, copilot)
        if cust is None:
            return  # batch_upload_view already rendered table; no single customer selected yet
    else:
        cust = build_customer_form(df)

    st.sidebar.divider()
    name_input = st.sidebar.text_input(
        "Customer name (optional)", value="", placeholder="e.g. Maria"
    )
    if name_input.strip():
        cust["name"] = name_input.strip()

    st.sidebar.divider()
    ai_status = "Groq Llama 3.3 connected" if llm_available() else "Offline (rule-based)"
    ai_color = COLOR_HIGH if llm_available() else COLOR_MUTED
    st.sidebar.markdown(
        f'<div style="font-size:12px;color:{COLOR_MUTED}">AI engine</div>'
        f'<div style="font-size:13px;color:{ai_color};font-weight:600">{ai_status}</div>',
        unsafe_allow_html=True,
    )

    # Main area ---------------------------------------------------------------
    result = copilot.predict(cust)

    render_hero(cust, result)

    top_cols = st.columns([1, 1, 1])
    with top_cols[0]:
        render_gauge(result["probability"], copilot.threshold)
    with top_cols[1]:
        render_drivers_panel("Why likely YES", result["positive_drivers"], "+")
    with top_cols[2]:
        render_drivers_panel("Why likely NO", result["negative_drivers"], "-")

    st.markdown('<div class="sc-divider"></div>', unsafe_allow_html=True)

    st.markdown(
        f'<div class="sc-section-title">What to say</div>',
        unsafe_allow_html=True,
    )
    render_flow(result["flow"])

    st.markdown('<div class="sc-divider"></div>', unsafe_allow_html=True)
    st.markdown(
        f'<div class="sc-section-title">Live objection handling</div>',
        unsafe_allow_html=True,
    )
    render_objection_handler(cust)

    st.markdown('<div class="sc-divider"></div>', unsafe_allow_html=True)

    bottom_cols = st.columns([2, 1])
    with bottom_cols[0]:
        st.markdown(
            '<div class="sc-card-title" style="font-size:13px">Customer profile (raw fields)</div>',
            unsafe_allow_html=True,
        )
        profile_view = {k: str(cust.get(k)) for k in CATEGORICAL + NUMERIC}
        st.dataframe(
            pd.DataFrame({"value": profile_view}).reset_index().rename(
                columns={"index": "field"}
            ),
            hide_index=True,
            use_container_width=True,
        )
    with bottom_cols[1]:
        render_secondary_stats(result)
        with st.expander("Model card"):
            st.json(copilot.metrics)


if __name__ == "__main__":
    main()
