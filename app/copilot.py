"""Subscription Companion — agent-facing call copilot.

Pillar 01 rule: agent picks the customer (or call comes in inbound). Tool
reacts with confidence, drivers, conversation flow. No ranked lead lists.
"""

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from features import CATEGORICAL, NUMERIC, load_processed  # noqa: E402
from inference import Copilot  # noqa: E402


st.set_page_config(page_title="Subscription Companion", layout="wide")


@st.cache_resource
def get_copilot():
    return Copilot()


@st.cache_data
def get_data():
    return load_processed()


def gauge_color(p: float) -> str:
    if p >= 0.5:
        return "#1f8a3a"
    if p >= 0.2:
        return "#c98a00"
    return "#9a9a9a"


def render_gauge(p: float):
    pct = int(round(p * 100))
    color = gauge_color(p)
    st.markdown(
        f"""
        <div style="border:1px solid #ddd;border-radius:10px;padding:18px 24px;background:#fafafa">
          <div style="font-size:13px;color:#666;letter-spacing:0.5px;text-transform:uppercase">Subscribe probability (calibrated)</div>
          <div style="font-size:48px;font-weight:700;color:{color};margin-top:4px">{pct}%</div>
          <div style="font-size:13px;color:#666">Above {int(round(get_copilot().threshold*100))}% → predicted YES</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_drivers(title: str, drivers: list, polarity: str):
    st.markdown(f"**{title}**")
    if not drivers:
        st.caption("None surfaced.")
        return
    for d in drivers:
        sign = "▲" if polarity == "+" else "▼"
        st.markdown(
            f"- {sign} **{d['driver']}**  \n  &nbsp;&nbsp;_value: `{d['value']}` · shap: `{d['shap']:+.3f}`_"
        )


def render_flow(flow: dict):
    st.subheader("What to say")
    st.caption(
        f"Confidence: `{flow['confidence_bucket']}` · Profile: {flow['profile_summary']}"
    )

    st.markdown("### 1. Opener — read this out loud")
    st.success(flow["stage_1_opener"])

    st.markdown("### 2. Talking points — use after they engage")
    for tp in flow["stage_2_talking_points"]:
        with st.container():
            st.markdown(f"**Why:** {tp['driver']}")
            st.info(tp["say"])

    st.markdown("### 3. Diagnostic questions — keep them talking")
    for q in flow["stage_3_diagnostic_questions"]:
        st.markdown(f"- {q}")

    st.markdown("### 4. If they push back — pre-empts")
    for op in flow["stage_4_objection_preempts"]:
        with st.container():
            st.markdown(f"**Risk:** {op['driver']}")
            st.warning(op["say"])

    st.markdown("### 5. Close — read this out loud")
    st.success(flow["stage_5_close"])


def build_customer_form(df: pd.DataFrame) -> dict:
    st.subheader("Manual entry")
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


def main():
    st.title("Subscription Companion")
    st.caption(
        "Agent picks the customer. Tool surfaces context, confidence, and what to say. "
        "Not a robocaller. Not a call-priority queue."
    )

    df = get_data()
    copilot = get_copilot()

    mode = st.sidebar.radio("Customer source", ["Pick from dataset", "Manual entry"])

    if mode == "Pick from dataset":
        st.sidebar.write(f"Dataset rows: {len(df):,}")
        idx = st.sidebar.number_input(
            "Row index", min_value=0, max_value=len(df) - 1, value=3, step=1
        )
        cust = df.iloc[int(idx)].to_dict()
        st.sidebar.write(f"True label in data: `{cust['y']}`")
    else:
        cust = build_customer_form(df)

    st.divider()

    profile_cols = st.columns([2, 1])
    with profile_cols[0]:
        st.subheader("Customer profile")
        profile_view = {k: cust.get(k) for k in CATEGORICAL + NUMERIC}
        st.dataframe(pd.DataFrame([profile_view]).T.rename(columns={0: "value"}))

    with profile_cols[1]:
        result = copilot.predict(cust)
        render_gauge(result["probability"])
        st.caption(
            f"Raw model: {result['raw_probability']*100:.1f}% · "
            f"label: **{result['label']}**"
        )

    st.divider()
    cols = st.columns(2)
    with cols[0]:
        render_drivers("Why likely YES", result["positive_drivers"], "+")
    with cols[1]:
        render_drivers("Why likely NO", result["negative_drivers"], "-")

    st.divider()
    render_flow(result["flow"])

    st.divider()
    with st.expander("Model card"):
        st.json(copilot.metrics)


if __name__ == "__main__":
    main()
