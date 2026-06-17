"""
Duckstradamus — Smart Charging (Streamlit)
==========================================
Run:
    cd UI
    python generate_data.py     # once, to create data.json
    streamlit run app.py

Reads data.json (predicted / actual / naive hourly prices for all of 2024).
"""

import json
import time
from pathlib import Path

import numpy as np
import streamlit as st
import plotly.graph_objects as go

# ────────────────────────── page config + theme ──────────────────────────────

st.set_page_config(
    page_title="Duckstradamus — Smart Charging",
    page_icon="⚡",
    layout="wide",
)

ACCENT  = "#ff9f1c"
TEAL    = "#2ec4b6"
NAIVE   = "#9b8cff"
BG      = "#0f1419"
PANEL   = "#1a2129"
TEXTDIM = "#8a97a6"

st.markdown(f"""
<style>
  .stApp {{ background: {BG}; }}
  h1, h2, h3, p, label, .stMarkdown {{ color: #e8edf2; }}
  div[data-testid="stMetric"] {{
    background: {PANEL};
    border: 1px solid #2c3744;
    border-radius: 12px;
    padding: 16px 18px;
  }}
  div[data-testid="stMetricValue"] {{ font-size: 26px; }}
  .block-container {{ padding-top: 2rem; }}
  .duck-title {{ font-size: 30px; font-weight: 700; }}
  .duck-title span {{ color: {ACCENT}; }}
</style>
""", unsafe_allow_html=True)

# ────────────────────────── load data ────────────────────────────────────────

@st.cache_data
def load_data():
    path = Path(__file__).parent / "data.json"
    if not path.exists():
        return None
    return json.loads(path.read_text())

DATA = load_data()

if DATA is None:
    st.error("`data.json` not found. Run `python generate_data.py` first.")
    st.stop()

DEMO_DAY  = DATA["demo_day"]
DAYS_2024 = DATA["days_2024"]
HOURS = [f"{h:02d}:00" for h in range(24)]

# ────────────────────────── helpers ──────────────────────────────────────────

def hours_needed(energy, power):
    return int(max(1, min(24, np.ceil(energy / power))))

def cheapest_hours(prices, n):
    return sorted(sorted(range(len(prices)), key=lambda i: prices[i])[:n])

def uniform_hours(n):
    return sorted(set(int(round(i * 23 / max(1, n - 1))) for i in range(n)))

def cost(prices, chosen, power, hours):
    mean = np.mean([prices[i] for i in chosen])
    return mean * power * hours

def saved_metric(label, value, sub=""):
    """Custom metric card with an orange value — for the headline savings number."""
    st.markdown(f"""
    <div style="background:{PANEL}; border:1px solid {ACCENT};
                border-radius:12px; padding:16px 18px;">
      <div style="font-size:14px; color:{TEXTDIM};">{label}</div>
      <div style="font-size:30px; font-weight:700; color:{ACCENT}; margin-top:4px;">{value}</div>
      <div style="font-size:13px; color:{ACCENT}; opacity:.8; margin-top:2px;">{sub}</div>
    </div>
    """, unsafe_allow_html=True)

# ────────────────────────── sidebar inputs ───────────────────────────────────

with st.sidebar:
    st.markdown('<div class="duck-title">Duck<span>stradamus</span> 🦆⚡</div>',
                unsafe_allow_html=True)
    st.markdown("### Settings")
    energy = st.number_input("Daily energy demand (MWh)", value=2.0, min_value=0.1, step=0.1)
    power  = st.number_input("Max power capacity (MW)",   value=0.5, min_value=0.1, step=0.1)
    n_hours = hours_needed(energy, power)
    st.caption(f"→ {n_hours} charging hours/day "
               f"({power} MW × {n_hours} h = {power*n_hours:.1f} MWh)")
    page = st.radio("Page", ["Daily Charging", "2024 Savings"], index=0)

# ════════════════════════ PAGE 1: DAILY ══════════════════════════════════════

if page == "Daily Charging":
    st.markdown(f"## Plan tomorrow's charging")

    # Day selection — defaults to demo day; picker lives at the bottom of the page.
    all_dates = [d["date"] for d in DAYS_2024]
    default_idx = all_dates.index(DEMO_DAY["date"]) if DEMO_DAY["date"] in all_dates else 0
    chosen_date = st.session_state.get("chosen_date", DEMO_DAY["date"])
    if chosen_date not in all_dates:
        chosen_date = DEMO_DAY["date"]
    day = next(d for d in DAYS_2024 if d["date"] == chosen_date)

    st.caption(f"Forecast for {day['date']}")

    col_go, col_truth, _ = st.columns([1, 1, 4])
    go_clicked    = col_go.button("Go ▶", type="primary", use_container_width=True)
    show_truth    = col_truth.toggle("Show true prices")

    pred   = day["predicted"]
    actual = day["actual"]
    charge = cheapest_hours(pred, n_hours)

    chart_slot  = st.empty()
    status_slot = st.empty()

    # Fixed y-axis range so the chart never rescales during the animation
    y_max = max(max(pred), max(actual)) * 1.1

    def make_fig(n_points=24, highlight=None, with_truth=False):
        highlight = highlight or []
        fig = go.Figure()
        # predicted curve — always 24 categories, pad unrevealed hours with None
        y_pred_partial = pred[:n_points] + [None] * (24 - n_points)
        fig.add_trace(go.Scatter(
            x=HOURS, y=y_pred_partial,
            mode="lines", name="Predicted price",
            line=dict(color=ACCENT, width=3), fill="tozeroy",
            fillcolor="rgba(255,159,28,0.08)",
        ))
        if with_truth:
            fig.add_trace(go.Scatter(
                x=HOURS, y=actual, mode="lines", name="True price",
                line=dict(color=TEAL, width=2, dash="dash"),
            ))
        if highlight:
            fig.add_trace(go.Bar(
                x=[HOURS[i] for i in highlight],
                y=[pred[i] for i in highlight],
                name="Charge here", marker_color=ACCENT, opacity=0.85,
                width=0.5,
            ))
        fig.update_layout(
            template="plotly_dark", paper_bgcolor=PANEL, plot_bgcolor=PANEL,
            height=440, margin=dict(l=40, r=20, t=30, b=40),
            yaxis_title="NZD / MWh", legend=dict(orientation="h", y=1.12),
            font=dict(color=TEXTDIM),
            xaxis=dict(range=[-0.5, 23.5], type="category"),
            yaxis=dict(range=[0, y_max]),
        )
        return fig

    if go_clicked:
        # 1. animate prediction drawing in
        status_slot.info("Drawing predicted prices…")
        for k in range(1, 25):
            chart_slot.plotly_chart(make_fig(n_points=k, with_truth=show_truth),
                                    use_container_width=True, key=f"draw{k}")
            time.sleep(0.08)
        # 2. highlight charging hours one by one
        status_slot.info(f"Selecting the {n_hours} cheapest predicted hours…")
        time.sleep(0.3)
        for j in range(1, len(charge) + 1):
            chart_slot.plotly_chart(
                make_fig(highlight=charge[:j], with_truth=show_truth),
                use_container_width=True, key=f"hl{j}")
            time.sleep(0.2)
        st.session_state["day_done"] = True
        status_slot.success(
            "Recommended charging hours: " + ", ".join(HOURS[i] for i in charge))
    else:
        done = st.session_state.get("day_done", False)
        if done:
            # already animated this session — show full curve + charging hours
            chart_slot.plotly_chart(
                make_fig(highlight=charge, with_truth=show_truth),
                use_container_width=True, key="static")
        else:
            # fresh load — empty chart, prompt the user to press Go
            chart_slot.plotly_chart(
                make_fig(n_points=0, with_truth=False),
                use_container_width=True, key="static")
            status_slot.info("Press **Go ▶** to reveal the price forecast.")

    # stats (cost on TRUE price = what you'd actually pay) — always shown
    unif = uniform_hours(n_hours)
    smart_cost   = cost(actual, charge, power, n_hours)
    uniform_cost = cost(actual, unif,   power, n_hours)
    saved = uniform_cost - smart_cost
    pct = 100 * saved / uniform_cost if uniform_cost else 0

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Charging hours", f"{n_hours} h",
              f"{power*n_hours:.1f} MWh")
    c2.metric("Smart charging", f"${smart_cost:,.0f}")
    c3.metric("Charge anytime", f"${uniform_cost:,.0f}")
    with c4:
        saved_metric("You save", f"${saved:,.0f}", f"{pct:.1f}% cheaper")

    # ── Advanced (bottom of page) ─────────────────────────────────────────────
    st.markdown("---")
    with st.expander("Advanced", expanded=False):
        picked = st.selectbox("Forecast day", all_dates, index=all_dates.index(chosen_date))
        if picked != chosen_date:
            st.session_state["chosen_date"] = picked
            st.session_state["day_done"] = False
            st.rerun()

# ════════════════════════ PAGE 2: YEARLY ═════════════════════════════════════

else:
    st.markdown("## What you would have saved in 2024")
    st.caption(f"Smart charging vs flat spread · {n_hours} h/day "
               f"({energy} MWh @ {power} MW)")

    cum_model = cum_naive = cum_uniform = 0.0
    m_save, n_save = 0.0, 0.0
    labels, model_cum, naive_cum = [], [], []

    for d in DAYS_2024:
        smart  = cheapest_hours(d["predicted"], n_hours)
        naiveH = cheapest_hours(d["naive"],     n_hours)
        unif   = uniform_hours(n_hours)
        cM = cost(d["actual"], smart,  power, n_hours)
        cN = cost(d["actual"], naiveH, power, n_hours)
        cU = cost(d["actual"], unif,   power, n_hours)
        cum_model += cM; cum_naive += cN; cum_uniform += cU
        m_save += (cU - cM); n_save += (cU - cN)
        labels.append(d["date"]); model_cum.append(m_save); naive_cum.append(n_save)

    total_save = cum_uniform - cum_model
    pct = 100 * total_save / cum_uniform if cum_uniform else 0
    vs_naive = cum_naive - cum_model

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        saved_metric("Total saved vs flat", f"${total_save:,.0f}", f"{pct:.1f}% cheaper")
    c2.metric("Smart charging cost", f"${cum_model:,.0f}")
    c3.metric("Flat charging cost",  f"${cum_uniform:,.0f}")
    c4.metric("vs Naive model", f"${vs_naive:,.0f}",
              "extra saved" if vs_naive >= 0 else "naive better")

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=labels, y=model_cum, mode="lines",
        name="Cumulative smart saving", line=dict(color=ACCENT, width=3),
        fill="tozeroy", fillcolor="rgba(255,159,28,0.1)"))
    fig.add_trace(go.Scatter(x=labels, y=naive_cum, mode="lines",
        name="Cumulative naive saving", line=dict(color=NAIVE, width=2, dash="dash")))
    fig.update_layout(
        template="plotly_dark", paper_bgcolor=PANEL, plot_bgcolor=PANEL,
        height=460, margin=dict(l=50, r=20, t=30, b=40),
        yaxis_title="Cumulative saving (NZD)",
        legend=dict(orientation="h", y=1.1), font=dict(color=TEXTDIM),
    )
    st.plotly_chart(fig, use_container_width=True)
