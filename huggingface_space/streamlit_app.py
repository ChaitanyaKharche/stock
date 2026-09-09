"""ProfitBook UI.

Design notes, because several of these were bug reports rather than taste:

  * "Position Size 0.00" was a headline metric on every HOLD. It is a FRACTION OF
    CAPITAL, not a share count, and the same name carries three different values at three
    nesting levels of the response. A unitless "0.00" duplicating what
    options_strategy.contracts already says is noise. Shown only when there IS a
    position, labelled.

  * The result gave a verdict with no reason. "HOLD, 17%" cannot distinguish "nothing is
    happening" from "the timeframes disagree" from "confidence missed the gate by two".
    The API returns verdict.blocking_reason and the gates it measured against.

  * Icons use a LIGATURE font: the element contains the text "keyboard_arrow_right" and
    the font substitutes a glyph. A blanket font-family override broke the substitution
    and printed the glyph names over every expander label. Icons keep their own family.

  * Symbol entry assumed the visitor thinks in tickers. The bundled name table is loaded
    from disk, so searching "nvidia" works with no network, no key and no entitlement.
"""
import inspect
import json
from pathlib import Path

import pandas as pd
import requests
import streamlit as st

# The URL where the FastAPI backend runs inside the Space. Port 7860 is container-internal;
# only 8501 (this UI) is exposed, which is why the two talk over localhost.
API_URL = "http://localhost:7860"

# run.sh writes the API's startup outcome here before launching Streamlit. Without it, a
# backend that died at import -- e.g. config.py raising because a Space secret is unset --
# is indistinguishable from a transient blip, and every button just says
# "Failed to connect". The status file turns that into an actual diagnosis.
API_STATUS_FILE = "/tmp/api_status"

# Visitors who do not live in tickers know "Nvidia", not "NVDA". The table is BUNDLED
# (built offline by tools/build_symbol_table.py) rather than looked up live, so search
# works with no network and no vendor entitlement -- this app has enough failure modes
# that depend on a vendor already.
_SYMBOLS_JSON = Path(__file__).resolve().parent / "trade_analysis" / "symbols.json"
try:
    SYMBOL_NAMES: dict[str, str] = json.loads(_SYMBOLS_JSON.read_text(encoding="utf-8"))
except Exception:                                            # noqa: BLE001
    SYMBOL_NAMES = {}

# "NVDA - NVIDIA Corporation", so Streamlit's own option filter matches either half.
SYMBOL_OPTIONS = [f"{k} - {v}" for k, v in SYMBOL_NAMES.items()]
_DEFAULT = next((o for o in SYMBOL_OPTIONS if o.startswith("NVDA ")),
                SYMBOL_OPTIONS[0] if SYMBOL_OPTIONS else "NVDA")
# Older Streamlit builds have no free-text selectbox. Probing the signature is safer than
# a try/except around a widget that may already have half-rendered.
_FREE_TEXT_OK = "accept_new_options" in inspect.signature(st.selectbox).parameters

TIMEFRAMES = ["15m", "1h", "4h", "1d"]
STRATEGIES = ["momentum", "gap", "reversal"]
STRATEGY_HELP = {
    "momentum": "Follows an established move.",
    "gap": "Trades a measured overnight gap; stands down when there is not one.",
    "reversal": "Fades an extended move. Often disagrees with momentum on the same bars.",
}
SIGNAL_STYLE = {"CALLS": "🟢", "PUTS": "🔴", "HOLD": "⚪"}


def parse_symbol(choice: str) -> str:
    """'NVDA - NVIDIA Corporation' -> 'NVDA'. A raw ticker passes straight through."""
    return (choice or "").split(" - ")[0].strip().upper()


def api_startup_problem() -> str | None:
    """The reason the backend is not answering, if run.sh recorded one."""
    try:
        with open(API_STATUS_FILE, encoding="utf-8") as fh:
            text = fh.read().strip()
    except OSError:
        return None
    return None if text in ("", "ready") else text


def timeframe_table(data: dict) -> pd.DataFrame | None:
    """One row per bar set: what each timeframe actually contributed."""
    setups = (data.get("momentum_analysis", {})
                  .get("momentum_analysis", {})
                  .get("timeframe_setups", {}))
    tech = (data.get("details", {}) or {}).get("tech_setups", {}) or {}
    if not setups:
        return None
    rows = []
    for tf, s in setups.items():
        t = tech.get(tf, {}) or {}
        ds = s.get("direction_score")
        rows.append({
            "bars": tf,
            "direction": ("bullish" if (ds or 0) > 0 else
                          "bearish" if (ds or 0) < 0 else "flat"),
            "dir score": None if ds is None else round(ds, 3),
            "momentum": round(s.get("momentum_score", 0), 3),
            "RSI": t.get("rsi"),
            "ADX": t.get("adx"),
            "trend": t.get("trend"),
            "vol regime": (s.get("volatility", {}) or {}).get("volatility_regime"),
        })
    return pd.DataFrame(rows).set_index("bars")


st.set_page_config(page_title="ProfitBook", page_icon="📈", layout="wide")

# Montserrat, and reclaim the dead band above the title. Streamlit reserves ~6rem at the
# top of .block-container for a toolbar this app never uses, so the header sat a third of
# the way down the fold. The font is applied by selector rather than by theme config
# because [theme] font only accepts sans serif / serif / monospace.
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@300;400;500;600;700&display=swap');

    html, body, .stApp, [class*='css'],
    h1, h2, h3, h4, h5, h6, p, li, label, span, div,
    button, input, select, textarea,
    [data-testid='stMetricValue'], [data-testid='stMetricLabel'],
    [data-testid='stMetricDelta'], [data-testid='stMarkdownContainer'] {
        font-family: 'Montserrat', -apple-system, sans-serif !important;
    }

    /* Streamlit draws expander arrows and alert icons with a LIGATURE font: the element
       literally contains the text "keyboard_arrow_right" and the font substitutes a
       glyph. Forcing Montserrat on span/div broke the substitution, so the raw name
       rendered on top of every expander label. Icons must keep their own family. */
    [data-testid='stIconMaterial'],
    [data-testid='stAlertDynamicIcon'],
    [data-testid='stTooltipIcon'],
    .material-icons, [class*='material-symbols'], [class*='MaterialIcon'] {
        font-family: 'Material Symbols Rounded', 'Material Icons Rounded',
                     'Material Icons' !important;
    }

    /* the gap above the title */
    .block-container, .main .block-container,
    [data-testid='stAppViewContainer'] > .main > div > .block-container {
        padding-top: 1.2rem !important;
    }
    [data-testid='stHeader'] { height: 0rem; background: transparent; }
    [data-testid='stToolbar'] { right: 0.5rem; }

    h1 { font-weight: 700 !important; letter-spacing: -0.02em; margin-bottom: 0.2rem; }
    h2, h3 { font-weight: 600 !important; letter-spacing: -0.01em; }
    [data-testid='stMetricValue'] { font-weight: 600 !important; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("📈 ProfitBook Trading Analysis Engine")
st.caption("Multi-timeframe momentum, sentiment and options analysis. "
           "Research tool — not investment advice.")

_problem = api_startup_problem()
if _problem:
    st.error("The analysis backend is not running, so no signal can be produced.")
    with st.expander("Why", expanded=True):
        st.code(_problem)

# Session banner, rendered from the exchange calendar BEFORE any request, so it still
# appears when the backend is down. On 2026-09-07 (Labor Day) the app served Friday's
# tape with nothing indicating it, which reads as a broken feed rather than a shut market.
try:
    from trade_analysis.market_session import banner as _banner, market_status
    _ms = market_status()
    if _ms.get("is_open"):
        st.success(_banner(_ms), icon="📈")
    elif _ms.get("state") == "unknown":
        st.info(_banner(_ms))
    else:
        st.warning(_banner(_ms), icon="🕒")
except Exception as _e:                                      # noqa: BLE001
    st.caption(f"Session status unavailable: {_e}")

st.divider()

with st.form("signal"):
    c1, c2, c3, c4 = st.columns([2.6, 1, 1.4, 1])
    if SYMBOL_OPTIONS:
        kw = {"accept_new_options": True} if _FREE_TEXT_OK else {}
        choice = c1.selectbox(
            "Symbol or company", SYMBOL_OPTIONS,
            index=SYMBOL_OPTIONS.index(_DEFAULT) if _DEFAULT in SYMBOL_OPTIONS else 0,
            help=("Type a ticker or a company name — 'nvidia' finds NVDA."
                  + ("" if _FREE_TEXT_OK
                     else " For a symbol not listed, use the box below.")),
            **kw)
        symbol = parse_symbol(choice)
    else:
        symbol = c1.text_input("Symbol", "NVDA").strip().upper()

    timeframe = c2.selectbox(
        "Timeframe", TIMEFRAMES, index=0,
        help="Which bars dominate the score. All four are always read; "
             "this shifts the emphasis.")
    strategy = c3.selectbox(
        "Strategy", STRATEGIES, index=0,
        help="  \n\n".join(f"**{k}** — {v}" for k, v in STRATEGY_HELP.items()))
    c4.markdown("<div style='height:1.85rem'></div>", unsafe_allow_html=True)
    submitted = c4.form_submit_button("Analyze", use_container_width=True, type="primary")

    if SYMBOL_OPTIONS and not _FREE_TEXT_OK:
        other = st.text_input(
            "…or any other ticker", "",
            placeholder="e.g. ASML — overrides the choice above when filled")
        if other.strip():
            symbol = other.strip().upper()

if submitted:
    if not symbol:
        st.warning("Choose or type a symbol first.")
        st.stop()
    with st.spinner(f"Analysing {symbol} …"):
        try:
            response = requests.post(f"{API_URL}/predict/enhanced/", params={
                "symbol": symbol, "timeframe": timeframe, "strategy_mode": strategy},
                timeout=180)
        except requests.exceptions.ConnectionError:
            problem = api_startup_problem()
            if problem:
                st.error("The analysis backend is not running.")
                st.code(problem)
            else:
                st.error("Could not reach the analysis API on port 7860. It answered at "
                         "startup, so this is likely transient — try again.")
            st.stop()
        except requests.exceptions.Timeout:
            st.error("The analysis took longer than 180s and was abandoned.")
            st.stop()

    if response.status_code != 200:
        st.error(f"API error {response.status_code} for {symbol}. "
                 "If this symbol is unusual, the data provider may not carry it.")
        try:
            st.json(response.json())
        except Exception:                                    # noqa: BLE001
            st.text(response.text)
        st.stop()

    data = response.json()
    sig = data.get("signal", "HOLD")
    verdict = data.get("verdict") or {}
    market = data.get("market") or {}
    quote = data.get("quote") or {}
    icon = SIGNAL_STYLE.get(sig, "⚪")

    sym_out = data.get("symbol", symbol)
    company = SYMBOL_NAMES.get(sym_out, "")
    st.subheader(f"{icon}  {sym_out} — {sig}" + (f"   ·   {company}" if company else ""))

    if quote.get("price"):
        chg = quote.get("change_pct")
        arrow = "" if chg is None else ("▲" if chg > 0 else "▼" if chg < 0 else "▬")
        rng = (f"   ·   day {quote['day_low']:,.2f}–{quote['day_high']:,.2f}"
               if quote.get("day_low") and quote.get("day_high") else "")
        tail = (f"  <span style='font-size:0.6em;opacity:.75'>{arrow} {chg:+.2f}%{rng}"
                "</span>") if chg is not None else ""
        st.markdown(f"### {quote['price']:,.2f}{tail}", unsafe_allow_html=True)

    bits = []
    if market.get("reference_session"):
        bits.append(f"session {market['reference_session']}")
    if market.get("latest_bar"):
        bits.append(f"newest bar {str(market['latest_bar'])[:16]}")
    if not market.get("is_open", True):
        bits.append("market closed")
    if bits:
        st.caption("  ·  ".join(bits))
    if market.get("warning"):
        st.warning(market["warning"], icon="⚠️")

    # Tiles adapt to the verdict. A "Position Size 0.00" on a HOLD is not information;
    # the margin to the gate that produced the HOLD is.
    conf = data.get("confidence", 0)
    gate = verdict.get("min_confidence")
    t1, t2, t3 = st.columns(3)
    if sig == "HOLD":
        t1.metric("Confidence", f"{conf:.0f}%",
                  delta=None if gate is None else f"{conf - gate:+.0f} vs {gate} gate",
                  delta_color="off")
        score, thr = verdict.get("weighted_score"), verdict.get("score_threshold")
        t2.metric("Score", "—" if score is None else f"{score:+.3f}",
                  delta=None if thr is None else f"needs ±{thr:.2f}", delta_color="off")
        agree = verdict.get("direction_agreement")
        t3.metric("Timeframe agreement",
                  "—" if agree is None else f"{agree * 100:.0f}%",
                  delta=(verdict.get("direction") or "").title() or None,
                  delta_color="off")
    else:
        t1.metric("Confidence", f"{conf:.0f}%",
                  delta=None if gate is None else f"{conf - gate:+.0f} vs {gate} gate")
        # position_size is a FRACTION OF CAPITAL. Shown only when there is a position,
        # and labelled, because "0.00" with no unit was the original complaint.
        t2.metric("Allocation", f"{data.get('position_size', 0) * 100:.0f}% of capital")
        t3.metric("Hold", data.get("expected_hold_time", "—"))

    if verdict.get("blocking_reason"):
        st.info(verdict["blocking_reason"], icon="💡")

    notes = [p.strip() for p in str(data.get("reasoning", "")).split(". ")
             if p.strip().startswith(("Reversal", "Gap", "Overnight"))]
    if notes:
        st.caption("  ·  ".join(notes))

    df = timeframe_table(data)
    if df is not None:
        st.markdown("**What each timeframe contributed**")
        st.dataframe(df, use_container_width=True)
        sel = {"15m": "15m", "1h": "hourly", "4h": "4h", "1d": "daily"}.get(timeframe)
        if sel:
            st.caption(f"Your **{timeframe}** selection weights these bars most heavily "
                       f"toward **{sel}**; all four are always read.")

    opts = data.get("options_strategy") or {}
    if opts.get("contracts"):
        st.markdown("**Suggested options structure**")
        st.dataframe(pd.DataFrame(opts["contracts"]), use_container_width=True)
        if opts.get("risk_management"):
            st.caption(f"Risk management: {opts['risk_management']}")
    elif opts.get("reasoning"):
        st.caption(f"Options: {opts.get('strategy', '—')} — {opts['reasoning']}")

    alt = (data.get("details", {}) or {}).get("alternative_data", {}) or {}
    sent = (data.get("details", {}) or {}).get("sentiment", {}) or {}
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("VIX", alt.get("vix_level", "—"))
    m2.metric("Implied vol", f"{alt.get('implied_vol_pct', 0):.0f}%"
              if alt.get("implied_vol_pct") else "—")
    m3.metric("Put/call", alt.get("put_call_ratio", "—"))
    m4.metric("Sentiment", f"{sent.get('composite_score', 0):+.2f}",
              delta=f"{sent.get('confidence', '—')} confidence", delta_color="off")

    with st.expander("Headlines the sentiment model read"):
        themes = sent.get("key_themes") or []
        if themes:
            st.dataframe(pd.DataFrame(themes), use_container_width=True)
        else:
            st.caption("No headlines available for this symbol.")

    with st.expander("Gap model (diagnostic only — does not vote)"):
        st.caption(
            "This checkpoint was measured and found degenerate: fed six different "
            "symbols it moved gap_probability by 0.10 on a 0–100 scale, and out of "
            "sample it emitted one constant direction per symbol at the majority-class "
            "hit rate. Reported for transparency, removed from the decision.")
        st.json((data.get("details", {}) or {}).get("tft_prediction") or {})

    with st.expander("Raw response"):
        st.json(data)
