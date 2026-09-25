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

  * 2026-09-25: the app used to OPEN on a BUY/HOLD signal generator -- the one thing a
    year of pre-registered research found does not work. It now opens on what was found,
    and the analyzer is the last tab, labelled as the engineering demo it is. The research
    tabs read ONE bundled file (research_snapshot.json, built by
    tools/build_research_snapshot.py) and never touch the live lab.

  * The analyzer used to refetch and refit on every submit. Identical requests are now
    cached for five minutes. Inside tabs, `st.stop()` would blank every tab after the one
    that called it, so the analyzer returns instead.
"""
import inspect
import json
import time
from pathlib import Path

import altair as alt
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

HERE = Path(__file__).resolve().parent
SNAPSHOT_JSON = HERE / "research_snapshot.json"

# Visitors who do not live in tickers know "Nvidia", not "NVDA". The table is BUNDLED
# (built offline by tools/build_symbol_table.py) rather than looked up live, so search
# works with no network and no vendor entitlement -- this app has enough failure modes
# that depend on a vendor already.
_SYMBOLS_JSON = HERE / "trade_analysis" / "symbols.json"
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
CLUSTER_COLORS = ["#d1495b", "#8d99ae", "#2e86ab"]     # sell-off, quiet, rally


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


@st.cache_data(show_spinner=False)
def load_snapshot() -> dict | None:
    try:
        return json.loads(SNAPSHOT_JSON.read_text(encoding="utf-8"))
    except Exception:                                        # noqa: BLE001
        return None


class ApiError(Exception):
    def __init__(self, status: int, body: str):
        super().__init__(f"HTTP {status}")
        self.status, self.body = status, body


@st.cache_data(ttl=300, show_spinner=False)
def analyze(symbol: str, timeframe: str, strategy: str) -> dict:
    """One analysis per (symbol, timeframe, strategy) per five minutes.

    Errors raise instead of returning, so a failed request is never cached and the next
    click retries it.
    """
    r = requests.post(f"{API_URL}/predict/enhanced/", params={
        "symbol": symbol, "timeframe": timeframe, "strategy_mode": strategy}, timeout=180)
    if r.status_code != 200:
        raise ApiError(r.status_code, r.text)
    return r.json()


def _money(x: float) -> str:
    return f"{'−' if x < 0 else '+'}${abs(x):,.2f}"


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

st.title("📈 ProfitBook")
st.caption("A year of 0DTE options research on QQQ and SPY: what was tested, what failed, "
           "and what is being forward-tested now. A research demo — not investment advice, "
           "and nothing here places an order.")

SNAP = load_snapshot()


# =========================================================================== tab 1
def render_found():
    if not SNAP:
        st.warning("research_snapshot.json is missing from this build.")
        return
    st.markdown("#### The short version")
    st.markdown(
        "The question was whether a discretionary 0DTE options strategy could be "
        "automated profitably. Every test was **pre-registered** (the rules were written "
        "down before the data was looked at), so a result could not be tuned into "
        "existence. Most answers were **no** — and each *no* closed a door that would "
        "otherwise have absorbed months.")
    cols = st.columns(len(SNAP["headlines"][:3]))
    for c, h in zip(cols, SNAP["headlines"][:3]):
        with c.container(border=True):
            st.markdown(f"**{h['title']}**")
            st.write(h["plain"])
            st.markdown(f"**{h['number']}**")
            st.caption(h["source"])
    cols = st.columns(len(SNAP["headlines"][3:]) or 1)
    for c, h in zip(cols, SNAP["headlines"][3:]):
        with c.container(border=True):
            st.markdown(f"**{h['title']}**")
            st.write(h["plain"])
            st.markdown(f"**{h['number']}**")
            st.caption(h["source"])

    st.markdown("#### How it is organised")
    a, b, c = st.columns(3)
    a.markdown("**HPC** — settles what old data can settle, in hours. The straddle cost "
               "model and the volatility forecasts ran on Northeastern's cluster.")
    b.markdown("**Live lab** — a frozen, pre-registered paper-trading test that records "
               "every decision *before* it can see a price. See *Live forward test*.")
    c.markdown("**This demo** — shows the work to someone in minutes, and shares no code "
               "with the live lab so nothing here can disturb it.")

    dc = SNAP.get("day_clusters")
    if dc:
        st.markdown("#### What kind of QQQ day was it?")
        st.caption("An unsupervised model (k-means) sorting 2,631 sessions by the shape of "
                   "the day, with the trader's own journal laid over it. " + dc["caveat"])
        names = {s["cluster"]: s["name"] for s in dc["summary"]}
        pts = pd.DataFrame(dc["points"])
        pts["type"] = pts["cluster"].map(names)
        pts = pts[(pts["oc"].abs() <= 4) & (pts["range"] > 0)]
        order = [names[i] for i in sorted(names)]
        scale = alt.Scale(domain=order, range=CLUSTER_COLORS[: len(order)])
        dots = alt.Chart(pts).mark_circle(size=14, opacity=0.35).encode(
            x=alt.X("oc:Q", title="QQQ open-to-close return (%)"),
            y=alt.Y("range:Q", title="High–low range (% of open)",
                    scale=alt.Scale(type="log")),
            color=alt.Color("type:N", scale=scale, title="Day type"),
            tooltip=["day", "type", alt.Tooltip("oc:Q", format="+.2f"),
                     alt.Tooltip("range:Q", format=".2f")])
        cent = pts.groupby("type", as_index=False).agg(
            oc=("oc", "mean"), range=("range", "median"), n=("day", "count"))
        bubbles = alt.Chart(cent).mark_circle(opacity=0.85, stroke="white",
                                              strokeWidth=1.5).encode(
            x="oc:Q", y="range:Q",
            size=alt.Size("n:Q", scale=alt.Scale(range=[600, 3000]), legend=None),
            color=alt.Color("type:N", scale=scale, legend=None),
            tooltip=["type", alt.Tooltip("n:Q", title="sessions")])
        st.altair_chart((dots + bubbles).properties(height=380), use_container_width=True)
        tab = pd.DataFrame([{"day type": s["name"], "what it looks like": s["looks_like"],
                             "sessions": s["sessions"], "days he traded": s["traded_days"],
                             "his total": _money(s["total"]),
                             "his median day": _money(s["median_per_day"]),
                             "95% CI of mean day": f"{s['ci'][0]:+.0f} to {s['ci'][1]:+.0f}"}
                            for s in dc["summary"]])
        st.dataframe(tab, hide_index=True, use_container_width=True)
        s = dc["stability"]
        st.caption(f"Stable, but not tradeable: the same {dc['k']} types reappear across "
                   f"seeds (ARI ≥ {s['seeds_ari_min']}) and when fitted on 2016–20 and "
                   f"applied to 2021–26 (ARI {s['fit_2016_20_apply_2021_26_ari']}). They "
                   f"mostly sort days by how they closed, which is only known at 16:00.")


# =========================================================================== tab 2
def render_live():
    if not SNAP:
        st.warning("research_snapshot.json is missing from this build.")
        return
    ft = SNAP["forward_test"]
    st.markdown("#### A paper-trading lab that cannot cheat")
    st.write("Thirteen setups (now fifteen) run live every weekday on QQQ and SPY 0DTE "
             "options and on 15 stocks and ETFs. Each decision is written to disk *before* "
             "any price is requested, the rules are frozen by hash, and the trade counter "
             "never resets. It never places an order.")
    m = st.columns(4)
    m[0].metric("Running since", ft["start"])
    m[1].metric("Option sessions", ft["sessions_options"])
    m[2].metric("Option trades (ATM)", sum(r["trades"] for r in ft["options_atm"]))
    m[3].metric("Share trades", sum(r["trades"] for r in ft["shares"]))
    st.warning("**Not evidence yet.** The pre-registered bar is 200 trades per setup. "
               + ft["note"], icon="⚠️")

    daily = pd.DataFrame(ft["daily"])
    if not daily.empty:
        long = daily.melt("day", var_name="arm", value_name="net")
        st.altair_chart(alt.Chart(long).mark_bar().encode(
            x=alt.X("day:N", title=None), y=alt.Y("net:Q", title="net $ per session"),
            color=alt.Color("arm:N", scale=alt.Scale(range=["#2e86ab", "#f18f01"])),
            xOffset="arm:N", tooltip=["day", "arm", alt.Tooltip("net:Q", format="+,.2f")]
        ).properties(height=260), use_container_width=True)

    a, b = st.columns(2)
    with a:
        st.markdown("**Options, at-the-money contract**")
        st.dataframe(pd.DataFrame(ft["options_atm"]), hide_index=True,
                     use_container_width=True)
    with b:
        st.markdown("**Shares, $10,000 per trade**")
        st.dataframe(pd.DataFrame(ft["shares"]), hide_index=True, use_container_width=True)
    st.caption("INSUFFICIENT = below the pre-registered bar: 200 trades per setup on options; "
               "200 trades AND 60 sessions on shares, because 15 correlated names reach 200 "
               "raw trades in weeks without adding 15x the information.")

    bf = SNAP.get("backfill")
    if bf:
        st.markdown("#### The trader's own six lines, replayed over the sessions already run")
        st.info("**Backfill, not prospective.** Replayed on "
                f"{len(bf['days'])} past sessions through the real runner code with "
                "historical 1-minute bars and quotes, config "
                f"`{bf['config_hash']}`. Kept apart from the record above and never counted "
                "toward it. Differences from live: " + "; ".join(bf["differences_from_live"])
                + ".", icon="🔁")
        days = pd.DataFrame(bf["days"])
        days["note"] = days["live_n"].map(
            lambda n: "" if n else "live options blocked (entitlement lapsed)")
        show = days.rename(columns={
            "day": "session", "live_net": "live record $", "back13_net": "replay, 13 old $",
            "six_net": "Six_Lines $", "sixnc_net": "Six_Lines_NoCap $",
            "newcfg_net": "replay, new config $"})[
            ["session", "live record $", "replay, 13 old $", "Six_Lines $",
             "Six_Lines_NoCap $", "replay, new config $", "note"]]
        st.dataframe(show, hide_index=True, use_container_width=True)

        # Compare like with like: only the sessions the live options arm actually traded.
        # The replay also runs 2026-09-08..11, when the live arm was blocked, and summing
        # those into one side only would compare two different sets of days.
        same = [d for d in bf["days"] if d["live_n"]]
        blocked = [d for d in bf["days"] if not d["live_n"]]
        tot = {k: sum(d[k] for d in same) for k in
               ("live_net", "back13_net", "six_net", "sixnc_net", "newcfg_net")}
        flips = sum(1 for d in same if (d["live_net"] > 0) != (d["newcfg_net"] > 0))
        st.markdown(f"**On the {len(same)} sessions the live options arm traded**")
        c = st.columns(5)
        c[0].metric("Live record", _money(tot["live_net"]))
        c[1].metric("Replay, same 13 setups", _money(tot["back13_net"]),
                    delta=f"{100 * (tot['back13_net'] / tot['live_net'] - 1):+.1f}% vs live"
                    if tot["live_net"] else None, delta_color="off")
        c[2].metric("Six_Lines (his spec)", _money(tot["six_net"]))
        c[3].metric("Six_Lines_NoCap", _money(tot["sixnc_net"]))
        c[4].metric("Sessions that flip sign", f"{flips} of {len(same)}")
        if blocked:
            st.caption(f"The replay also covers {len(blocked)} sessions the live options arm "
                       f"could not trade ({blocked[0]['day']} → {blocked[-1]['day']}); there "
                       f"the new config would have made "
                       f"{_money(sum(d['newcfg_net'] for d in blocked))}. "
                       "The replay of the same 13 setups landing within a few percent of the "
                       "live record is the check that the replay is faithful.")
        if bf.get("six_trades"):
            with st.expander("Every six-line trade in the replay"):
                st.dataframe(pd.DataFrame(bf["six_trades"]), hide_index=True,
                             use_container_width=True)


# =========================================================================== tab 3
def render_fill():
    if not SNAP:
        st.warning("research_snapshot.json is missing from this build.")
        return
    fe = SNAP["fill_explorer"]
    st.markdown("#### Where you get filled decides whether there is an edge at all")
    st.write(f"{fe['n_trades']:,} real short at-the-money 0DTE SPY straddles over "
             f"{fe['n_sessions']} sessions ({fe['span'][0]} → {fe['span'][1]}), each "
             "held 30 minutes and priced from the actual bid and ask. Move the slider from "
             "the **mid price** most backtests assume to the **bid/ask** you actually get.")
    f = st.slider("Fill position", 0.0, 1.0, 0.0, 0.05,
                  help="0 = filled at the mid both ways. 1 = pay the full spread both ways.",
                  format="%.2f")
    g = min(fe["grid"], key=lambda r: abs(r["f"] - f))
    c = st.columns(3)
    c[0].metric("Mean P&L per straddle",
                f"{'+' if g['mean'] >= 0 else '−'}${abs(g['mean']):.4f}")
    c[1].metric("Session-clustered t", f"{g['t']:+.2f}",
                delta="significant" if abs(g["t"]) > 2 else "noise",
                delta_color="normal" if g["t"] > 2 else
                ("inverse" if g["t"] < -2 else "off"))
    c[2].metric("Win rate", f"{g['win'] * 100:.1f}%")
    grid = pd.DataFrame(fe["grid"])
    line = alt.Chart(grid).mark_line(point=True).encode(
        x=alt.X("f:Q", title="fill position (0 = mid, 1 = bid/ask)"),
        y=alt.Y("t:Q", title="session-clustered t"),
        tooltip=[alt.Tooltip("f:Q", format=".2f"), alt.Tooltip("t:Q", format="+.2f"),
                 alt.Tooltip("mean:Q", format="+.4f")])
    zero = alt.Chart(pd.DataFrame({"t": [2, -2]})).mark_rule(strokeDash=[4, 4],
                                                               color="#999").encode(y="t:Q")
    here = alt.Chart(pd.DataFrame({"f": [g["f"]]})).mark_rule(color="#d1495b").encode(x="f:Q")
    st.altair_chart((line + zero + here).properties(height=300), use_container_width=True)
    sig_until = max((r["f"] for r in fe["grid"] if r["t"] > 2), default=None)
    flips_at = min((r["f"] for r in fe["grid"] if r["mean"] < 0), default=None)
    if sig_until is not None and flips_at is not None:
        st.markdown(f"The edge stops being distinguishable from noise once you pay about "
                    f"**{sig_until:.0%} of the spread**, and the average trade loses money "
                    f"from about **{flips_at:.0%}**. Real fills are at 100%.")
    st.caption("The win rate stays above 60% the whole way — the trap. The worst 1% of "
               "trades carry 94% of the loss, so this would look like it was working for "
               "months. Source: " + fe["source"] + ".")


# =========================================================================== tab 4
def render_bugs():
    if not SNAP:
        st.warning("research_snapshot.json is missing from this build.")
        return
    st.markdown("#### The one bug class")
    st.write("Every serious defect here had the same shape: **no crash, no error, a "
             "confident-looking number that measured nothing.** Each was found by asking "
             "what would make a number wrong, and testing that.")
    st.dataframe(pd.DataFrame(SNAP["silent_bugs"]).rename(columns={
        "where": "where", "defect": "the defect", "cost": "what it cost"}),
        hide_index=True, use_container_width=True)
    st.markdown("#### The 398,854-parameter model that collapsed")
    tr = SNAP["tft_regimes"]
    st.write("Fed three deliberately opposite markets, the Temporal Fusion Transformer's "
             "gap probability barely moved. It was removed from the decision and replaced "
             "by a six-parameter HAR volatility model that does respond to its input.")
    st.dataframe(pd.DataFrame(tr["rows"], index=tr["columns"]).T, use_container_width=True)
    st.caption("Source: " + tr["source"])


# =========================================================================== tab 5
def render_analyzer():
    st.caption("The original engineering demo: multi-timeframe momentum, sentiment and an "
               "options chain combined into a verdict. **It calls direction, which the "
               "research above found no edge in** — it is here to show the plumbing, the "
               "permutation-tested strategy modes and the honest volatility panel.")
    problem = api_startup_problem()
    if problem:
        st.error("The analysis backend is not running, so no signal can be produced.")
        with st.expander("Why", expanded=True):
            st.code(problem)

    # Session banner, rendered from the exchange calendar BEFORE any request, so it still
    # appears when the backend is down. On 2026-09-07 (Labor Day) the app served Friday's
    # tape with nothing indicating it, which reads as a broken feed rather than a shut
    # market.
    try:
        from trade_analysis.market_session import banner as _banner, market_status
        _ms = market_status()
        if _ms.get("is_open"):
            st.success(_banner(_ms), icon="📈")
        elif _ms.get("state") == "unknown":
            st.info(_banner(_ms))
        else:
            st.warning(_banner(_ms), icon="🕒")
    except Exception as _e:                                  # noqa: BLE001
        st.caption(f"Session status unavailable: {_e}")

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
        submitted = c4.form_submit_button("Analyze", use_container_width=True,
                                          type="primary")

        if SYMBOL_OPTIONS and not _FREE_TEXT_OK:
            other = st.text_input(
                "…or any other ticker", "",
                placeholder="e.g. ASML — overrides the choice above when filled")
            if other.strip():
                symbol = other.strip().upper()

    if not submitted:
        return
    if not symbol:
        st.warning("Choose or type a symbol first.")
        return
    t0 = time.time()
    with st.spinner(f"Analysing {symbol} …"):
        try:
            data = analyze(symbol, timeframe, strategy)
        except requests.exceptions.ConnectionError:
            problem = api_startup_problem()
            if problem:
                st.error("The analysis backend is not running.")
                st.code(problem)
            else:
                st.error("Could not reach the analysis API on port 7860. It answered at "
                         "startup, so this is likely transient — try again.")
            return
        except requests.exceptions.Timeout:
            st.error("The analysis took longer than 180s and was abandoned.")
            return
        except ApiError as exc:
            st.error(f"API error {exc.status} for {symbol}. "
                     "If this symbol is unusual, the data provider may not carry it.")
            try:
                st.json(json.loads(exc.body))
            except Exception:                                # noqa: BLE001
                st.text(exc.body)
            return
    took = time.time() - t0
    st.caption(f"answered in {took:.1f}s" + (" — from the 5-minute cache" if took < 0.5
                                              else ""))

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

    alt_ = (data.get("details", {}) or {}).get("alternative_data", {}) or {}
    sent = (data.get("details", {}) or {}).get("sentiment", {}) or {}
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("VIX", alt_.get("vix_level", "—"))
    m2.metric("Implied vol", f"{alt_.get('implied_vol_pct', 0):.0f}%"
              if alt_.get("implied_vol_pct") else "—")
    m3.metric("Put/call", alt_.get("put_call_ratio", "—"))
    m4.metric("Sentiment", f"{sent.get('composite_score', 0):+.2f}",
              delta=f"{sent.get('confidence', '—')} confidence", delta_color="off")

    # HAR volatility forecast vs what the option market charges. This is the one model
    # in the app that is both small and measurably working, so it gets a row of its own
    # rather than an expander -- and the honest caveat travels with it.
    vol = data.get("volatility") or {}
    if vol.get("available"):
        st.markdown("**Volatility: forecast vs the option market**")
        v1, v2, v3 = st.columns(3)
        v1.metric("HAR forecast (next session)", f"{vol['forecast_vol_pct']:.1f}%",
                  delta=f"last realised {vol['last_realised_vol_pct']:.1f}%",
                  delta_color="off")
        if vol.get("implied_vol_pct"):
            v2.metric("Implied (option chain)", f"{vol['implied_vol_pct']:.1f}%")
            v3.metric("Variance risk premium", f"{vol['premium_vol_pts']:+.1f} pts",
                      delta="options dearer" if vol['premium_vol_pts'] > 0
                            else "options cheaper", delta_color="off")
        else:
            v2.metric("Implied (option chain)", "—")
            v3.metric("Variance risk premium", "—")
        # Where the implied vol came from. It matters: the free chain has no bid/ask, so
        # this IV is inverted from a TRADE price and nothing here can see the spread --
        # which is the quantity that killed the premium in the 0DTE study.
        _src = (alt_.get('data_quality', {}) or {}).get('iv_source')
        if _src:
            st.caption(f'Implied vol: {_src}.')
        st.caption(
            f"HAR-RV fitted on {vol['n_days_fitted']} of this symbol's own daily bars "
            f"(Garman-Klass variance incl. the overnight gap, lognormal-corrected "
            f"x{vol.get('lognormal_correction', 1):.2f}). In-sample QLIKE "
            f"{vol['qlike_har']:.3f} vs {vol['qlike_naive']:.3f} for naive persistence "
            f"-- {'beats' if vol['beats_naive'] else 'does NOT beat'} it.  "
            "**A positive premium is not a trade:** measured over 618 sessions of SPY "
            "0DTE data it is real (t=+7.5 gross) and 1.84x too small to survive the "
            "straddle spread, losing at t=-5.4 net. See the *Fill-price explorer* tab.")
    elif vol.get("reason"):
        st.caption(f"Volatility forecast unavailable: {vol['reason']}")

    with st.expander("Headlines the sentiment model read"):
        themes = sent.get("key_themes") or []
        if themes:
            st.dataframe(pd.DataFrame(themes), use_container_width=True)
        else:
            st.caption("No headlines available for this symbol.")

    with st.expander("Gap model — the 398,854-parameter TFT that collapsed (does not vote)"):
        st.caption(
            "Measured and found degenerate: fed six different symbols it moved "
            "gap_probability by 0.10 on a 0–100 scale, a violent crash and a flat day "
            "differ by 0.1 points, and out of sample it emitted one constant direction per "
            "symbol at the majority-class hit rate. Shown so the failure is visible, not "
            "hidden — see the *Silent-bug gallery* tab.")
        st.json((data.get("details", {}) or {}).get("tft_prediction") or {})

    with st.expander("Raw response"):
        st.json(data)


tabs = st.tabs(["What I found", "Live forward test", "Fill-price explorer",
                "Silent-bug gallery", "Signal analyzer (demo)"])
with tabs[0]:
    render_found()
with tabs[1]:
    render_live()
with tabs[2]:
    render_fill()
with tabs[3]:
    render_bugs()
with tabs[4]:
    render_analyzer()
