"""Tests for the OPTIONS arm's staleness guards (runner.py): bar, underlying, option.

What can lie here, in order of danger:

  1. A STALE SIGNAL OPENING A POSITION. On a 09:35 start the feed admits 09:30-09:34 in
     one batch and every bar is evaluated against the CURRENT chain. Crabel_Stretch took
     its 09:31 signal at 09:36:41 this way -- an honest fill on a dishonest signal, and a
     trade the shares arm refused on the same bar. The sharpest test is that exact case.
  2. THE 5m BUCKET MEASURED FROM THE WRONG STAMP. 5m buckets are stamped at their CLOSE,
     1m bars at their OPEN. Measuring a bucket's age from its close stamp makes this arm
     60s more lenient than the shares arm, so the two records would again disagree.
  3. A SUPPRESSED SIGNAL VANISHING. A skip that is not written makes the signal count
     depend on start time. It must land in signals.jsonl with reason "stale_bar".
  4. A QUOTE FUTURE-DATED BY THE TICK'S OWN STALL. `now` is sampled before the bar pull;
     a quote fetched after a slow pull looks like it came from the future and is thrown
     away. Age must be measured against receipt, as the shares arm has done since 977f4ae.
  5. A DELAYED OPTION QUOTE FILLING. Nothing in the runner checked option quote age; the
     only defence was autostart holding both runners until a post-open preflight passed,
     which is what blinded them to the open. Now refused at the point of use.

Each was verified to fail against runner.py as of e4dfaf6, before the back-port.
"""
from __future__ import annotations

import datetime as dt
import json
from types import SimpleNamespace

import pytest

from trade_analysis.live_lab import runner
from trade_analysis.live_lab.session import Signal
from trade_analysis.live_lab.setups import ALL_SETUPS as REAL_SETUPS

DAY = dt.date(2026, 9, 23)


def _t(hhmmss: str) -> dt.datetime:
    return dt.datetime.combine(DAY, dt.time.fromisoformat(hhmmss))


class _AlwaysSignals:
    """A setup that fires on every bar, so the only thing under test is the guard."""

    def __init__(self, tf):
        self.id, self.timeframe = f"STUB_{tf}", tf
        self.max_per_day, self.max_per_direction = 99, None

    def evaluate(self, ctx):
        return SimpleNamespace(direction="long", state={})


@pytest.fixture
def lab(tmp_path, monkeypatch):
    lab = runner.LiveLab(["QQQ"], lab_dir=tmp_path)   # hashes the REAL setups
    # Swapped in only after construction: _evaluate reads the module global per call.
    monkeypatch.setattr(runner, "ALL_SETUPS", [_AlwaysSignals("1m"), _AlwaysSignals("5m")])
    lab.opened = []
    lab._open = lambda setup, sig, sym, ctx, quote, now, bar_ts, day: \
        lab.opened.append((setup.id, bar_ts))
    yield lab
    lab.feed.close()


SESS = SimpleNamespace(context=lambda bar_ts, tf, quote: SimpleNamespace())


def _skips(lab):
    path = lab.store.root / "signals.jsonl"
    if not path.exists():
        return []
    return [r for r in map(json.loads, path.read_text().splitlines())
            if r.get("phase") == "SKIP"]


# ------------------------------------------------------ 1. the Crabel case, exactly

def test_the_0931_signal_seen_at_0936_is_not_opened(lab):
    """2026-09-23: bar 09:31, filled 09:36:49. That fill must not happen."""
    lab._evaluate("QQQ", SESS, _t("09:31:00"), "1m", {"mid": 1.0}, _t("09:36:49"), DAY)
    assert lab.opened == [], "a 5.8-minute-old signal opened a position"


def test_a_healthy_bar_still_opens(lab):
    """~1.1 min from bar open to decision is a normal tick. The guard must not bite it."""
    lab._evaluate("QQQ", SESS, _t("09:31:00"), "1m", {"mid": 1.0}, _t("09:32:05"), DAY)
    assert lab.opened == [("STUB_1m", _t("09:31:00"))]


# ------------------------------------------------------ 2. the 5m stamp

def test_5m_bucket_age_is_measured_from_its_newest_1m_bar(lab):
    """Bucket 09:30-09:35 is stamped 09:35; its newest 1m bar opened 09:34.

    At 09:37:30 that 1m bar is 210s old -> stale, exactly as the shares arm would judge
    the same minute. Measured from the 09:35 stamp it would read 150s and pass.
    """
    lab._evaluate("QQQ", SESS, _t("09:35:00"), "5m", {"mid": 1.0}, _t("09:37:30"), DAY)
    assert lab.opened == [], "5m bucket judged from its close stamp: 60s too lenient"


def test_5m_bucket_on_time_still_opens(lab):
    lab._evaluate("QQQ", SESS, _t("09:35:00"), "5m", {"mid": 1.0}, _t("09:35:03"), DAY)
    assert lab.opened == [("STUB_5m", _t("09:35:00"))]


# ------------------------------------------------------ 3. the skip is recorded

def test_a_suppressed_signal_is_written_as_a_skip(lab):
    lab._evaluate("QQQ", SESS, _t("09:31:00"), "1m", {"mid": 1.0}, _t("09:36:49"), DAY)
    skips = _skips(lab)
    assert [s["skip_reason"] for s in skips] == ["stale_bar"], skips
    assert skips[0]["bar_age_min"] == pytest.approx(5.82, abs=0.01)


# ------------------------------------------------------ 4. quote age at receipt

def test_a_quote_fetched_after_a_slow_bar_pull_is_not_called_future_dated(lab):
    """`now` = 10:00:00 at the top of the tick; the bar pull stalls 20s; the quote is
    stamped 10:00:20 and received 10:00:20.3. It is 0.3s old, not 20s in the future."""
    seen = {}
    quote = {"ts": _t("10:00:20"), "recv_ts": _t("10:00:20.300000"), "mid": 1.0}
    lab.feed = SimpleNamespace(minute_bars=lambda sym, day, now: [],
                               stock_quote=lambda sym: dict(quote),
                               close=lambda: None)
    lab.sessions["QQQ"] = SimpleNamespace(accept_bars=lambda bars, now: [],
                                          degraded_bars=[], bars_5m=[])
    lab._manage_open = lambda sym, sess, q, now: seen.setdefault("quote", q)
    lab._tick(_t("10:00:00"), DAY)
    assert seen["quote"] is not None, "fresh quote discarded as future-dated"
    outages = lab.store.root / "outages.jsonl"
    kinds = [json.loads(l)["kind"] for l in outages.read_text().splitlines()] \
        if outages.exists() else []
    assert "future_quote" not in kinds


def test_the_guard_is_outside_the_config_hash():
    """Adding an operational guard must not fork the frozen options history."""
    cfg = runner.build_config(["QQQ", "SPY"], 1500)
    assert cfg["config_hash"] == "1f7247d7839d9950", cfg["config_hash"]


# ------------------------------------------------------ 5. a delayed OPTION quote

def _chain(ts):
    return [{"ts": ts, "strike": k, "right": "call", "bid": 1.00, "ask": 1.02,
             "mid": 1.01, "spread": 0.02, "bid_size": 10.0, "ask_size": 10.0}
            for k in (99.0, 100.0, 101.0)]


@pytest.fixture
def open_lab(tmp_path, monkeypatch):
    """A lab whose chain the test controls, with the wall clock pinned at 10:00:05."""
    lab = runner.LiveLab(["QQQ"], lab_dir=tmp_path)
    lab.feed.close()
    monkeypatch.setattr(runner, "now_et", lambda: _t("10:00:05"))
    lab.chain = []
    lab.feed = SimpleNamespace(zero_dte=lambda sym, day: DAY,
                               chain_quotes=lambda sym, exp: lab.chain,
                               open_interest=lambda sym, exp: {},
                               close=lambda: None)
    return lab


def _signal(lab):
    setup = next(s for s in REAL_SETUPS if s.id == "Crabel_Stretch")
    sig = Signal(setup.id, "long", stop=99.0, target=None)
    ctx = SimpleNamespace(price=100.0, vwap=100.0, bars_1m=(), bars_5m=(), tf="1m")
    lab._open(setup, sig, "QQQ", ctx, {"mid": 100.0}, _t("10:00:05"), _t("10:00:00"), DAY)
    rows = map(json.loads, (lab.store.root / "signals.jsonl").read_text().splitlines())
    return [r for r in rows if r.get("phase") == "FILL"]


def test_a_delayed_option_chain_does_not_fill(open_lab):
    """ATM last quoted 15 minutes ago: a delayed feed. Until 2026-09-24 nothing in the
    runner looked, and only a gate that held both runners past the open kept this out."""
    open_lab.chain = _chain(_t("09:45:05"))
    fills = _signal(open_lab)
    assert open_lab.open_pos == [], "filled at a 15-minute-old option price"
    assert fills[-1]["status"] == "SKIPPED"
    assert fills[-1]["skip_reason"].startswith("stale_option_quote"), fills[-1]


def test_a_live_option_chain_still_fills_all_three_arms(open_lab):
    open_lab.chain = _chain(_t("10:00:04"))
    fills = _signal(open_lab)
    assert fills[-1]["status"] == "OPEN", fills[-1]
    assert sorted(p.arm for p in open_lab.open_pos) == ["ATM", "ATM+1", "ATM-1"]


def test_option_threshold_is_preflights_definition_of_delayed():
    """One definition of DELAYED, not two that can drift."""
    from trade_analysis.live_lab import preflight
    assert getattr(runner, "OPTION_QUOTE_MAX_AGE_SEC", None) == preflight.DELAYED_THRESHOLD_SEC
