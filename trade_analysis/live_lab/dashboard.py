"""Read-only reporting over the lab's JSONL files.

Ranking is by EXPECTANCY WITH A CONFIDENCE INTERVAL AND n, never by raw P&L, because
setups fire at wildly different rates and a big number from four trades is not evidence.

A setup is reported PROMISING only if all four pre-registered conditions hold:
    1. n >= 200 completed ATM trades
    2. clears Holm within its family
    3. |effect| exceeds its own achieved MDE
    4. survives a first-half / second-half split with the same sign

Anything short of that prints INSUFFICIENT, whatever the P&L looks like.

    python -m trade_analysis.live_lab.dashboard
    python -m trade_analysis.live_lab.dashboard --today
"""
from __future__ import annotations

import argparse
import datetime as dt
import math
from collections import defaultdict

import random

from .setups import ALL_SETUPS, DEAD_SETUPS, SLOW_SETUPS
from .store import DEFAULT_LAB_DIR, LabStore

MIN_N_FOR_VERDICT = 200
STALE_FILL_MIN = 3.0   # a fill later than this missed its bar; see fill_latency()
POWERED_N = 377                 # +10% mean return, Holm m=13, 80% power
FAMILY_SIZE = len(ALL_SETUPS)

# Checkpoints fixed by research/forward_test_preregistration.md. The first two are
# DIAGNOSTIC ONLY -- pipeline health, never effect claims.
CHECKPOINTS = [(50, "total", "diagnostic only"), (100, "total", "diagnostic only"),
               (MIN_N_FOR_VERDICT, "per-setup", "first serious evaluation"),
               (POWERED_N, "per-setup", "powered for +10%")]


def load_freeze(store):
    """The freeze record. Its accepted hashes are the ONLY filter applied to trades."""
    import json
    p = store.root / "FREEZE.json"
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def session_coverage(store) -> list[str]:
    """Which weekdays since the freeze produced a session, and which silently did not.

    The freeze forbids excluding a session. That rule is only enforceable if a MISSING
    session is visible -- and until now nothing looked. A day where the machine was asleep,
    the scheduled task never fired, or the runner died before writing its summary left no
    trace at all, so the record would quietly have a hole in it and the trade counts would
    still look healthy.

    Classification, from artefacts on disk only:
      daily/<date>.json exists          -> RECORDED
      only logs/<date>.log exists       -> started but never finished; holiday if it says so
      neither                           -> never ran
    """
    import datetime as _dt
    fz = load_freeze(store)
    if not fz or not fz.get("start_date"):
        return ["  (no FREEZE.json -- cannot determine coverage)"]
    start = _dt.date.fromisoformat(fz["start_date"])
    today = _dt.date.today()
    daily_dir, log_dir = store.root / "daily", store.root / "logs"
    recorded, holidays, gaps = [], [], []
    d = start
    while d <= today:
        if d.weekday() < 5:
            if (daily_dir / f"{d.isoformat()}.json").exists():
                recorded.append(d)
            else:
                lg = log_dir / f"{d.isoformat()}.log"
                txt = ""
                if lg.exists():
                    try:
                        txt = lg.read_text(encoding="utf-8", errors="replace")
                    except OSError:
                        pass
                if "market holiday" in txt or "looks like a market holiday" in txt:
                    holidays.append(d)
                else:
                    gaps.append((d, "started, no summary written" if txt else "never ran"))
        d += _dt.timedelta(days=1)
    out = [f"  weekdays since freeze {start}: {len(recorded) + len(holidays) + len(gaps)}"
           f"   recorded {len(recorded)}   holiday {len(holidays)}   "
           f"UNACCOUNTED {len(gaps)}"]
    if gaps:
        out.append("  MISSING SESSIONS -- the record has holes, and the freeze cannot")
        out.append("  exclude a day it never saw. Investigate before reading any result:")
        for g, why in gaps[:10]:
            out.append(f"    {g}  ({why})")
        if len(gaps) > 10:
            out.append(f"    ... and {len(gaps) - 10} more")
    else:
        out.append("  no unexplained gaps -- every weekday is accounted for")
    return out


def fill_latency(store, arm=None) -> list[str]:
    """How stale was the signal when each fill happened?

    Both arms fill at the live NBBO on the tick that admits a closed bar, which is
    normally ~1 minute after the bar stamp -- the same convention the backtests use. But
    when the vendor feed goes down, bars cannot be admitted at all; they queue, and when
    the feed returns the whole backlog is evaluated and filled at the THEN-current price.

    That is honest -- nothing is fabricated, and the outage is logged -- but a signal
    filled 40 minutes late is not the trade the backtest priced. entry_bar_ts and entry_ts
    are both recorded, so those fills stay identifiable forever. This surfaces them
    instead of leaving them to be noticed by accident.
    """
    import datetime as _dt
    rows = [t for t in store.read("trades.jsonl")
            if t.get("entry_bar_ts") and t.get("entry_ts")
            and (arm is None or t.get("arm") == arm)]
    if not rows:
        return ["  (no trades with a recorded signal bar)"]
    # EXPECTED lag depends on how the setup's bars are stamped, and pooling the two makes
    # the median meaningless. A 1-minute bar is OPEN-stamped, so its close is only knowable
    # a minute later and an honest fill lands at bar_ts + ~60s. A 5-minute bucket is
    # CLOSE-stamped, so an honest fill lands at bar_ts + a few seconds. Reporting raw lag
    # made a normal session look like it had 0.1-minute fills, i.e. lookahead. It did not.
    tf = {s.id: s.timeframe for s in ALL_SETUPS}
    lags = []
    for t in rows:
        try:
            raw = (_dt.datetime.fromisoformat(t["entry_ts"])
                   - _dt.datetime.fromisoformat(t["entry_bar_ts"])).total_seconds() / 60.0
        except (ValueError, TypeError):
            continue
        expected = 1.0 if tf.get(t.get("setup_id")) == "1m" else 0.0
        lags.append((raw - expected, t))
    if not lags:
        return ["  (no parsable timestamps)"]
    v = sorted(x for x, _ in lags)
    stale = [(x, t) for x, t in lags if x > STALE_FILL_MIN]
    out = [f"  EXCESS lag over each setup's own convention (1m bars open-stamped,"
           f" 5m close-stamped)",
           f"  n={len(v)}  median {v[len(v)//2]:+.1f} min  p90 "
           f"{v[int(0.9*(len(v)-1))]:+.1f}  max {max(v):+.1f}",
           f"  fills more than {STALE_FILL_MIN:.0f} min after their signal bar: "
           f"{len(stale)} ({100*len(stale)/len(lags):.1f}%)"]
    if stale:
        net_all = sum(t.get("pnl_net", 0.0) or 0.0 for _, t in lags)
        net_ok = sum(t.get("pnl_net", 0.0) or 0.0 for x, t in lags
                     if x <= STALE_FILL_MIN)
        out.append(f"  net including them ${net_all:+,.2f}   excluding them "
                   f"${net_ok:+,.2f}")
        out.append("  a stale fill is a FEED OUTAGE artefact, not a strategy result --")
        out.append("  the backtest fills one minute after the bar, so these are not the")
        out.append("  same trade and should be reported separately, never silently kept")
        by = {}
        for x, t in stale:
            by.setdefault(t.get("entry_ts", "")[:10], []).append(x)
        for d in sorted(by)[-5:]:
            out.append(f"    {d}: {len(by[d])} stale, worst {max(by[d]):.0f} min")
    return out


def signal_reconciliation(store, arms_per_signal=3) -> list[str]:
    """Every DECISION must end as a recorded trade. This is the general leak detector.

    It does not care WHY a position vanished -- supervisor kill, crash, sleep, power loss.
    It just checks the invariant: decisions x arms == trades, per session. On 2026-09-01/02/03
    the supervisor terminated both arms at 15:55, the exact minute they were due to flatten,
    and every position still open at the close was silently dropped. The trade counts still
    looked healthy, so nothing surfaced it for three sessions.

    The loss was not random: it removed exactly one exit type (`eod`), i.e. the trades that
    had NOT yet stopped out. On a trend day those are the winners.
    """
    sig = [x for x in store.read("signals.jsonl") if x.get("phase") == "DECISION"]
    tr = store.read("trades.jsonl")
    if not sig:
        return ["  (no decisions recorded)"]
    by_sig, by_tr = defaultdict(int), defaultdict(int)
    for x in sig:
        by_sig[str(x.get("bar_ts") or x.get("ts"))[:10]] += 1
    for t in tr:
        by_tr[str(t.get("entry_ts"))[:10]] += 1
    out = [f"  {'session':<13}{'decided':>9}{'expected':>10}{'recorded':>10}{'lost':>7}"]
    total_lost = 0
    for d in sorted(by_sig):
        exp = by_sig[d] * arms_per_signal
        got = by_tr.get(d, 0)
        lost = exp - got
        total_lost += max(lost, 0)
        flag = "  <-- POSITIONS LOST" if lost > 0 else ("  <-- DUPLICATES" if lost < 0 else "")
        out.append(f"  {d:<13}{by_sig[d]:>9}{exp:>10}{got:>10}{lost:>7}{flag}")
    out.append(f"  total unrecorded positions: {total_lost}")
    if total_lost:
        out.append("  a session that ends with positions unrecorded is a DATA LOSS, not a")
        out.append("  result. Check live_lab_data/recovery_archive before the next session")
        out.append("  overwrites the snapshot -- see live_lab/reconstruct_eod.py")
    return out


def checkpoint_status(store, rows) -> list[str]:
    fz = load_freeze(store)
    out = []
    total = sum(r.get("n", 0) for r in rows.values())
    out.append(f"  frozen {fz['start_date'] if fz else '(no FREEZE.json)'}"
               f"   total ATM trades since freeze: {total}")
    for n, scope, label in CHECKPOINTS:
        if scope == "total":
            hit = total >= n
            out.append(f"    [{'x' if hit else ' '}] {n:>4} total      {label}"
                       f"{'' if hit else f'   ({n - total} to go)'}")
        else:
            ready = sorted(k for k, r in rows.items() if r.get("n", 0) >= n)
            best = max((r.get("n", 0) for r in rows.values()), default=0)
            out.append(f"    [{'x' if ready else ' '}] {n:>4} per setup  {label}"
                       f"   ({len(ready)}/{FAMILY_SIZE} there; best setup has {best})")
    if fz:
        out.append("  " + "-" * 96)
        out.append("  CLOCK RULE: the counter never resets. No session, setup or date can be")
        out.append("  excluded. A setup performing badly is a RESULT, not a reason to restart it.")
    return out


# --------------------------------------------------------------------------- stats


def _day_bootstrap(values, days, reps=4000, seed=20260827):
    """Day-clustered bootstrap of the mean. Returns (lo, hi, se, p_two_sided)."""
    if len(values) < 5:
        return (None, None, None, None)
    rng = random.Random(seed)
    by_day = defaultdict(list)
    for v, d in zip(values, days):
        by_day[d].append(v)
    keys = list(by_day)
    if len(keys) < 3:
        return (None, None, None, None)
    means = []
    for _ in range(reps):
        pool = []
        for _ in range(len(keys)):
            pool.extend(by_day[keys[rng.randrange(len(keys))]])
        if pool:
            means.append(sum(pool) / len(pool))
    if not means:
        return (None, None, None, None)
    means.sort()
    lo = means[int(0.025 * len(means))]
    hi = means[int(0.975 * len(means))]
    m = sum(means) / len(means)
    se = math.sqrt(sum((x - m) ** 2 for x in means) / max(len(means) - 1, 1))
    neg = sum(1 for x in means if x <= 0) / len(means)
    pos = sum(1 for x in means if x >= 0) / len(means)
    p = max(2 * min(neg, pos), 1.0 / reps)
    return (lo, hi, se, p)


def _holm(pvals: dict[str, float], m: int) -> dict[str, float]:
    order = sorted(pvals, key=lambda k: pvals[k])
    adj, run = {}, 0.0
    for i, k in enumerate(order):
        run = max(run, (m - i) * pvals[k])
        adj[k] = min(run, 1.0)
    return adj


# --------------------------------------------------------------------------- report


def build(store: LabStore, arm="ATM"):
    # The accepted config hashes from FREEZE.json are the ONLY filter applied to trades.
    # This enforces the clock rule from both directions: trades under a forked definition
    # cannot silently pool into a frozen setup's count, and no OTHER exclusion -- date,
    # session, regime, drawdown -- has anywhere to attach.
    fz = load_freeze(store)
    accepted = set((fz or {}).get("accepted_config_hashes") or [])
    trades = [t for t in store.read("trades.jsonl")
              if t.get("arm") == arm and t.get("return_pct") is not None
              and (not accepted or t.get("config_hash") in accepted)]
    sigs = [s for s in store.read("signals.jsonl")
            if not accepted or s.get("config_hash") in accepted
            or s.get("phase") == "FILL"]      # FILL rows carry no hash; keyed by signal_id

    decided = defaultdict(int)
    skipped = defaultdict(lambda: defaultdict(int))
    for s in sigs:
        if s.get("phase") == "DECISION":
            decided[s["setup_id"]] += 1
        elif s.get("phase") == "SKIP":
            skipped[s.get("setup_id", "?")][s.get("skip_reason", "?")] += 1
        elif s.get("phase") == "FILL" and s.get("status") == "SKIPPED":
            skipped[s.get("setup_id", "?")][s.get("skip_reason", "?")] += 1

    rows, pvals = {}, {}
    for setup in ALL_SETUPS:
        mine = [t for t in trades if t["setup_id"] == setup.id]
        n = len(mine)
        if n == 0:
            rows[setup.id] = {"n": 0, "signals": decided.get(setup.id, 0)}
            continue
        rets = [t["return_pct"] for t in mine]
        nets = [t["pnl_net"] for t in mine if t.get("pnl_net") is not None]
        days = [str(t["entry_ts"])[:10] for t in mine]
        und = [t["underlying_return"] for t in mine if t.get("underlying_return") is not None]
        wins = [r for r in rets if r > 0]
        losses = [r for r in rets if r <= 0]
        lo, hi, se, p = _day_bootstrap(rets, days)
        u_lo, u_hi, _, u_p = _day_bootstrap(und, [str(t["entry_ts"])[:10] for t in mine
                                                  if t.get("underlying_return") is not None])
        half = n // 2
        h1 = sum(rets[:half]) / half if half else None
        h2 = sum(rets[half:]) / (n - half) if n - half else None
        rows[setup.id] = {
            "n": n, "signals": decided.get(setup.id, 0),
            "mean": sum(rets) / n, "median": sorted(rets)[n // 2],
            "win_rate": len(wins) / n,
            "avg_win": (sum(wins) / len(wins)) if wins else 0.0,
            "avg_loss": (sum(losses) / len(losses)) if losses else 0.0,
            "expectancy": sum(rets) / n,
            "net": sum(nets),
            "ci": (lo, hi), "se": se, "p": p,
            "mde": (2.8 * se) if se else None,
            "und_mean": (sum(und) / len(und)) if und else None,
            "und_ci": (u_lo, u_hi),
            "half1": h1, "half2": h2,
            "max_dd": _max_dd(nets),
            "mean_hold": sum(t["hold_minutes"] for t in mine
                             if t.get("hold_minutes")) / max(n, 1),
            # .get, not [] -- trades.jsonl is append-only and accumulates rows written by
            # different code paths (live fills, reconstructed EOD exits, future arms). A
            # missing OPTIONAL field must degrade one statistic, never crash the whole
            # report: a dashboard that dies on an unexpected key is how a data problem
            # becomes invisible.
            "mean_entry_spread": (sum(t.get("entry_spread_pct") or 0.0 for t in mine)
                                  / max(n, 1)),
            "skips": dict(skipped.get(setup.id, {})),
        }
        if p is not None:
            pvals[setup.id] = p

    adj = _holm(pvals, FAMILY_SIZE) if pvals else {}
    for k, v in adj.items():
        rows[k]["holm"] = v
    return rows, trades


def _max_dd(nets):
    peak = cum = dd = 0.0
    for x in nets:
        cum += x
        peak = max(peak, cum)
        dd = min(dd, cum - peak)
    return dd


def verdict(r: dict) -> str:
    if r.get("n", 0) < MIN_N_FOR_VERDICT:
        return f"INSUFFICIENT (n={r.get('n',0)}/{MIN_N_FOR_VERDICT})"
    if r.get("holm") is None or r["holm"] >= 0.05:
        return "no effect (Holm)"
    if r.get("mde") and abs(r["mean"]) < r["mde"]:
        return "underpowered for its own effect"
    h1, h2 = r.get("half1"), r.get("half2")
    if h1 is None or h2 is None or (h1 > 0) != (h2 > 0):
        return "fails the split-half check"
    return "*** PROMISING ***"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Live lab dashboard (read-only).")
    ap.add_argument("--lab-dir", default=str(DEFAULT_LAB_DIR))
    ap.add_argument("--arm", default="ATM", choices=["ATM", "ATM-1", "ATM+1"])
    ap.add_argument("--today", action="store_true")
    args = ap.parse_args(argv)

    store = LabStore(args.lab_dir)
    rows, trades = build(store, args.arm)

    if args.today:
        from .clock import today_et
        today = today_et().isoformat()
        t = [x for x in trades if str(x["entry_ts"]).startswith(today)]
        openp = store.load_open_positions()
        print(f"\nTODAY {today}   closed={len(t)}  open={len(openp)}  "
              f"net=${sum(x['pnl_net'] for x in t if x.get('pnl_net')):+.2f}")
        for x in sorted(t, key=lambda z: z["entry_ts"]):
            print(f"  {x['entry_ts'][11:16]} {x['setup_id']:<26} {x['direction']:<5} "
                  f"K={x['strike']:<7} {x['exit_reason']:<8} {x['pnl_net']:+8.2f}")
        for p in openp:
            print(f"  OPEN  {p['setup_id']:<26} {p['direction']:<5} K={p['strike']}")
        return 0

    print("\n" + "=" * 112)
    print("FORWARD TEST CHECKPOINTS")
    print("=" * 112)
    for line in checkpoint_status(store, rows):
        print(line)
    print("  " + "-" * 96)
    print("  SIGNAL RECONCILIATION (decisions -> recorded trades)")
    for line in signal_reconciliation(store, 3 if args.arm != "SHARES" else 1):
        print(line)
    print("  " + "-" * 96)
    print("  FILL LATENCY (signal bar -> actual fill)")
    for line in fill_latency(store, arm=args.arm):
        print(line)
    print("  " + "-" * 96)
    print("  SESSION COVERAGE")
    for line in session_coverage(store):
        print(line)

    print("\n" + "=" * 112)
    print(f"LIVE LAB -- arm={args.arm}   family m={FAMILY_SIZE}   "
          f"promotion bar: n>={MIN_N_FOR_VERDICT}, Holm<0.05, |effect|>MDE, split-half agrees")
    print("=" * 112)
    print(f"  {'setup':<27}{'sig':>5}{'n':>5}{'win%':>7}{'mean':>9}{'expect':>9}"
          f"{'net$':>10}{'maxDD':>9}{'Holm':>8}   verdict")
    order = sorted(ALL_SETUPS, key=lambda s: -(rows.get(s.id, {}).get("n", 0)))
    for s in order:
        r = rows.get(s.id, {})
        if not r.get("n"):
            tag = ("  [DEAD - cannot accumulate evidence]" if s.id in DEAD_SETUPS
                   else "  [SLOW]" if s.id in SLOW_SETUPS else "")
            print(f"  {s.id:<27}{r.get('signals',0):>5}{0:>5}{'-':>7}{'-':>9}{'-':>9}"
                  f"{'-':>10}{'-':>9}{'-':>8}   no trades yet{tag}")
            continue
        h = r.get("holm")
        print(f"  {s.id:<27}{r['signals']:>5}{r['n']:>5}{100*r['win_rate']:>6.1f}%"
              f"{100*r['mean']:>8.2f}%{100*r['expectancy']:>8.2f}%{r['net']:>10.2f}"
              f"{r['max_dd']:>9.2f}{(f'{h:.4f}' if h is not None else '-'):>8}   {verdict(r)}"
              + ("  [DEAD]" if s.id in DEAD_SETUPS
                 else "  [SLOW]" if s.id in SLOW_SETUPS else ""))

    print("\n  UNDERLYING TWIN -- separates 'no directional edge' from 'option structure eats it'")
    print(f"  {'setup':<27}{'n':>5}{'option mean':>13}{'underlying mean':>17}   reading")
    for s in order:
        r = rows.get(s.id, {})
        if not r.get("n") or r.get("und_mean") is None:
            continue
        o, u = r["mean"], r["und_mean"]
        if u <= 0 and o <= 0:
            read = "no directional edge"
        elif u > 0 and o <= 0:
            read = "direction OK, option structure eats it"
        elif u > 0 and o > 0:
            read = "both positive"
        else:
            read = "option positive, underlying not -- suspect noise"
        print(f"  {s.id:<27}{r['n']:>5}{100*o:>12.2f}%{100*u:>16.3f}%   {read}")

    tot = sum(r.get("n", 0) for r in rows.values())
    print(f"\n  total closed {args.arm} trades: {tot}")
    if tot < MIN_N_FOR_VERDICT:
        print(f"  Nothing here is evidence yet. First verdict expected around "
              f"n={MIN_N_FOR_VERDICT} per setup.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
