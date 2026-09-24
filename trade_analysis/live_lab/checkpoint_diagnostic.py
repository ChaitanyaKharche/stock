"""The 50/100-trade DIAGNOSTIC checkpoint. Pipeline health only -- no effect claims.

    python -m trade_analysis.live_lab.checkpoint_diagnostic

WHAT THIS IS ALLOWED TO SAY, AND WHAT IT IS NOT
-----------------------------------------------
`live_lab_data/FREEZE.json` and `research/forward_test_preregistration.md` §3 define the
50- and 100-trade checkpoints identically and narrowly:

    "DIAGNOSTIC ONLY. Pipeline health, data quality, outages. No effect claims."

The first serious evaluation is **200 per setup**, and the promotion bar (§4) is four
conditions, not one. So this file answers "is the apparatus recording what it claims to
record" and nothing else. It deliberately does NOT rank setups by P&L, test any setup, or
suggest promoting or dropping one.

P&L is printed anyway, because §3 says reading it is "not forbidden -- the dashboard shows
it, and pretending otherwise would be theatre". It is labelled as an observation, and no
conclusion is drawn from it here.

Both checkpoints were passed long ago without being recorded -- options ATM crossed 50 and
100 during the first week and stands at 173 -- so this runs them late rather than not at
all. Running it late is itself a finding and is stated in the output.

THE ONE PRE-REGISTERED ALARM THIS FILE EXISTS TO CHECK
------------------------------------------------------
`live_lab_data/shares/FREEZE.json` justifies pooling the arm's two config hashes on the
grounds that the shares arm has no cross-symbol coupling -- caps are per (setup, symbol),
the one-open-position rule filters on symbol, every trade is an independent $10,000 clip.
It then names the single thing that would falsify that:

    "The one residual coupling is operational, not definitional: symbols share one feed
     connection ... a sustained rise in `bars_failed` outages would break this
     justification and must be treated as such."

That is a monitoring commitment made in advance, so it is checked here first and by
session, normalised per session rather than in total -- a raw count rises simply because
sessions accumulate, which would look alarming while meaning nothing.

READ-ONLY, AND IT NEVER TOUCHES A REPLACE TARGET
------------------------------------------------
Only append-only files and the per-session `daily/*.json` summaries are read.
`positions_open.json` is never opened: it is the destination of an `os.replace`, and on
Windows that makes an external reader crash the WRITER with WinError 5. That is not
hypothetical -- it killed the shares arm at 10:51:09 on 2026-09-15. See
`research/incident_2026-09-15_shares_arm_gap.md`.
"""
from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

LAB = Path(__file__).resolve().parents[2] / "live_lab_data"

# NYSE closures inside the forward test's span. Without these a closed market reads as
# missing data, which is the opposite of the truth and would send someone hunting a
# non-existent outage. Extend as the test runs past them.
HOLIDAYS = {
    "2026-09-07",   # Labor Day
    "2026-11-26",   # Thanksgiving
    "2026-12-25",   # Christmas
}


def jsonl(p: Path):
    if not p.exists():
        return
    for line in p.open(encoding="utf-8", errors="replace"):
        line = line.strip()
        if not line:
            continue
        try:
            yield json.loads(line)
        except Exception:
            continue


def dailies(d: Path) -> list[dict]:
    if not d.exists():
        return []
    out = []
    for p in sorted(d.glob("*.json")):
        try:
            out.append(json.loads(p.read_text(encoding="utf-8")))
        except Exception:
            print(f"    [FAIL] unreadable daily record: {p.name}")
    return out


def rule(title: str) -> None:
    print()
    print("=" * 78)
    print(f"  {title}")
    print("=" * 78)


def provenance(name: str, recs: list[dict], freeze: dict) -> bool:
    """Every session must have run under a hash the freeze accepts. This is the check that
    makes the trade counter mean anything: an unrecognised hash is a different experiment
    wearing the same name."""
    accepted = set(freeze.get("accepted_config_hashes", {}))
    seen = collections.Counter(r.get("config_hash") for r in recs)
    ok = True
    print(f"  {name}: {len(recs)} session records, start_date {freeze.get('start_date')}")
    for h, n in seen.most_common():
        mark = "OK  " if h in accepted else "FAIL"
        if h not in accepted:
            ok = False
        note = freeze.get("accepted_config_hashes", {}).get(h, "NOT IN FREEZE MANIFEST")
        print(f"    [{mark}] {h}  {n:>3} sessions -- {note[:60]}")
    early = [r["date"] for r in recs if r.get("date", "") < freeze.get("start_date", "")]
    if early:
        ok = False
        print(f"    [FAIL] {len(early)} session(s) precede start_date: {early}")
    return ok


def coverage(name: str, recs: list[dict]) -> None:
    """Which sessions exist. A hole is not a failure by itself -- the arm may not have been
    running -- but an undocumented hole read as a continuous record is."""
    dates = sorted(r["date"] for r in recs if r.get("date"))
    if not dates:
        print(f"  {name}: no sessions")
        return
    print(f"  {name}: {len(dates)} sessions, {dates[0]} -> {dates[-1]}")
    import datetime as dt
    d0 = dt.date.fromisoformat(dates[0])
    d1 = dt.date.fromisoformat(dates[-1])
    have = set(dates)
    missing = []
    d = d0
    while d <= d1:
        if d.weekday() < 5 and d.isoformat() not in have and d.isoformat() not in HOLIDAYS:
            missing.append(d.isoformat())
        d += dt.timedelta(days=1)
    if missing:
        print(f"    [WARN] {len(missing)} weekday(s) inside the span with NO session "
              f"record:")
        print(f"           {', '.join(missing)}")
        print(f"           Not necessarily a defect -- holidays and arm downtime both look")
        print(f"           like this -- but the span must not be quoted as continuous.")
    else:
        print("    [OK]   no weekday gaps inside the span")


def bars_failed_guard(recs: list[dict]) -> bool:
    """The shares freeze's own falsification condition, checked per session."""
    per = collections.Counter()
    sess = set()
    for r in jsonl(LAB / "shares" / "outages.jsonl"):
        day = str(r.get("ts", ""))[:10]
        if not day:
            continue
        sess.add(day)
        if r.get("kind") == "bars_failed":
            per[day] += 1
    days = sorted(sess)
    if not days:
        print("    no shares outage records")
        return True
    print("    bars_failed per session (the pooling guard):")
    for d in days:
        bar = "#" * min(60, per[d])
        print(f"      {d}  {per[d]:>4}  {bar}")
    half = max(1, len(days) // 2)
    early = sum(per[d] for d in days[:-half]) / max(1, len(days) - half)
    late = sum(per[d] for d in days[-half:]) / half
    print(f"    mean per session -- earlier half {early:.1f}, recent half {late:.1f}")
    # "Sustained rise" needs both a material ratio AND a non-trivial level; a jump from
    # 0.2 to 0.8 triples the ratio while meaning nothing operationally.
    if late > 3 * max(early, 0.5) and late >= 5:
        print("    [ALARM] sustained rise in bars_failed. The shares FREEZE names this as")
        print("            the condition that BREAKS the justification for pooling its two")
        print("            config hashes. Treat it as such -- do not pool until resolved.")
        return False
    print("    [OK]   no sustained rise; the two-hash pooling justification still holds")
    return True


def quality(name: str, recs: list[dict], trades: Path | None = None,
            primary: str = "ATM") -> bool:
    """Per-session health, and a RECONCILIATION of the summary layer against the log.

    The net column is computed from `trades.jsonl` -- the append-only primary record --
    and the daily summary is then CHECKED against it, never trusted in its place. That
    ordering is the point of the function.

    An earlier version did the opposite, reading `r.get("atm_net", r.get("net", ...))`.
    That fallback silently substitutes a DIFFERENT QUANTITY when the first key is absent:
    sessions 2026-09-01..03 record `net`, which is the sum over all three strike arms,
    where every other session records `atm_net`, which is the primary arm alone. The
    fallback therefore printed an all-arms number in an ATM column, roughly 3x too large,
    with no warning -- +$2,413 of phantom profit on 09-03 by itself. Same silent-fallback
    shape as the NaN placebo in the sweep audit: the code had an answer for the missing
    case and the answer was wrong.
    """
    per = collections.defaultdict(lambda: collections.defaultdict(float))
    if trades:
        for r in jsonl(trades):
            if r.get("exit_ts"):
                per[r["exit_ts"][:10]][r.get("arm") or primary] += float(
                    r.get("pnl_net") or 0)

    print(f"  {name}   net column is {primary} from trades.jsonl, not the daily summary")
    print(f"    {'date':<12}{'closed':>7}{'skipfill':>9}{'degraded':>9}"
          f"{'retry/1k':>10}{f'{primary} net':>12}  summary field")
    tot_sf = tot_dg = 0
    mismatches = []
    for r in sorted(recs, key=lambda x: x.get("date", "")):
        d = r.get("date", "")
        feed = r.get("feed") or {}
        calls = float(feed.get("calls") or 0)
        rate = 1000 * float(feed.get("retries") or 0) / calls if calls else 0.0
        dg = r.get("degraded_bars") or {}
        ndg = sum(dg.values()) if isinstance(dg, dict) else 0
        sf = int(r.get("fills_skipped") or 0)
        tot_sf += sf
        tot_dg += ndg

        truth = per[d].get(primary)
        alls = sum(per[d].values()) if per[d] else None
        if "atm_net" in r:
            field, val = "atm_net", float(r["atm_net"])
        elif "net" in r:
            field, val = "net", float(r["net"])
        else:
            field, val = "(absent)", None

        note = field
        if truth is not None and val is not None:
            if abs(val - truth) > 0.01:
                if alls is not None and abs(val - alls) < 0.01:
                    note = f"{field} = ALL ARMS, not {primary}"
                else:
                    note = f"{field} disagrees ({val:+,.2f})"
                mismatches.append((d, field, val, truth))
        ts = f"{truth:+,.2f}" if truth is not None else "--"
        print(f"    {d:<12}{int(r.get('trades_closed') or 0):>7}"
              f"{sf:>9}{ndg:>9}{rate:>10.2f}{ts:>12}  {note}")

    print(f"    totals: fills_skipped={tot_sf}, degraded_bars={tot_dg}")
    if mismatches:
        print(f"    [FAIL] {len(mismatches)} session(s) whose summary field does not equal")
        print(f"           the {primary} total in trades.jsonl. The primary record is")
        print(f"           authoritative and intact, so nothing is lost -- but any tool")
        print(f"           summing 'the daily net' across sessions mixes two different")
        print(f"           quantities. Sum trades.jsonl per arm instead.")
        for d, f_, v, t in mismatches:
            print(f"             {d}  {f_}={v:+,.2f}  vs {primary}={t:+,.2f}")
    else:
        print(f"    [OK]   every daily summary equals the {primary} total in trades.jsonl")
    if tot_sf == 0:
        print("    [OK]   no fill was ever skipped -- every decided signal became a trade")
    else:
        print(f"    [WARN] {tot_sf} skipped fills: decided signals that never became")
        print("           trades. Selection risk if it correlates with conditions.")
    return not mismatches


def funnel(name: str, recs: list[dict], sig_path: Path) -> None:
    dec = sum(int(r.get("signals_decided") or 0) for r in recs)
    skip = sum(int(r.get("signals_skipped") or 0) for r in recs)
    tot = dec + skip
    print(f"  {name}: {dec:,} decided / {tot:,} evaluated "
          f"({100*dec/tot if tot else 0:.1f}% conversion)")
    reasons = collections.Counter(
        r.get("skip_reason") for r in jsonl(sig_path) if r.get("phase") == "SKIP")
    if reasons:
        print("    why signals were skipped:")
        for k, v in reasons.most_common(8):
            print(f"      {str(k):<28}{v:>7,}  ({100*v/sum(reasons.values()):.1f}%)")


def outages(name: str, path: Path) -> None:
    kinds = collections.Counter(r.get("kind") for r in jsonl(path))
    if not kinds:
        print(f"  {name}: none recorded")
        return
    print(f"  {name}: {sum(kinds.values()):,} records")
    for k, v in kinds.most_common(12):
        print(f"      {str(k):<24}{v:>8,}")


def restarts(name: str, path: Path) -> None:
    ev = collections.Counter(r.get("kind") for r in jsonl(path))
    starts = ev.get("start", 0)
    print(f"  {name}: {starts} arm start(s) recorded")
    for k in ("recovered", "aborted", "stopped"):
        if ev.get(k):
            print(f"      {k:<24}{ev[k]:>6}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.parse_args(argv)

    opt_d = dailies(LAB / "daily")
    sha_d = dailies(LAB / "shares" / "daily")
    opt_f = json.loads((LAB / "FREEZE.json").read_text(encoding="utf-8"))
    sha_f = json.loads((LAB / "shares" / "FREEZE.json").read_text(encoding="utf-8"))

    # `trades_closed` in the daily record counts EVERY strike arm (ATM, ATM-1, ATM+1),
    # so it is ~3x the checkpoint's unit. FREEZE.json sets "primary_arm": "ATM", and the
    # 50/100 counters are on that arm.
    n_opt = sum(1 for r in jsonl(LAB / "trades.jsonl")
                if r.get("exit_ts") and r.get("arm") == "ATM")
    n_all = sum(int(r.get("trades_closed") or 0) for r in opt_d)
    print("=" * 78)
    print("  LIVE LAB -- 50/100-TRADE DIAGNOSTIC CHECKPOINT")
    print("  DIAGNOSTIC ONLY: pipeline health, data quality, outages. NO EFFECT CLAIMS.")
    print("=" * 78)
    print(f"  Both checkpoints were passed without being recorded. The PRIMARY (ATM) arm")
    print(f"  has closed {n_opt} trades -- {n_all} across all three strike arms -- so 50")
    print(f"  and 100 are long behind us. This runs them late; that it is late is itself")
    print(f"  part of the record.")

    rule("1. PROVENANCE -- did every session run under an accepted config hash?")
    ok_p = provenance("OPTIONS", opt_d, opt_f)
    print()
    ok_s = provenance("SHARES ", sha_d, sha_f)

    rule("2. THE PRE-REGISTERED POOLING GUARD (shares FREEZE, own words)")
    ok_b = bars_failed_guard(sha_d)

    rule("3. SESSION COVERAGE")
    coverage("OPTIONS", opt_d)
    print()
    coverage("SHARES ", sha_d)

    rule("4. DATA QUALITY, AND SUMMARY-vs-LOG RECONCILIATION")
    ok_r1 = quality("OPTIONS", opt_d, LAB / "trades.jsonl", "ATM")
    print()
    ok_r2 = quality("SHARES ", sha_d, LAB / "shares" / "trades.jsonl", "SHARES")

    rule("5. SIGNAL FUNNEL")
    funnel("OPTIONS", opt_d, LAB / "signals.jsonl")
    print()
    funnel("SHARES ", sha_d, LAB / "shares" / "signals.jsonl")

    rule("6. OUTAGES BY KIND")
    outages("OPTIONS", LAB / "outages.jsonl")
    print()
    outages("SHARES ", LAB / "shares" / "outages.jsonl")

    rule("7. ARM LIFECYCLE")
    restarts("OPTIONS", LAB / "events.jsonl")
    print()
    restarts("SHARES ", LAB / "shares" / "events.jsonl")

    rule("VERDICT -- apparatus only")
    allok = ok_p and ok_s and ok_b and ok_r1 and ok_r2
    print(f"    provenance (both arms)      {'PASS' if ok_p and ok_s else 'FAIL'}")
    print(f"    shares pooling guard        {'PASS' if ok_b else 'ALARM'}")
    print(f"    summary-vs-log reconcile    {'PASS' if ok_r1 and ok_r2 else 'FAIL'}")
    print()
    if allok:
        print("    The apparatus is recording what it claims to record. That is the ONLY")
        print("    claim this checkpoint is permitted to make.")
    else:
        print("    An integrity check FAILED. Resolve before the 200-trade evaluation,")
        print("    because the promotion bar assumes the counter is meaningful.")
    print()
    print("    NOT permitted here and not attempted: ranking setups by P&L, testing any")
    print("    setup, or promoting or dropping one. First serious evaluation is 200 per")
    print("    setup (FREEZE.json), where the bar is four conditions and not one.")
    return 0 if allok else 1


if __name__ == "__main__":
    sys.exit(main())
