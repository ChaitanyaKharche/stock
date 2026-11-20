"""Per-setup baseline of the two frozen arms: does dropping the losing setups leave a
profitable book? DESCRIPTIVE. Not pre-registered, not audited.

    python -m trade_analysis.news_arms.setup_baseline

Reads the DISCOVERY replays only (shares 2018-2022, options 2020-2022; the holdout is not
opened here, and its per-setup P&L has never been computed) and the live lab record since
the freeze (2026-08-28). Writes research/setup_baseline.json and
research/setup_baseline_results.md; every number in the markdown comes from the JSON.

Three things are measured:
1. per setup, over the replayed years: trades, total, $ per trade, $ per session, by year;
2. a pick-then-check split inside discovery: choose the setups that were net positive in the
   first part, score only the second part (choosing on all of it is hindsight);
3. the live record: how much of each arm's total rests on its largest days, how many days
   reach +$500, and each setup with and without those days.
"""
from __future__ import annotations

import datetime as dt
import json
import subprocess
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "research"
LAB = ROOT / "live_lab_data"
LIVE_FROM = "2026-08-28"
SPLIT = {"shares": "2020-12-31", "options": "2021-06-30"}
TARGET = 500.0                      # the owner's "happy with $500 a day"


def _rows(p: Path) -> list[dict]:
    out = []
    with open(p, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out


def per_setup(trades: list[dict], n_days: int) -> dict:
    """setup -> trades, total, per trade, per session, win rate, total by calendar year."""
    acc = defaultdict(lambda: {"n": 0, "total": 0.0, "wins": 0, "by_year": defaultdict(float)})
    for t in trades:
        a = acc[t["setup_id"]]
        a["n"] += 1
        a["total"] += t["pnl_net"]
        a["wins"] += t["pnl_net"] > 0
        a["by_year"][t["entry_ts"][:4]] += t["pnl_net"]
    return {k: {"trades": a["n"], "total": round(a["total"], 2), "per_trade": round(a["total"] / a["n"], 2),
                "per_session": round(a["total"] / n_days, 2), "win_rate": round(a["wins"] / a["n"], 4),
                "by_year": {y: round(v, 2) for y, v in sorted(a["by_year"].items())}}
            for k, a in sorted(acc.items(), key=lambda kv: -kv[1]["total"])}


def split_test(trades: list[dict], days: list[str], split: str) -> dict:
    """Choose setups net positive up to `split`; report them, and the whole book, after it."""
    first, second = defaultdict(float), defaultdict(float)
    for t in trades:
        (first if t["entry_ts"][:10] <= split else second)[t["setup_id"]] += t["pnl_net"]
    d1 = [d for d in days if d <= split]
    d2 = [d for d in days if d > split]
    chosen = sorted(k for k, v in first.items() if v > 0)
    return {"split": split, "first_sessions": len(d1), "second_sessions": len(d2), "chosen": chosen,
            "chosen_first_per_session": round(sum(first[k] for k in chosen) / max(len(d1), 1), 2),
            "chosen_second_per_session": round(sum(second[k] for k in chosen) / max(len(d2), 1), 2),
            "chosen_second_each": {k: round(second[k], 2) for k in chosen},
            "book_second_per_session": round(sum(second.values()) / max(len(d2), 1), 2)}


def tail_summary(trades: list[dict], target: float = TARGET) -> dict:
    """Day-level view of a live arm: totals with the largest 1, 2, 3 days removed."""
    by_day = defaultdict(float)
    for t in trades:
        by_day[t["entry_ts"][:10]] += t["pnl_net"]
    days = sorted(by_day)
    nets = [by_day[d] for d in days]
    top = sorted(by_day.items(), key=lambda kv: -kv[1])[:3]
    total = sum(nets)
    srt = sorted(nets)
    med = srt[len(srt) // 2] if len(srt) % 2 else (srt[len(srt) // 2 - 1] + srt[len(srt) // 2]) / 2
    out = {"sessions": len(days), "first": days[0], "last": days[-1], "trades": len(trades),
           "total": round(total, 2), "mean_per_session": round(total / len(days), 2), "median_session": round(med, 2),
           "days_at_or_above_target": sum(n >= target for n in nets),
           "days_at_or_below_minus_target": sum(n <= -target for n in nets), "target": target,
           "top_days": [{"date": d, "net": round(v, 2)} for d, v in top], "without_top": {}}
    removed = 0.0
    for i, (d, v) in enumerate(top, start=1):
        removed += v
        out["without_top"][str(i)] = {"total": round(total - removed, 2),
                                      "per_session": round((total - removed) / (len(days) - i), 2)}
    return out


def live_per_setup(trades: list[dict], exclude_days: set[str]) -> dict:
    acc = defaultdict(lambda: [0, 0.0, 0.0, 0])
    for t in trades:
        a = acc[t["setup_id"]]
        a[0] += 1
        a[1] += t["pnl_net"]
        a[3] += t["pnl_net"] > 0
        if t["entry_ts"][:10] not in exclude_days:
            a[2] += t["pnl_net"]
    return {k: {"trades": a[0], "total": round(a[1], 2), "total_without_top_days": round(a[2], 2),
                "per_trade": round(a[1] / a[0], 2), "win_rate": round(a[3] / a[0], 4)}
            for k, a in sorted(acc.items(), key=lambda kv: -kv[1][1])}


def off_ladder(option_trades: list[dict]) -> dict:
    """Live option signals whose 'ATM' strike sat more than 1.5 points from the underlying,
    or whose neighbour arm was more than 2 strikes away: the chain served that minute was
    incomplete and the runner took the nearest strike that had a quote."""
    sig = defaultdict(dict)
    for t in option_trades:
        sig[(t["entry_ts"][:10], t.get("signal_id"))][t["arm"]] = t
    bad, total = Counter(), 0
    for (day, _), arms in sig.items():
        a = arms.get("ATM")
        if not a:
            continue
        total += 1
        wings = [abs(arms[k]["strike"] - a["strike"]) for k in ("ATM+1", "ATM-1") if k in arms]
        if abs(a["strike"] - a["entry_underlying"]) > 1.5 or any(w > 2.01 for w in wings):
            bad[day] += 1
    return {"signals": total, "off_ladder": sum(bad.values()), "by_day": dict(sorted(bad.items()))}


def build() -> dict:
    from . import analyze as A
    sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True,
                         text=True).stdout.strip()
    res = {"generated": dt.datetime.now().isoformat(timespec="seconds"), "repo_sha": sha,
           "status": "descriptive; not pre-registered; not audited; holdout not opened",
           "discovery": {}, "live": {}}
    for arm in ("shares", "options"):
        rows, trades, meta, _ = A.load(arm, "discovery")
        days = sorted(r["date"] for r in rows)
        res["discovery"][arm] = {
            "window": [days[0], days[-1]], "sessions": len(days), "trades": len(trades),
            "config_hashes": meta["hashes"], "book_per_session": round(sum(r["net"] for r in rows) / len(days), 2),
            "per_setup": per_setup(trades, len(days)), "split": split_test(trades, days, SPLIT[arm])}
    all_opt = [t for t in _rows(LAB / "trades.jsonl") if t.get("pnl_net") is not None
               and t.get("entry_ts", "")[:10] >= LIVE_FROM]
    live = {"options": [t for t in all_opt if t.get("arm") == "ATM"],
            "shares": [t for t in _rows(LAB / "shares" / "trades.jsonl") if t.get("pnl_net") is not None
                       and t.get("entry_ts", "")[:10] >= LIVE_FROM]}
    for arm, trades in live.items():
        tail = tail_summary(trades)
        top2 = {d["date"] for d in tail["top_days"][:2]}
        res["live"][arm] = {**tail, "per_setup": live_per_setup(trades, top2), "excluded_for_per_setup": sorted(top2),
                            "reconstructed_trades": sum(1 for t in trades if t.get("exit_mode") == "reconstructed_from_history")}
    res["live"]["options"]["off_ladder"] = off_ladder([t for t in all_opt if t.get("strike") is not None])
    return res


def _money(x, d=0):
    return "—" if x is None else f"{x:+,.{d}f}"


def _pos(res: dict, arm: str) -> str:
    ps = res["discovery"][arm]["per_setup"]
    pos = [f"{k} ({v['trades']:,} trades, {_money(v['per_session'], 1)})" for k, v in ps.items() if v["total"] > 0]
    return ", ".join(pos) if pos else "none"


def render(res: dict) -> str:
    L = [f"""---
task_id: R-2026-11-20-01
trace_id: https://claude.ai/code/session_01TqhCAiwZiZ56SHrYyFXGVT
role: experiment-runner
repo_sha: {res['repo_sha']}
inputs:
  - live_lab_backfill/news_arms/2018..2022/shares, 2020..2022/options (discovery replays, gate-2 audited in news_arms)
  - live_lab_data/trades.jsonl, live_lab_data/shares/trades.jsonl (frozen forward test, from {LIVE_FROM})
decision_rule: "none: descriptive. No test, no verdict, no change to the lab."
outputs:
  - research/setup_baseline_results.md
  - research/setup_baseline.json
evidence:
  - "every number below: research/setup_baseline.json, written by trade_analysis/news_arms/setup_baseline.py"
assumptions:
  - "options replay is SPY only; the live options arm trades QQQ and SPY"
  - "the shares rules were written on 2026 microstructure and replayed on 2018-2022 bars"
open_questions:
  - "the holdout (shares 2023-2026-08, options 2023) has never been read per setup; a pick-then-check on it needs its own pre-registration"
errors: []
status: done
verified_by: null
---

# Per-setup baseline: does dropping the losing setups leave a profitable book?

Generated {res['generated']} by `trade_analysis/news_arms/setup_baseline.py`. **Descriptive.
Not pre-registered, not audited.** The question came from the owner on 2026-11-14: profitable
trades show up in the terminal every day, yet the day ends at −$2,000 to −$3,000, so are good
setups being dragged down by the others, and is $500 a day already achieved?

## The answer in plain words

**No. Over the replayed years there is no profitable group of setups to keep.** Setups that
ended above zero, with their trade counts and dollars per session: shares {_pos(res, 'shares')};
options {_pos(res, 'options')}. Choosing the setups that were ahead in the first part of the
history and keeping only those for the rest gives {_money(res['discovery']['shares']['split']['chosen_second_per_session'])} a session on shares (none
qualified) and {_money(res['discovery']['options']['split']['chosen_second_per_session'])} a session on options. The live totals rest on a few crash days: without
its two largest days the options arm is at {_money(res['live']['options']['without_top']['2']['per_session'])} a session and the shares arm at
{_money(res['live']['shares']['without_top']['2']['per_session'])}.
"""]
    for arm in ("shares", "options"):
        d = res["discovery"][arm]
        years = sorted({y for v in d["per_setup"].values() for y in v["by_year"]})
        L.append(f"## {arm.capitalize()} arm, replay {d['window'][0]} → {d['window'][1]}\n")
        L.append(f"{d['sessions']:,} sessions, {d['trades']:,} trades, hash {d['config_hashes']}. Whole book: "
                 f"**{_money(d['book_per_session'])} a session**.\n")
        L.append("| setup | trades | total $ | $ per trade | $ per session | win % | " + " | ".join(years) + " |")
        L.append("|---|---|---|---|---|---|" + "---|" * len(years))
        for k, v in d["per_setup"].items():
            L.append(f"| {k} | {v['trades']:,} | {_money(v['total'])} | {_money(v['per_trade'], 2)} | {_money(v['per_session'], 1)} | "
                     f"{100 * v['win_rate']:.1f} | " + " | ".join(_money(v["by_year"].get(y, 0.0)) for y in years) + " |")
        s = d["split"]
        pos = [k for k, v in d["per_setup"].items() if v["total"] > 0]
        L.append(f"\nPositive over the whole window (hindsight): {', '.join(pos) if pos else 'none'}"
                 + (f", together {_money(sum(d['per_setup'][k]['per_session'] for k in pos), 1)} a session." if pos else "."))
        L.append(f"\n**Pick-then-check.** First part to {s['split']} ({s['first_sessions']} sessions), second part "
                 f"{s['second_sessions']} sessions. Net positive in the first part: "
                 f"{', '.join(s['chosen']) if s['chosen'] else '**none**'}. Those setups alone in the second part: "
                 f"**{_money(s['chosen_second_per_session'])} a session**"
                 + (f" ({', '.join(f'{k} {_money(v)}' for k, v in s['chosen_second_each'].items())} in total)" if s["chosen"] else "")
                 + f". Whole book in the second part: {_money(s['book_second_per_session'])} a session.\n")
    L.append("## Live lab since the freeze\n")
    L.append("Frozen forward test (`live_lab_data/FREEZE.json`): no setup may be dropped, and the first per-setup "
             "evaluation is at 200 trades per setup. Everything in this section is a diagnostic, not an effect claim.\n")
    L.append("| | options (ATM) | shares |")
    L.append("|---|---|---|")
    o, s = res["live"]["options"], res["live"]["shares"]
    rows = [("sessions", f"{o['sessions']}", f"{s['sessions']}"),
            ("trades", f"{o['trades']:,}", f"{s['trades']:,}"),
            ("total", _money(o["total"]), _money(s["total"])),
            ("mean per session", _money(o["mean_per_session"]), _money(s["mean_per_session"])),
            ("median session", _money(o["median_session"]), _money(s["median_session"])),
            (f"sessions at +${TARGET:,.0f} or better", f"{o['days_at_or_above_target']}", f"{s['days_at_or_above_target']}"),
            (f"sessions at −${TARGET:,.0f} or worse", f"{o['days_at_or_below_minus_target']}", f"{s['days_at_or_below_minus_target']}"),
            ("three largest days", ", ".join(f"{d['date']} {_money(d['net'])}" for d in o["top_days"]),
             ", ".join(f"{d['date']} {_money(d['net'])}" for d in s["top_days"]))]
    for i in ("1", "2", "3"):
        rows.append((f"per session without the largest {i}", _money(o["without_top"][i]["per_session"]),
                     _money(s["without_top"][i]["per_session"])))
    rows.append(("trades whose exit was rebuilt from history", f"{o['reconstructed_trades']}", f"{s['reconstructed_trades']}"))
    for r in rows:
        L.append(f"| {r[0]} | {r[1]} | {r[2]} |")
    ol = o["off_ladder"]
    L.append(f"\nOff-ladder option fills: {ol['off_ladder']} of {ol['signals']} live signals were filled on a strike more than "
             f"1.5 points from the underlying or with a neighbour arm more than two strikes away ({ol['by_day']}). No nearer "
             f"strike had a usable quote in the chain that minute and the runner took the nearest one that did. Cause not "
             f"established: both days were 4% selloffs, so a thin or crossed market, strikes not listed that far from the "
             f"prior close, or a feed gap are all possible. Those rows are in the record as ATM trades.\n")
    for arm in ("options", "shares"):
        a = res["live"][arm]
        L.append(f"### Live {arm}, per setup (diagnostic)\n")
        L.append(f"Last column leaves out the two largest days, {', '.join(a['excluded_for_per_setup'])}.\n")
        L.append("| setup | trades | total $ | $ per trade | win % | total without the two largest days |")
        L.append("|---|---|---|---|---|---|")
        for k, v in a["per_setup"].items():
            L.append(f"| {k} | {v['trades']:,} | {_money(v['total'])} | {_money(v['per_trade'], 2)} | {100 * v['win_rate']:.1f} | "
                     f"{_money(v['total_without_top_days'])} |")
        L.append("")
    L.append("""## What this does and does not say

- It says the mechanical setups, as frozen, have no positive subset over the replayed years
  that survives being chosen in advance. Seeing a +$300 trade in the terminal is real; so are
  the many small losers beside it, and which trade wins is only known afterwards.
- It does not say anything about the owner's own discretionary trading, which is a different
  record (`journal_reverse_engineering_results.md`).
- It does not use the holdout. A fair forward check of "keep only setup X" exists in the
  data (shares 2023–2026-08, options 2023) and needs its own pre-registration before a read.
- The live per-setup columns are below the lab's own 200-trade checkpoint for every setup
  except those at or past it, and the live totals are tail-dominated; they are not evidence
  for or against any setup.
- Replay limits as in `news_arms/preregistration.md` §9: vendor-corrected bars, fills on the
  minute, no outages; VWAP_Reclaim, ORB_5min and MOMO_CHASE replay with low trade-for-trade
  fidelity.
""")
    return "\n".join(L)


def main() -> int:
    res = build()
    (OUT / "setup_baseline.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    (OUT / "setup_baseline_results.md").write_text(render(res), encoding="utf-8")
    for arm in ("shares", "options"):
        d = res["discovery"][arm]
        s = d["split"]
        pos = [k for k, v in d["per_setup"].items() if v["total"] > 0]
        print(f"{arm} discovery: book {d['book_per_session']:+.0f}/session; positive setups {pos or 'none'}; "
              f"pick-then-check {s['chosen'] or 'none'} -> {s['chosen_second_per_session']:+.0f}/session")
    for arm in ("options", "shares"):
        a = res["live"][arm]
        print(f"live {arm}: {a['sessions']} sessions, total {a['total']:+.0f}, median {a['median_session']:+.0f}, "
              f">=+{TARGET:.0f}: {a['days_at_or_above_target']}, <=-{TARGET:.0f}: {a['days_at_or_below_minus_target']}, "
              f"top {[(d['date'], round(d['net'])) for d in a['top_days']]}, without top 2: {a['without_top']['2']['per_session']:+.0f}/session")
    print("written: research/setup_baseline.json, research/setup_baseline_results.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
