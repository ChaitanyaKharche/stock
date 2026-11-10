"""Render research/news_arms/results_tables.md from the analysis JSONs, so no number in the
write-up is typed by hand (preregistration section 10: analysis.json holds every number quoted).

    python -m trade_analysis.news_arms.report
"""
from __future__ import annotations

import json
from pathlib import Path

from . import join as J

OUT = J.OUT


def _f(x, d=3, sign=True):
    if x is None:
        return "—"
    return f"{x:+.{d}f}" if sign else f"{x:.{d}f}"


def _ci(c):
    if not c or c[0] is None:
        return "no CI (n < 5)"
    return f"[{c[0]:+.2f}, {c[1]:+.2f}]"


def _status(c):
    if c.get("never_tested"):
        return "never tested"
    if c.get("tested"):
        s = "tested"
        s += ", BH pass" if c.get("bh_q10") else ", BH fail"
        return s
    if c.get("descriptive_placebo"):
        return "descriptive + placebo"
    return "reported only"


def _pl(c):
    p = c.get("placebo")
    if not p:
        return "—"
    return f"{p['rank']:.3f} → {p['verdict']}"


def q1_table(d: dict, single: bool = False) -> str:
    key = "q1_single_category" if single else "q1"
    native = "$/day" if d["arm"] == "shares" else "$/day (ATM)"
    rows = [f"| category | sessions | trades | effect σ (vs B0) | 95% CI (day-clustered) | {native} | "
            f"undiluted σ | placebo rank → verdict | overlap | status |",
            "|---|---|---|---|---|---|---|---|---|---|"]
    for cat, c in d[key].items():
        conf = c.get("confounded_by_overlap")
        conf_s = "confounded" if conf else ("—" if conf is None else "stable")
        rows.append(f"| {cat} | {c['n_sessions']} | {c['n_trades']} | {_f(c['effect_sigma'])} | {_ci(c['ci95'])} | "
                    f"{_f(c['effect_native'], 0)} | {_f(c['undiluted_sigma'])} | {_pl(c)} | {conf_s} | {_status(c)} |")
    return "\n".join(rows)


def q3_table(d: dict) -> str:
    rows = ["| condition | sessions | trades | effect σ | 95% CI | $/day | placebo mean ± sd | placebo rank → verdict | status |",
            "|---|---|---|---|---|---|---|---|---|"]
    for k, c in d["q3"].items():
        p = c.get("placebo")
        pm = f"{p['mean']:+.3f} ± {p['sd']:.3f}" if p else "—"
        rows.append(f"| {k} | {c['n_sessions']} | {c['n_trades']} | {_f(c['effect_sigma'])} | {_ci(c['ci95'])} | "
                    f"{_f(c['effect_native'], 0)} | {pm} | {_pl(c)} | {_status(c) if k != 'no_event' else 'baseline complement'} |")
    return "\n".join(rows)


def regime_table(d: dict, cells) -> str:
    rows = ["| cell | regime | sessions in regime | condition sessions | effect σ vs regime B0 | 95% CI |",
            "|---|---|---|---|---|---|"]
    for name, c in cells:
        for reg, v in c.get("post_hoc_within_regime", {}).items():
            rows.append(f"| {name} | {reg} | {v['n_regime']} | {v['n_condition']} | {_f(v['effect_sigma'])} | {_ci(v['ci95'])} |")
    return "\n".join(rows)


def year_table(d: dict, cells) -> str:
    years = sorted({y for _, c in cells for y in c["by_year"]})
    rows = ["| cell | " + " | ".join(f"{y} (n, σ)" for y in years) + " |", "|---|" + "---|" * len(years)]
    for name, c in cells:
        rows.append(f"| {name} | " + " | ".join(
            (f"{c['by_year'][y]['n']}, {c['by_year'][y]['z_mean']:+.2f}" if y in c["by_year"] else "—")
            for y in years) + " |")
    return "\n".join(rows)


def metrics_table(d: dict, cells) -> str:
    arm = d["arm"]
    cols = ["win_rate_trade", "win_rate_day", "expectancy_usd", "trades_per_session", "avg_winner_usd",
            "avg_loser_usd", "max_drawdown_usd"]
    cols += ["expectancy_bp"] if arm == "shares" else ["return_on_premium_mean", "median_premium_usd",
                                                        "median_entry_spread_pct", "underlying_twin_bp"]
    rows = ["| cell | " + " | ".join(cols) + " |", "|---|" + "---|" * len(cols)]
    for name, c in cells:
        m = c["metrics"]
        vals = []
        for k in cols:
            v = m.get(k)
            if v is None:
                vals.append("—")
            elif k in ("win_rate_trade", "win_rate_day", "return_on_premium_mean", "median_entry_spread_pct"):
                vals.append(f"{v:.1%}")
            elif k in ("trades_per_session",):
                vals.append(f"{v:.1f}")
            elif k == "expectancy_bp":
                vals.append(f"{v:+.2f} bp")
            elif k == "underlying_twin_bp":
                vals.append(f"{v:+.1f} bp")
            else:
                vals.append(f"{v:+,.0f}")
        rows.append(f"| {name} | " + " | ".join(vals) + " |")
    if arm == "options":
        rows.append("")
        rows.append("| cell | exit mix (stop / target / time / trail / eod) | shutdown exits excluded |")
        rows.append("|---|---|---|")
        for name, c in cells:
            e = c["metrics"].get("exit_mix", {})
            rows.append(f"| {name} | " + " / ".join(f"{e.get(k, 0):.0%}" for k in ("stop", "target", "time", "trail", "eod"))
                        + f" | {c['metrics'].get('exits_shutdown_excluded', 0)} |")
    return "\n".join(rows)


def q5_table(d: dict) -> str:
    rows = ["| category | pre trades | pre $/trade | post trades | post $/trade | sessions excluded (unknown time) |",
            "|---|---|---|---|---|---|"]
    for cat, v in d["q5"]["pre_post"].items():
        if v["pre_n"] + v["post_n"] == 0:
            continue
        rows.append(f"| {cat} | {v['pre_n']} | {_f(v['pre_mean_usd'], 2)} | {v['post_n']} | {_f(v['post_mean_usd'], 2)} | "
                    f"{v['sessions_excluded_unknown_time']} |")
    f = d["q5"]["first60"]
    rows.append("")
    rows.append(f"First-60-minute line (entries 09:30–10:30): scheduled_today {f['scheduled_today']['n_sessions']} sessions, "
                f"mean {_f(f['scheduled_today']['first60_net_mean_usd'], 0)} $/session; non-release "
                f"{f['non_release']['n_sessions']} sessions, {_f(f['non_release']['first60_net_mean_usd'], 0)}; "
                f"difference {_f(f.get('difference_usd'), 0)} {_ci(f.get('ci95'))}.")
    return "\n".join(rows)


def q7_table(d: dict, top: int = 10) -> str:
    q = d["q7"]
    cells = sorted([c for c in q["cells"].values() if c["tested"]], key=lambda c: c["p_boot"])[:top]
    rows = [f"Tested cells m = {q['m_tested']}; BH q=0.10 passes: {len(q['bh_pass'])}; of those with |effect| ≥ 0.25 σ: "
            f"{len(q['bh_pass_and_quarter_sigma'])}; shortlist: {q['shortlist'] or 'empty'}.", "",
            "| setup | category | sessions | setup trades | effect σ (arm σ) | 95% CI | p (boot) | BH | placebo rank | one-sided rate | fidelity |",
            "|---|---|---|---|---|---|---|---|---|---|---|"]
    for c in cells:
        osr = "—" if c["one_sided_rate"] is None else f"{c['one_sided_rate']:.0%}"
        rows.append(f"| {c['setup']} | {c['category']} | {c['n_sessions']} | {c['n_trades']} | {_f(c['effect_sigma'])} | "
                    f"{_ci(c['ci95'])} | {c['p_boot']:.4f} | {'pass' if c.get('bh_q10') else 'fail'} | "
                    f"{c['placebo']['rank']:.3f} | {osr} | {'LOW' if c['low_fidelity'] else 'ok'} |")
    return "\n".join(rows)


def q6_table(q6: dict) -> str:
    rows = ["| category | options effect σ | options CI | shares effect σ | shares CI | options − shares | divergence |",
            "|---|---|---|---|---|---|---|"]
    for cat, v in q6.items():
        rows.append(f"| {cat} | {_f(v['options_effect'])} | {_ci(v['options_ci'])} | {_f(v['shares_effect'])} | "
                    f"{_ci(v['shares_ci'])} | {_f(v['difference'])} | {'YES' if v['divergence'] else 'no'} |")
    return "\n".join(rows)


def header(d: dict) -> str:
    ex = d["excluded_no_0dte"]
    return (f"**{d['arm']} discovery** {d['window'][0]} → {d['window'][1]}: {d['n_sessions']} sessions, "
            f"{d['n_trades']:,} trades, hash {d['config_hashes']}, code `{d['repo_sha']}`, "
            f"σ(discovery) = ${d['sigma_discovery_usd']:,.0f}/day, B0 = ${d['b0']['net_mean']:+,.0f}/day "
            f"(median ${d['b0']['net_median']:+,.0f}), excluded no-0DTE sessions {ex['n']}. "
            f"Regime cuts {d['regime_cuts_med20_range_bp'][0]:.1f} / {d['regime_cuts_med20_range_bp'][1]:.1f} bp. "
            f"Seeds {d['seeds']}, bootstrap {d['n_boot']}, placebo {d['n_placebo']}.")


def render() -> str:
    parts = ["# news_arms discovery tables (generated by `trade_analysis.news_arms.report`; do not edit)", ""]
    both = {}
    for arm in ("shares", "options"):
        p = OUT / f"analysis_{arm}_discovery.json"
        if not p.exists():
            continue
        d = json.loads(p.read_text(encoding="utf-8"))
        both[arm] = d
        focus = [(k, d["q1"][k]) for k in ("FOMC", "CPI", "NFP", "PCE", "financial_stress")] + \
                [(k, d["q3"][k]) for k in ("scheduled", "unscheduled_surprise", "no_event")]
        parts += [f"## {arm}", "", header(d), "",
                  f"### Q1 / Q4 — by category vs B0 ({arm})", "", q1_table(d), "",
                  f"### Q1 sensitivity — single-category sessions only ({arm})", "", q1_table(d, single=True), "",
                  f"### Q3 — scheduled vs unscheduled surprise ({arm})", "", q3_table(d), "",
                  f"### Q2 — metric block, selected cells ({arm})", "", metrics_table(d, focus), "",
                  f"### By year, selected cells ({arm})", "", year_table(d, focus), "",
                  f"### Post hoc, NOT pre-registered — within trailing-range regime ({arm})", "",
                  regime_table(d, focus), "",
                  f"### Q5 — pre / post, descriptive ({arm})", "", q5_table(d), "",
                  f"### Q7 — setup × category, ten smallest p ({arm})", "", q7_table(d), ""]
    p6 = OUT / "analysis_q6_discovery.json"
    if p6.exists() and len(both) == 2:
        parts += ["## Q6 — options minus shares (each arm's own σ and window)", "",
                  q6_table(json.loads(p6.read_text(encoding="utf-8"))["q6"]), ""]
    return "\n".join(parts)


def main() -> int:
    out = OUT / "results_tables.md"
    out.write_text(render(), encoding="utf-8")
    print(f"written: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
