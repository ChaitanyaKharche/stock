"""Phase 0: is the IV-beats-HAR result carried by a handful of sessions?

    python -m trade_analysis.hpc.dm_concentration --data data/vrp
    python -m trade_analysis.hpc.dm_concentration --data data/vrp --json out.json

WHY THIS RUNS BEFORE ANYTHING ELSE ON THE CLUSTER
-------------------------------------------------
This was written to audit a headline that turns out to be SUPERSEDED, and the first thing
it did was prove that. Recorded widely (including in an agent memory file) as:

    implied_variance QLIKE 0.411136 vs HAR-RV-J 0.541358, DM t=-2.30, p=0.0222

That 0.541358 is a HAR fitted WITHOUT the lognormal retransformation correction. See
`har_baseline.predict_har`: exp(X @ beta) is the conditional median, not the mean, and the
missing exp(s2/2) factor -- 1.423 here -- cost HAR 20% of its QLIKE. With the correction
in place the real table is

    implied_variance 0.411136   HAR-RV 0.430642   HAR-RV-J 0.430788

a 4.5% gap, not 24.1%, at t=-0.55 / p=0.583. **The option market does not beat HAR.**
The 24.1% figure should not be quoted again.

So the concentration question below is no longer "why is this effect so large" but the
more basic "what is actually in the differential", and the answer is worth having. If a
near-zero mean is a wash between two big opposing tails, then

  * the pre-registration's Holm correction across 7 models will not keep it,
  * Hansen's SPA / the Model Confidence Set certainly will not,
  * and no amount of GPU time on a neural arm is worth spending against a benchmark
    whose own advantage does not survive resampling.

So this is a gate, not a curiosity. It is also nearly free: no model is trained, nothing
is fitted that `har_baseline` does not already fit, and the 2024 held-out block is never
touched.

WHAT IT IS NOT
--------------
This does not test the periodicity hypothesis (that IV's edge is really a time-of-day
term HAR lacks). That needs HARP predictors and is a separate experiment. This only asks
whether the number we already have is robust enough to be worth explaining.

METHOD
------
Everything is imported from `har_baseline` -- the split, the cleaning, the HAR fit, the
QLIKE and the DM test. Nothing here re-implements them, so this file cannot drift away
from the result it is auditing. The unit of analysis is the SESSION throughout, matching
`dm_test`, because intraday origins are ~0.9 autocorrelated and treating them as
independent inflates n by roughly 300x.

Five questions, each with a number that can fail:

  1. reproduction     -- do we get the recorded QLIKE and t back?
  2. sign             -- on how many sessions does IV actually win? A real edge should
                         win often, not win enormously on a few.
  3. concentration    -- what share of the total differential lives in the top 1% / 5%
                         / 10% of sessions, ranked by how much they favour IV?
  4. leave-k-out      -- drop the k sessions most favourable to IV. How small is the k
                         that takes |t| under 1.96? If it is single digits, the result is
                         a few days of 2023, not a property of the option market.
  5. resampling       -- session-cluster bootstrap and a distribution-free sign test.
                         Both ignore the mean's magnitude and ask only whether the
                         direction survives.
"""
from __future__ import annotations

import argparse
import json
import math
import sys

import numpy as np
import pandas as pd

from .har_baseline import (TARGET, build_predictions, clean, dm_test, evaluate,
                           load, qlike, split)

T_CRIT = 1.96
BOOT = 10_000
SEED = 20260911


def session_diffs(val: pd.DataFrame, pred_a: np.ndarray,
                  pred_b: np.ndarray) -> pd.Series:
    """Session-mean QLIKE differential, A minus B. Negative = A beat B.

    Identical aggregation to har_baseline.dm_test, so the mean of this Series is exactly
    the `mean_diff` that test reports. Kept as a Series so we can interrogate the
    sessions individually, which dm_test does not expose.
    """
    y = val[TARGET].to_numpy(float)
    d = pd.DataFrame({"date": val["date"].to_numpy(),
                      "d": qlike(y, pred_a) - qlike(y, pred_b)})
    return d.groupby("date")["d"].mean()


def t_of(d: pd.Series) -> float:
    if len(d) < 3:
        return float("nan")
    se = d.std(ddof=1) / np.sqrt(len(d))
    return float(d.mean() / se) if se > 0 else float("nan")


def leave_k_out(d: pd.Series) -> dict:
    """Drop the k sessions most favourable to IV, one at a time, and watch t decay.

    Dropping the most extreme observations is not a fair significance test -- it is an
    INFLUENCE measure. The question it answers is "how many days of 2023 is this claim
    standing on", and the honest reading of a small answer is fragility, not refutation.
    """
    order = d.sort_values().index            # most negative (most pro-IV) first
    out = []
    for k in range(0, min(26, len(d) - 3)):
        kept = d.drop(order[:k]) if k else d
        out.append({"k": k, "t": t_of(kept), "mean": float(kept.mean()),
                    "n": int(len(kept))})
    breaks = next((r["k"] for r in out if not np.isnan(r["t"]) and abs(r["t"]) < T_CRIT),
                  None)
    return {"curve": out, "k_to_lose_significance": breaks}


def bootstrap(d: pd.Series, n: int = BOOT, seed: int = SEED) -> dict:
    """Resample SESSIONS with replacement. The session is the independent unit here."""
    rng = np.random.default_rng(seed)
    vals = d.to_numpy(float)
    idx = rng.integers(0, len(vals), size=(n, len(vals)))
    samples = vals[idx]
    means = samples.mean(axis=1)
    ses = samples.std(axis=1, ddof=1) / np.sqrt(len(vals))
    with np.errstate(divide="ignore", invalid="ignore"):
        ts = np.where(ses > 0, means / ses, np.nan)
    return {
        "mean_ci95": [float(np.percentile(means, 2.5)),
                      float(np.percentile(means, 97.5))],
        "frac_mean_negative": float((means < 0).mean()),
        "frac_significant": float(np.nanmean(ts < -T_CRIT)),
        "t_ci95": [float(np.nanpercentile(ts, 2.5)),
                   float(np.nanpercentile(ts, 97.5))],
    }


def sign_test(d: pd.Series) -> dict:
    """Distribution-free: ignore magnitudes, count sessions won.

    This is the check a concentrated mean cannot pass. If IV's advantage is real and
    broad it wins on most sessions; if it is three catastrophic HAR days it wins on about
    half and this comes back null.
    """
    wins = int((d < 0).sum())
    n = int((d != 0).sum())
    try:
        from scipy import stats
        p = float(stats.binomtest(wins, n, 0.5).pvalue)
    except Exception:                                            # noqa: BLE001
        # math.erf, NOT np.math.erf -- `np.math` was removed in numpy 2.0, which is the
        # exact hole har_baseline.dm_test documents in its own scipy fallback. Writing it
        # the broken way here would have put the bug back in the file auditing it.
        z = (wins - n / 2) / np.sqrt(n / 4)
        p = float(2 * (1 - 0.5 * (1 + math.erf(abs(z) / np.sqrt(2)))))
    return {"sessions_won": wins, "n": n, "win_rate": wins / n if n else float("nan"),
            "p": p}


def concentration(d: pd.Series) -> dict:
    """Share of the summed differential contributed by the most pro-IV sessions."""
    total = float(d.sum())
    ranked = d.sort_values().to_numpy(float)
    neg, pos = ranked[ranked < 0], ranked[ranked > 0]
    gross = float(np.abs(ranked).sum())
    # `share_of_total` is nonsense when the total is a near-zero residual of two large
    # opposing sides -- it produced 356% on the first run. Report the gross flow and the
    # cancellation ratio instead, and only quote the share when the net is a meaningful
    # fraction of the gross.
    out = {"total": total, "gross": gross,
           "sum_negative": float(neg.sum()), "sum_positive": float(pos.sum()),
           "n_negative": int(len(neg)), "n_positive": int(len(pos)),
           "net_over_gross": float(total / gross) if gross else float("nan")}
    for pct in (1, 5, 10, 25):
        k = max(1, int(round(len(ranked) * pct / 100)))
        out[f"top{pct}pct"] = {
            "k_sessions": k,
            "sum": float(ranked[:k].sum()),
            "share_of_gross": float(ranked[:k].sum() / gross) if gross else float("nan"),
        }
    return out


def run(data_dir: str, challenger: str = "implied_variance") -> dict:
    df = clean(load(data_dir))
    parts = split(df)
    for k, v in parts.items():
        print(f"  {k:<11} {len(v):>7} origins over {v['date'].nunique():>4} sessions")

    preds = build_predictions(parts["train"], parts["validation"])
    table = evaluate(parts["validation"], preds)
    print(f"\n  VALIDATION (2023) -- target {TARGET}\n{table.to_string(index=False)}")

    if challenger not in preds:
        raise SystemExit(f"{challenger} not in predictions: {sorted(preds)}")
    bench = min((m for m in preds if m.startswith("HAR")),
                key=lambda m: table.set_index("model").loc[m, "QLIKE"])
    print(f"\n  challenger={challenger}   benchmark={bench} (best HAR by QLIKE)")

    val = parts["validation"]
    d = session_diffs(val, preds[challenger], preds[bench])
    official = dm_test(qlike(val[TARGET].to_numpy(float), preds[challenger]),
                       qlike(val[TARGET].to_numpy(float), preds[bench]),
                       val["date"].to_numpy())

    print("\n" + "=" * 78)
    print("1. REPRODUCTION (must match har_baseline or nothing below is readable)")
    print("=" * 78)
    print(f"  dm_test : mean {official['mean_diff']:+.6f}  t={official['t']:+.4f}  "
          f"p={official['p']:.4f}  n={official['n_sessions']}")
    print(f"  here    : mean {d.mean():+.6f}  t={t_of(d):+.4f}  n={len(d)}")
    agree = abs(d.mean() - official["mean_diff"]) < 1e-12
    print(f"  {'[OK] identical aggregation' if agree else '[FAIL] DIVERGED -- stop'}")

    print("\n" + "=" * 78)
    print("2. SIGN -- does IV win OFTEN, or win BIG on a few?")
    print("=" * 78)
    st = sign_test(d)
    print(f"  IV lower QLIKE on {st['sessions_won']}/{st['n']} sessions "
          f"= {100 * st['win_rate']:.1f}%   sign-test p={st['p']:.4f}")
    q = d.quantile([0, .01, .05, .25, .5, .75, .95, .99, 1])
    print("  session-differential quantiles:")
    for k, v in q.items():
        print(f"    p{100 * k:>5.1f}  {v:+.6f}")

    print("\n" + "=" * 78)
    print("3. CONCENTRATION -- where does the mean live?")
    print("=" * 78)
    con = concentration(d)
    print(f"  NET differential   {con['total']:+.5f}   over {len(d)} sessions")
    print(f"  gross flow         {con['gross']:.5f}   "
          f"(pro-challenger {con['sum_negative']:+.5f} on {con['n_negative']} sessions, "
          f"pro-benchmark {con['sum_positive']:+.5f} on {con['n_positive']})")
    print(f"  net / gross        {con['net_over_gross']:+.4f}   "
          f"<- near zero means the two sides cancel and the NET is a residual")
    for pct in (1, 5, 10, 25):
        c = con[f"top{pct}pct"]
        print(f"    most pro-challenger {pct:>2}% ({c['k_sessions']:>3} sessions): "
              f"{c['sum']:+.5f} = {100 * c['share_of_gross']:>5.1f}% of GROSS flow")

    print("\n" + "=" * 78)
    print("4. LEAVE-K-OUT -- how many sessions is the claim standing on?")
    print("=" * 78)
    lko = leave_k_out(d)
    print("     k   n     mean          t")
    for r in lko["curve"]:
        if r["k"] <= 10 or r["k"] % 5 == 0:
            print(f"  {r['k']:>4} {r['n']:>4}  {r['mean']:+.6f}  {r['t']:+.3f}")
    kb = lko["k_to_lose_significance"]
    if kb == 0:
        print(f"\n  the challenger is NOT significant at k=0 (|t|={abs(t_of(d)):.2f} < "
              f"{T_CRIT}), so there is no significance to remove. Read the curve in the"
              f"\n  other direction: it shows how fast the BENCHMARK pulls ahead once the"
              f"\n  challenger's few extreme wins are dropped.")
        flip = next((r["k"] for r in lko["curve"] if r["t"] > T_CRIT), None)
        if flip is not None:
            print(f"  benchmark becomes significantly better after dropping k={flip} "
                  f"session(s) -- {100 * flip / len(d):.1f}% of the sample.")
    else:
        print(f"\n  |t| falls under {T_CRIT} after dropping "
              f"{'k=' + str(kb) if kb is not None else '>25'} session(s)"
              + (f" -- {100 * kb / len(d):.1f}% of the sample" if kb else ""))

    print("\n" + "=" * 78)
    print("5. RESAMPLING -- session-cluster bootstrap")
    print("=" * 78)
    bs = bootstrap(d)
    print(f"  mean 95% CI      [{bs['mean_ci95'][0]:+.6f}, {bs['mean_ci95'][1]:+.6f}]")
    print(f"  t    95% CI      [{bs['t_ci95'][0]:+.3f}, {bs['t_ci95'][1]:+.3f}]")
    print(f"  P(mean < 0)      {bs['frac_mean_negative']:.4f}")
    print(f"  P(t < -1.96)     {bs['frac_significant']:.4f}   "
          f"<- power of the claim under resampling")

    print("\n" + "=" * 78)
    print("VERDICT")
    print("=" * 78)
    fragile = []
    t0 = t_of(d)
    if abs(t0) < T_CRIT:
        fragile.append(f"the challenger is not significant at all (t={t0:+.2f}, "
                       f"p={official['p']:.3f}) -- there is no effect here to audit")
    if kb is not None and kb > 0 and kb <= 10:
        fragile.append(f"dropping {kb} of {len(d)} sessions removes significance")
    if st["p"] > 0.05:
        fragile.append(f"the sign test is null (p={st['p']:.3f}): the challenger wins "
                       f"{100 * st['win_rate']:.0f}% of sessions, not reliably more than half")
    if bs["frac_significant"] < 0.5:
        fragile.append(f"only {100 * bs['frac_significant']:.0f}% of bootstrap resamples "
                       f"reach t < -1.96")
    if abs(con["net_over_gross"]) < 0.25:
        fragile.append(f"net/gross is {con['net_over_gross']:+.3f}: the two sides very "
                       f"nearly cancel, so the mean is a residual of opposing tails")
    if st["p"] < 0.05 and st["win_rate"] < 0.5:
        fragile.append(f"the sign test is significant IN THE BENCHMARK'S FAVOUR -- the "
                       f"challenger wins only {100 * st['win_rate']:.1f}% of sessions "
                       f"(p={st['p']:.4f})")

    if fragile:
        print("  FRAGILE. The IV advantage is not robust on this sample:")
        for f in fragile:
            print(f"    - {f}")
        print("\n  Consequence: it will not survive Holm across 7 models, and it will not")
        print("  survive SPA/MCS. Do NOT spend cluster time on a neural arm judged")
        print("  against it, and do NOT touch the 2024 block on the strength of it.")
    else:
        print("  ROBUST on every check here. The IV advantage is broad, not carried by a")
        print("  few sessions, and survives session-cluster resampling. The periodicity")
        print("  question (is this a time-of-day term HAR lacks?) is then the next test,")
        print("  and it is the one that decides whether the neural arm has any room.")

    return {"official": official, "reproduced_mean": float(d.mean()),
            "reproduced_t": t_of(d), "n_sessions": int(len(d)),
            "challenger": challenger, "benchmark": bench,
            "sign_test": st, "concentration": con,
            "leave_k_out": {k: v for k, v in lko.items()},
            "bootstrap": bs, "fragile": fragile,
            "quantiles": {str(k): float(v) for k, v in q.items()}}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data", default="data/vrp", help="directory of session parquets")
    ap.add_argument("--challenger", default="implied_variance")
    ap.add_argument("--json", default=None, help="also write the full result here")
    args = ap.parse_args(argv)
    res = run(args.data, args.challenger)
    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(res, fh, indent=2, default=str)
        print(f"\n  wrote {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
