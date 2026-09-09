"""HAR benchmark, QLIKE loss, Diebold-Mariano test, and the lookahead audit.

    python -m trade_analysis.hpc.har_baseline --data data/vrp
    python -m trade_analysis.hpc.har_baseline --data data/vrp --audit

RUN THIS BEFORE ANY GPU TIME. It establishes the number that has to be beaten, on a CPU,
in minutes. A neural model that does not beat HAR has produced nothing however good its
absolute error looks, and finding that out after a cluster allocation is the expensive way
to learn it.

Three things here are deliberate and are the difference between this and a result that
evaporates on inspection:

QLIKE, NOT MSE. The realised-variance target is itself a noisy estimator of the latent
variance. MSE over a noisy proxy rewards a model that predicts the noise; QLIKE is robust
to it in the sense of Patton (2011), and is the standard in the volatility literature for
exactly that reason. RMSE is reported alongside so the comparison is not cherry-picked,
never instead.

CLUSTERED BY SESSION. Minute-level observations inside one day are ~0.9 correlated at
30-minute separation. Treating ~300 origins per session as independent inflates the
effective sample by ~300x and will manufacture significance out of nothing. The
Diebold-Mariano test here uses one loss differential per SESSION.

THE AUDIT, AND HOW TO READ IT -- WHICH IS THE OPPOSITE OF THE OBVIOUS GUESS.
`--audit` DELIBERATELY INJECTS a lookahead: every feature is shifted one minute into the
future, so the model sees information it could not have had. Therefore:

    audit BETTER than honest  ->  the honest run did NOT already contain that minute.
                                  This is the PASS. The gap measures what one minute of
                                  future information is worth on this problem.
    audit SAME as honest      ->  the minute was ALREADY inside the honest features.
                                  That is the leak signature, and it is the alarming case.

Stated the wrong way round on first writing, which would have condemned a clean pipeline.
The gap is also a calibration worth keeping: if one minute of genuine lookahead buys ~1.4%
QLIKE here, a model claiming a far larger margin over HAR is claiming something bigger than
cheating with a minute of the future, and should be disbelieved until re-audited.

This project lost an entire result set to a 1-minute lookahead that produced 95.7% of a
measured edge. The audit costs one extra CPU run.
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd

HAR_FEATURES = ["rv_30m", "rv_from_open", "rv_prev_day", "rv_prev_5", "rv_prev_22"]
HAR_J_EXTRA = ["bipower_30m"]
TARGET = "rv_fwd_30"
EPS = 1e-12


def load(data_dir: str) -> pd.DataFrame:
    files = sorted(glob.glob(str(Path(data_dir) / "*.parquet")))
    if not files:
        raise SystemExit(f"no parquet files in {data_dir} -- run build_vrp_dataset first")
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    df["date"] = pd.to_datetime(df["date"]).dt.date
    return df.sort_values(["date", "t"]).reset_index(drop=True)


def split(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """The temporal split frozen in research/vrp_preregistration.md section 1."""
    d = pd.to_datetime(df["date"])
    return {
        "train": df[d < "2023-01-01"],
        "validation": df[(d >= "2023-01-01") & (d < "2024-01-01")],
        "heldout": df[d >= "2024-01-01"],
    }


def qlike(y: np.ndarray, yhat: np.ndarray) -> np.ndarray:
    """Per-observation QLIKE. Lower is better; minimised at yhat == y."""
    y = np.maximum(y, EPS)
    yhat = np.maximum(yhat, EPS)
    return y / yhat - np.log(y / yhat) - 1.0


def fit_har(train: pd.DataFrame, feats: list[str]) -> np.ndarray:
    """OLS in log space.

    Variance is right-skewed and strictly positive; fitting levels lets a handful of
    high-vol sessions dominate the normal equations, and lets the model predict a negative
    variance, which QLIKE cannot even score.
    """
    sub = train[feats + [TARGET]].replace([np.inf, -np.inf], np.nan).dropna()
    X = np.log(np.maximum(sub[feats].to_numpy(float), EPS))
    X = np.column_stack([np.ones(len(X)), X])
    y = np.log(np.maximum(sub[TARGET].to_numpy(float), EPS))
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    return beta


def predict_har(df: pd.DataFrame, feats: list[str], beta: np.ndarray) -> np.ndarray:
    X = np.log(np.maximum(df[feats].to_numpy(float), EPS))
    X = np.column_stack([np.ones(len(X)), X])
    return np.exp(X @ beta)


def dm_test(loss_a: np.ndarray, loss_b: np.ndarray, dates: np.ndarray) -> dict:
    """Diebold-Mariano on session-mean loss differentials.

    One number per session, not per minute. Negative `mean_diff` means A beat B.
    """
    d = pd.DataFrame({"date": dates, "d": loss_a - loss_b}).groupby("date")["d"].mean()
    n = len(d)
    if n < 8:
        return {"n_sessions": n, "mean_diff": float(d.mean()) if n else np.nan,
                "t": np.nan, "p": np.nan,
                "note": "too few sessions for a meaningful test"}
    se = d.std(ddof=1) / np.sqrt(n)
    t = float(d.mean() / se) if se > 0 else np.nan
    try:
        from scipy import stats
        p = float(2 * (1 - stats.t.cdf(abs(t), df=n - 1)))
    except ImportError:                                      # scipy absent on some nodes
        p = float(2 * (1 - 0.5 * (1 + np.math.erf(abs(t) / np.sqrt(2)))))
    return {"n_sessions": int(n), "mean_diff": float(d.mean()), "t": t, "p": p}


def evaluate(df: pd.DataFrame, preds: dict[str, np.ndarray]) -> pd.DataFrame:
    y = df[TARGET].to_numpy(float)
    rows = []
    for name, yhat in preds.items():
        ql, se = qlike(y, yhat), (y - yhat) ** 2
        rows.append({"model": name, "QLIKE": float(np.mean(ql)),
                     "RMSE": float(np.sqrt(np.mean(se))),
                     "n": int(len(y))})
    return pd.DataFrame(rows).sort_values("QLIKE").reset_index(drop=True)


def build_predictions(train: pd.DataFrame, test: pd.DataFrame) -> dict[str, np.ndarray]:
    """The benchmark set. Every neural candidate must beat all of these."""
    preds: dict[str, np.ndarray] = {}
    # Naive persistence: tomorrow looks like the last 30 minutes. Beating HAR is the bar;
    # failing to beat THIS would mean the pipeline is broken.
    preds["persistence_rv30m"] = test["rv_30m"].to_numpy(float)
    if "iv_var_atm" in test:
        # The option market's own forecast, scaled to the horizon. If nothing beats this,
        # the market is already efficient at this horizon and that is the finding.
        preds["implied_variance"] = test["iv_var_atm"].to_numpy(float)
    for name, feats in (("HAR-RV", HAR_FEATURES),
                        ("HAR-RV-J", HAR_FEATURES + HAR_J_EXTRA)):
        usable = [f for f in feats if f in train.columns]
        if len(usable) < 2:
            continue
        beta = fit_har(train, usable)
        preds[name] = predict_har(test, usable, beta)
    return preds


def run(data_dir: str, audit: bool = False) -> int:
    df = load(data_dir)
    needed = [TARGET] + [f for f in HAR_FEATURES if f in df.columns]
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=needed)

    if audit:
        # Shift features one minute INTO THE FUTURE, within each session. See the module
        # docstring for how to read the result: a BETTER score here is the PASS, because
        # it proves the honest run did not already contain this minute.
        cols = [c for c in df.columns if c not in ("date", "t") and not c.startswith(
            ("rv_fwd_", "vrp_"))]
        df[cols] = df.groupby("date")[cols].shift(-1)
        df = df.dropna(subset=needed)
        print("AUDIT MODE: features shifted +1 minute (they now peek at the future).\n")

    parts = split(df)
    for k, v in parts.items():
        print(f"  {k:<11} {len(v):>7} origins over {v['date'].nunique():>4} sessions")
    if parts["train"].empty or parts["validation"].empty:
        print("\nNot enough history for the frozen split. Build more sessions first.")
        return 1

    preds = build_predictions(parts["train"], parts["validation"])
    table = evaluate(parts["validation"], preds)
    print(f"\n  VALIDATION (2023) -- target {TARGET}, lower QLIKE is better")
    print(table.to_string(index=False))

    best_bench = min((m for m in preds if m.startswith("HAR")),
                     key=lambda m: table.set_index("model").loc[m, "QLIKE"], default=None)
    if best_bench:
        y = parts["validation"][TARGET].to_numpy(float)
        dates = parts["validation"]["date"].to_numpy()
        print(f"\n  Diebold-Mariano vs {best_bench}, clustered by session:")
        for name in preds:
            if name == best_bench:
                continue
            r = dm_test(qlike(y, preds[name]), qlike(y, preds[best_bench]), dates)
            verdict = ("beats" if r.get("mean_diff", 0) < 0 else "loses to")
            p = r.get("p")
            print(f"    {name:<22} {verdict} {best_bench}: "
                  f"mean dQLIKE {r.get('mean_diff', float('nan')):+.5f}  "
                  f"t={r.get('t', float('nan')):+.2f}  "
                  f"p={'nan' if p is None or np.isnan(p) else f'{p:.4f}'}  "
                  f"(n={r.get('n_sessions')} sessions)")

    print("\n  " + "-" * 74)
    if audit:
        print("  Compare with the non-audit run:")
        print("    audit BETTER  -> PASS. The honest features did not contain this minute;")
        print("                     the gap is what one minute of the future is worth.")
        print("    audit SAME    -> FAIL. The minute was already inside the honest run.")
    else:
        print("  This is the bar. Section 6 of the pre-registration requires beating the")
        print("  best HAR variant on QLIKE with Holm-corrected p < 0.05 across 7 models")
        print("  BEFORE the 2024 block may be touched. Run --audit before believing it.")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--data", default="data/vrp")
    ap.add_argument("--audit", action="store_true",
                    help="shift features +1 minute; a score improvement means lookahead")
    args = ap.parse_args(argv)
    return run(args.data, args.audit)


if __name__ == "__main__":
    raise SystemExit(main())
