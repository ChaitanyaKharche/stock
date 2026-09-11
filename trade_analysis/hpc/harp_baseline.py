"""#2 -- does HAR lack a diurnal term that implied variance has?

    python -m trade_analysis.hpc.harp_baseline --data data/vrp
    python -m trade_analysis.hpc.harp_baseline --data data/vrp --audit

THE QUESTION, AND WHY IT IS NOW CONFIRMATORY RATHER THAN DECISIVE
-----------------------------------------------------------------
Dumitru, Hizmeri & Izzeldin (*J. Banking and Finance* 170:107342, 2025) show that intraday
periodicity inflates the variance of RV and biases jump estimators, and that building HAR
predictors from periodicity-filtered returns buys **up to 7% on SPY** even at DAILY
horizons -- where the diurnal effect is at its smallest. At the 30-minute intraday origins
used here the deterministic time-of-day component is far larger.

This was originally queued to explain a 24.1% QLIKE gap in implied variance's favour. That
gap does not exist: it was an artifact of a HAR fitted without the lognormal
retransformation correction, and the real gap is 4.5% at p=0.583. See
`research/vrp_phase0_concentration.md`.

So the hypothesis inverts. The question is no longer "is IV's edge really a clock?" but
**"is HAR leaving free accuracy on the table, and how much further ahead of IV does it get
when it stops?"** A positive result here STRENGTHENS the negative on the variance arm. Its
real value is closing the "maybe HAR was just misspecified" objection permanently, so the
arm can be written up without that loose end.

Pre-committed before running, because a confirmatory test with a flexible reading is not a
test:

  * diurnal terms IMPROVE HAR  -> expected. HAR pulls further ahead of IV; the variance
    arm's negative is stronger and the objection is closed.
  * diurnal terms do NOTHING   -> also fine, and more interesting: the 5-minute RV
    features already absorb the U-shape, so there was never a clock to find.
  * diurnal terms make HAR WORSE than IV -> the only surprising branch, and it would mean
    the time-of-day encoding is overfitting the train block. Check the binned spec against
    the parsimonious Fourier one before believing it.

Nothing here may reopen the variance arm on its own. A better HAR is a better BENCHMARK,
not a tradeable result -- the arm is closed on the cost model (spread 1.84x the gross edge,
replicated four ways), which no forecast improvement touches.

WHAT IS TESTED
--------------
Three additions to the frozen HAR feature set, all fitted on TRAIN only:

  HAR-IV        + log(iv_var_atm). The benchmark the literature actually uses --
                Kambouroudis/McMillan/Tsakou (JFM 2021) find only HAR specs that include
                implied volatility enter the Model Confidence Set. Comparing IV-alone to
                HAR-alone, as the pre-registration does, is not the standard comparison.
  HAR-TOD       + 3 Fourier harmonics of time-of-day (6 parameters). Parsimonious, cannot
                fit a spike.
  HAR-TOD-bins  + one dummy per 5-minute slot (59 parameters). Maximally flexible; if this
                beats the Fourier spec by a lot it is fitting the train block, not a clock.
  HAR-IV-TOD    both, to separate "IV carries clock information" from "IV carries
                information beyond the clock".

TRUE HARP -- filtering the RETURNS rather than adding a regressor -- is a different and
bigger job: the frame on disk holds pre-computed RV, not returns, so it needs a rebuild
from the 1-minute archive via build_vrp_dataset. This file tests the FORECAST-side
hypothesis, which is the one that bears on the benchmark. If the diurnal terms move
anything here, the estimator-side rebuild becomes worth its cost; if they move nothing,
it does not.

MULTIPLE TESTING
----------------
Section 6 of the pre-registration requires Holm-corrected p < 0.05 across the model
family. Adding specifications adds tests, so Holm is applied here over every non-benchmark
model and reported beside the raw p. A raw p that survives and a Holm p that does not is
reported as NOT significant.
"""
from __future__ import annotations

import argparse
import sys

import numpy as np
import pandas as pd

from .har_baseline import (EPS, HAR_FEATURES, HAR_J_EXTRA, TARGET, clean, dm_test,
                           evaluate, load, qlike, split)

TOD_COL = "minutes_since_open"
IV_COL = "iv_var_atm"
N_HARMONICS = 3
SESSION_MINUTES = 390.0


# --------------------------------------------------------------------------- design matrix
def _tod_fourier(df: pd.DataFrame) -> np.ndarray:
    """sin/cos harmonics of time-of-day. Deterministic in the clock, so it cannot leak."""
    phase = 2.0 * np.pi * df[TOD_COL].to_numpy(float) / SESSION_MINUTES
    cols = []
    for k in range(1, N_HARMONICS + 1):
        cols += [np.sin(k * phase), np.cos(k * phase)]
    return np.column_stack(cols)


def _tod_bins(df: pd.DataFrame, levels: np.ndarray) -> np.ndarray:
    """One dummy per 5-minute slot, first level dropped.

    `levels` comes from TRAIN. A validation slot unseen in train gets an all-zero row,
    which is the right behaviour -- it falls back to the intercept rather than inventing a
    column, and it cannot silently reindex the matrix.
    """
    tod = df[TOD_COL].to_numpy(float)
    return np.column_stack([(tod == lv).astype(float) for lv in levels[1:]])


def design(df: pd.DataFrame, log_feats: list[str], *, iv: bool = False,
           fourier: bool = False, bins: np.ndarray | None = None) -> np.ndarray:
    X = [np.ones(len(df))]
    X.append(np.log(np.maximum(df[log_feats].to_numpy(float), EPS)))
    if iv:
        X.append(np.log(np.maximum(df[[IV_COL]].to_numpy(float), EPS)))
    if fourier:
        X.append(_tod_fourier(df))
    if bins is not None:
        X.append(_tod_bins(df, bins))
    return np.column_stack(X)


def fit(train: pd.DataFrame, build) -> tuple[np.ndarray, float]:
    """Log-space OLS. Returns (beta, resid_var) exactly like har_baseline.fit_har.

    resid_var comes back because the prediction needs it for the lognormal correction.
    Omitting that factor is what produced the 24.1% phantom this file was written to
    explain, so it is not optional and it is not applied anywhere else.
    """
    y = np.log(np.maximum(train[TARGET].to_numpy(float), EPS))
    X = build(train)
    ok = np.isfinite(X).all(axis=1) & np.isfinite(y)
    X, y = X[ok], y[ok]
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid_var = float(np.var(y - X @ beta, ddof=X.shape[1]))
    return beta, resid_var


def predict(df: pd.DataFrame, build, beta: np.ndarray, resid_var: float) -> np.ndarray:
    return np.exp(build(df) @ beta) * np.exp(resid_var / 2.0)


# ------------------------------------------------------------------------------ multiplicity
def holm(pvals: dict[str, float]) -> dict[str, float]:
    """Holm-Bonferroni step-down. Monotone, capped at 1."""
    items = sorted(pvals.items(), key=lambda kv: (np.inf if np.isnan(kv[1]) else kv[1]))
    m, out, running = len(items), {}, 0.0
    for i, (name, p) in enumerate(items):
        adj = min(1.0, (m - i) * p) if not np.isnan(p) else float("nan")
        running = max(running, adj) if not np.isnan(adj) else running
        out[name] = running
    return out


# ------------------------------------------------------------------------------------- run
def build_models(train: pd.DataFrame, test: pd.DataFrame) -> dict[str, np.ndarray]:
    har = [f for f in HAR_FEATURES if f in train.columns]
    harj = har + [f for f in HAR_J_EXTRA if f in train.columns]
    levels = np.sort(train[TOD_COL].unique())
    preds: dict[str, np.ndarray] = {}

    if IV_COL in test:
        preds["implied_variance"] = test[IV_COL].to_numpy(float)

    specs = {
        "HAR-RV": dict(log_feats=har),
        "HAR-RV-J": dict(log_feats=harj),
        "HAR-IV": dict(log_feats=har, iv=True),
        "HAR-TOD": dict(log_feats=har, fourier=True),
        "HAR-TOD-bins": dict(log_feats=har, bins=levels),
        "HAR-IV-TOD": dict(log_feats=har, iv=True, fourier=True),
    }
    for name, kw in specs.items():
        if kw.get("iv") and IV_COL not in train.columns:
            continue
        build = lambda d, kw=kw: design(d, **kw)          # noqa: E731
        beta, rv = fit(train, build)
        preds[name] = predict(test, build, beta, rv)
        preds[name] = np.nan_to_num(preds[name], nan=np.nanmedian(preds[name]),
                                    posinf=np.nanmedian(preds[name]))
    return preds


def run(data_dir: str, audit: bool = False) -> int:
    df = clean(load(data_dir))
    needed = [TARGET] + [f for f in HAR_FEATURES if f in df.columns]
    if audit:
        cols = [c for c in df.columns if c not in ("date", "t")
                and not c.startswith(("rv_fwd_", "vrp_"))]
        df[cols] = df.groupby("date")[cols].shift(-1)
        df = df.dropna(subset=needed)
        print("AUDIT MODE: features shifted +1 bar (they now peek at the future).\n")

    parts = split(df)
    for k, v in parts.items():
        print(f"  {k:<11} {len(v):>7} origins over {v['date'].nunique():>4} sessions")

    train, val = parts["train"], parts["validation"]
    if train.empty or val.empty:
        print("not enough history for the frozen split")
        return 1

    preds = build_models(train, val)
    table = evaluate(val, preds)
    print(f"\n  VALIDATION (2023) -- target {TARGET}, lower QLIKE is better")
    print(table.to_string(index=False))

    q = table.set_index("model")["QLIKE"]
    bench = q.drop(index=[m for m in q.index if not m.startswith("HAR")]).idxmin()
    print(f"\n  benchmark = {bench} (best HAR-family spec by QLIKE)")

    y = val[TARGET].to_numpy(float)
    dates = val["date"].to_numpy()
    lb = qlike(y, preds[bench])
    raw, rows = {}, []
    for name in preds:
        if name == bench:
            continue
        r = dm_test(qlike(y, preds[name]), lb, dates)
        raw[name] = r["p"]
        rows.append((name, r))
    adj = holm(raw)

    print(f"\n  Diebold-Mariano vs {bench}, clustered by session, Holm over {len(raw)} tests")
    print(f"  {'model':<20} {'dQLIKE':>10} {'t':>8} {'raw p':>8} {'Holm p':>8}  verdict")
    for name, r in sorted(rows, key=lambda kv: kv[1]["mean_diff"]):
        v = ("BEATS" if r["mean_diff"] < 0 and adj[name] < 0.05 else
             "beats (ns)" if r["mean_diff"] < 0 else
             "LOSES" if adj[name] < 0.05 else "loses (ns)")
        print(f"  {name:<20} {r['mean_diff']:>+10.5f} {r['t']:>+8.2f} "
              f"{r['p']:>8.4f} {adj[name]:>8.4f}  {v}")

    # --- the pre-committed reading -----------------------------------------------------
    print("\n  " + "-" * 74)
    iv_q = q.get("implied_variance", np.nan)
    base = q["HAR-RV"]
    print(f"  HAR-RV {base:.6f}   implied_variance {iv_q:.6f}   "
          f"gap {100 * (iv_q / base - 1):+.2f}%")
    for spec in ("HAR-IV", "HAR-TOD", "HAR-TOD-bins", "HAR-IV-TOD"):
        if spec in q:
            print(f"  {spec:<14} {q[spec]:.6f}   vs HAR-RV "
                  f"{100 * (q[spec] / base - 1):+.2f}%   vs IV "
                  f"{100 * (q[spec] / iv_q - 1):+.2f}%")

    tod_gain = 100 * (1 - min(q.get("HAR-TOD", np.inf),
                              q.get("HAR-TOD-bins", np.inf)) / base)
    print()
    if not np.isfinite(tod_gain):
        print("  no diurnal spec was fitted -- minutes_since_open missing from the frame")
    elif tod_gain < 0.5:
        print(f"  DIURNAL TERMS DO ESSENTIALLY NOTHING ({tod_gain:+.2f}%). The 5-minute RV")
        print("  features already absorb the intraday U-shape, so there was never a clock")
        print("  for IV to be secretly carrying. The 'maybe HAR is misspecified' objection")
        print("  is closed by measurement, and the estimator-side HARP rebuild is not")
        print("  worth its cost.")
    else:
        print(f"  diurnal terms buy {tod_gain:+.2f}% on HAR-RV. HAR pulls further ahead of")
        print("  IV, which STRENGTHENS the variance arm's negative. It does not reopen it:")
        print("  the arm is closed on the cost model, which no forecast gain touches.")
        fo, bi = q.get("HAR-TOD", np.nan), q.get("HAR-TOD-bins", np.nan)
        if np.isfinite(fo) and np.isfinite(bi) and bi < fo * 0.98:
            print(f"  CAUTION: the 59-parameter binned spec beats the 6-parameter Fourier")
            print(f"  one by {100 * (1 - bi / fo):.1f}% -- suspect train-block overfitting,")
            print("  not a sharper clock.")

    if audit:
        print("\n  AUDIT: this run must score BETTER than the honest one. If it does not,")
        print("  the honest features already contained the next bar and everything is void.")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data", default="data/vrp")
    ap.add_argument("--audit", action="store_true")
    args = ap.parse_args(argv)
    return run(args.data, args.audit)


if __name__ == "__main__":
    sys.exit(main())
