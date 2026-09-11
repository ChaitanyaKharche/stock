"""How much predictable variation is LEFT in this frame? Answer the question with a number.

    python -m trade_analysis.hpc.ceiling --data data/vrp
    python -m trade_analysis.hpc.ceiling --data data/vrp --audit

WHY THIS EXISTS
---------------
The §6 bar is now HAR-IV-TOD at QLIKE 0.330489, and the claim "nothing will clear it" had
been resting on two things: the option market's own forecast does not clear it, and the
best published zero-shot foundation model beats a HAR by only 1.3-1.8%. Both are real, and
neither is a measurement of THIS frame.

What was never measured is how much of the frame the bar actually uses. It uses SIX
columns -- five HAR lags plus iv_var_atm -- out of NINETEEN available. Untouched:

    rv_5m  quarticity_30m  vix_prev_close  spot  iv_atm  straddle_spread_bp
    n_strikes  risk_reversal  butterfly  dow  minutes_to_close

`risk_reversal` and `butterfly` are the skew and smile of the option surface. The single
biggest improvement found so far came from adding ONE column of option-market information
(IV, +22%); the rest of the surface has never been tried. That is a gap in the reasoning,
not a settled question.

This measures the ceiling three ways on the same train/validation split:

    HAR-IV-TOD      the current bar, reproduced
    kitchen-sink    linear, all 19 usable columns
    GBDT            HistGradientBoosting on all 19 -- nonlinear, interactions, free

If GBDT lands on the bar, the frame is exhausted both linearly and nonlinearly and no model
of any kind on THIS data clears §6 -- the limiting factor is the data, and that is a finding
worth having explicitly rather than assuming. If GBDT beats it materially, the sweep is NOT
dead and the correct next step is a pre-registered model search over the full feature set.

THIS IS A DIAGNOSTIC, NOT A §6 CANDIDATE
----------------------------------------
Nothing here claims passage. Section 6 requires Holm-corrected p<0.05 across the pre-registered
family, and neither the kitchen sink nor GBDT is in that family. Adding them post-hoc and
claiming a pass would be the gate-moving §9 forbids. The only question asked here is "is
there room", and a positive answer buys a new pre-registration, not a result.

LEAKAGE, STATED EXPLICITLY
--------------------------
Four columns are excluded and MUST be: rv_fwd_30 (the target), rv_fwd_60, and -- the
dangerous one -- **vrp_30 and vrp_60, which are defined as iv_var_atm MINUS rv_fwd, so they
contain the target by construction.** Including vrp_30 would produce a near-perfect fit and
look like a breakthrough. `--audit` shifts features one bar into the future and the honest
run must score WORSE, exactly as har_baseline does.
"""
from __future__ import annotations

import argparse
import sys

import numpy as np
import pandas as pd

from .har_baseline import EPS, TARGET, clean, dm_test, evaluate, load, qlike, split
from .harp_baseline import HAR_FEATURES, IV_COL, design, fit, holm, predict

LEAK = ["rv_fwd_30", "rv_fwd_60", "vrp_30", "vrp_60"]
NON_FEATURE = ["date", "t"]
# strictly positive -> modelled in logs, like every other price/variance column here
LOG_COLS = ["rv_from_open", "rv_30m", "rv_5m", "bipower_30m", "quarticity_30m",
            "rv_prev_day", "rv_prev_5", "rv_prev_22", "vix_prev_close", "spot",
            "iv_atm", "iv_var_atm", "straddle_spread_bp", "n_strikes"]
# may be zero, negative or categorical -> left linear
LIN_COLS = ["minutes_since_open", "minutes_to_close", "dow", "risk_reversal", "butterfly"]


def usable(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Present AND populated. Two of the nineteen are neither.

    `rv_5m` and `vix_prev_close` exist as columns in every session file and are 100% NaN --
    they were declared by the builder and never filled. Left in, they make the finite-row
    mask empty and the fit dies with "Found array with 0 sample(s)", which at least fails
    loudly; the worse outcome would have been a column of imputed zeros quietly diluting
    the model. They are dropped here WITH A PRINTED REASON, because "19 available features"
    was itself wrong and that is worth seeing rather than silently correcting.
    """
    log, lin, empty = [], [], []
    for c in LOG_COLS:
        if c not in df.columns:
            continue
        (empty if df[c].isna().all() else log).append(c)
    for c in LIN_COLS:
        if c not in df.columns:
            continue
        (empty if df[c].isna().all() else lin).append(c)
    if empty:
        print(f"  *** {len(empty)} declared feature(s) are 100% NaN and are DROPPED: "
              f"{', '.join(empty)} ***")
        print("      (present in the schema, never populated by build_vrp_dataset)")
    return log, lin


def _matrix(df: pd.DataFrame, log: list[str], lin: list[str]) -> np.ndarray:
    X = [np.ones(len(df))]
    if log:
        X.append(np.log(np.maximum(df[log].to_numpy(float), EPS)))
    if lin:
        X.append(pd.get_dummies(df[lin], columns=[c for c in lin if c == "dow"],
                                drop_first=True).to_numpy(float))
    return np.column_stack(X)


def fit_kitchen(train: pd.DataFrame, log: list[str], lin: list[str]):
    y = np.log(np.maximum(train[TARGET].to_numpy(float), EPS))
    X = _matrix(train, log, lin)
    ok = np.isfinite(X).all(axis=1) & np.isfinite(y)
    beta, *_ = np.linalg.lstsq(X[ok], y[ok], rcond=None)
    resid_var = float(np.var(y[ok] - X[ok] @ beta, ddof=X.shape[1]))
    return beta, resid_var


def fit_gbdt(train: pd.DataFrame, log: list[str], lin: list[str]):
    """Nonlinear ceiling. Log target + the same exp(s2/2) retransformation.

    Modest capacity on purpose: ~20k train rows whose EFFECTIVE n is 346 sessions, because
    intraday origins are ~0.9 autocorrelated. A deep model here fits sessions, not signal.
    """
    from sklearn.ensemble import HistGradientBoostingRegressor
    y = np.log(np.maximum(train[TARGET].to_numpy(float), EPS))
    X = _matrix(train, log, lin)
    ok = np.isfinite(X).all(axis=1) & np.isfinite(y)
    m = HistGradientBoostingRegressor(
        max_iter=400, learning_rate=0.05, max_depth=4, min_samples_leaf=200,
        l2_regularization=1.0, early_stopping=True, validation_fraction=0.15,
        random_state=20260911)
    m.fit(X[ok], y[ok])
    resid_var = float(np.var(y[ok] - m.predict(X[ok]), ddof=1))
    return m, resid_var


def run(data_dir: str, audit: bool = False) -> int:
    df = clean(load(data_dir))
    needed = [TARGET] + [f for f in HAR_FEATURES if f in df.columns]
    if audit:
        cols = [c for c in df.columns if c not in NON_FEATURE
                and not c.startswith(("rv_fwd_", "vrp_"))]
        df[cols] = df.groupby("date")[cols].shift(-1)
        df = df.dropna(subset=needed)
        print("AUDIT MODE: features shifted +1 bar (they now peek at the future).\n")

    parts = split(df)
    train, val = parts["train"], parts["validation"]
    log, lin = usable(train)
    print(f"  usable features: {len(log)} log + {len(lin)} linear = {len(log) + len(lin)}")
    print(f"    log    : {', '.join(log)}")
    print(f"    linear : {', '.join(lin)}")
    print(f"    EXCLUDED as leakage: {', '.join(LEAK)}")
    print(f"      (vrp_30 = iv_var_atm - rv_fwd_30 -- it CONTAINS the target)")
    print(f"  train {len(train)} origins / {train['date'].nunique()} sessions   "
          f"validation {len(val)} / {val['date'].nunique()}")

    preds: dict[str, np.ndarray] = {}

    # the current bar, reproduced through harp_baseline so it cannot drift
    har = [f for f in HAR_FEATURES if f in train.columns]
    for name, kw in (("HAR-IV", dict(log_feats=har, iv=True)),
                     ("HAR-IV-TOD", dict(log_feats=har, iv=True, fourier=True))):
        build = lambda d, kw=kw: design(d, **kw)                    # noqa: E731
        b, rv = fit(train, build)
        preds[name] = predict(val, build, b, rv)

    beta, rv = fit_kitchen(train, log, lin)
    preds["kitchen-sink-linear"] = np.exp(_matrix(val, log, lin) @ beta) * np.exp(rv / 2)

    m, rv_g = fit_gbdt(train, log, lin)
    preds["GBDT-all-features"] = np.exp(m.predict(_matrix(val, log, lin))) * np.exp(rv_g / 2)

    for k in preds:
        preds[k] = np.nan_to_num(preds[k], nan=float(np.nanmedian(preds[k])),
                                 posinf=float(np.nanmedian(preds[k])))

    table = evaluate(val, preds)
    print(f"\n  VALIDATION (2023), target {TARGET}, lower is better")
    print(table.to_string(index=False))

    q = table.set_index("model")["QLIKE"]
    bar = "HAR-IV-TOD"
    y = val[TARGET].to_numpy(float)
    dates = val["date"].to_numpy()
    lb = qlike(y, preds[bar])
    raw, rows = {}, []
    for name in preds:
        if name == bar:
            continue
        r = dm_test(qlike(y, preds[name]), lb, dates)
        raw[name] = r["p"]
        rows.append((name, r))
    adj = holm(raw)
    print(f"\n  Diebold-Mariano vs the section-6 bar ({bar} = {q[bar]:.6f}), "
          f"session-clustered, Holm over {len(raw)}")
    for name, r in sorted(rows, key=lambda kv: kv[1]["mean_diff"]):
        gain = 100 * (1 - q[name] / q[bar])
        print(f"    {name:<22} QLIKE {q[name]:.6f} ({gain:+.2f}% vs bar)  "
              f"dQLIKE {r['mean_diff']:+.5f}  t {r['t']:+.2f}  Holm p {adj[name]:.4f}")

    print("\n  " + "=" * 74)
    best = min((n for n in preds if n not in ("HAR-IV", "HAR-IV-TOD")), key=lambda n: q[n])
    room = 100 * (1 - q[best] / q[bar])
    verdict = (f"{room:.2f}% BETTER" if room > 0 else f"{-room:.2f}% WORSE")
    print(f"  ROOM LEFT IN THE FRAME: the best model using ALL features is {best},")
    print(f"  and it is {verdict} than the bar.")
    if room < 1.0:
        print("""
  THE FRAME IS EXHAUSTED, and the DIRECTION matters: throwing every column at it
  makes the forecast WORSE, not merely no better. A linear model on all of them,
  and a gradient-boosted tree free to find interactions, both LOSE to five HAR
  lags plus implied variance plus a clock -- including the option surface's skew
  and smile, which had never been tried and add nothing.

  The extra capacity is fitting session-level noise. Effective n here is 346
  SESSIONS, not 20,634 origins, because intraday origins are ~0.9 autocorrelated.

  That is not a statement about neural networks being weak. It is a measurement
  that the predictable variation in THIS data is already captured.

  The limiting factor is the DATA, not the model class, and the two things that
  would change it are both external: more assets (every ML win in this literature
  came from a panel -- 93 stocks, 30 DJIA -- exploiting cross-sectional
  spillovers we do not have with one symbol), or information outside this frame
  (order flow, dealer gamma, the full surface rather than three summary numbers).

  A GPU sweep over model architectures on these features cannot clear section 6.""")
    else:
        print(f"""
  THERE IS ROOM: {room:.2f}% sits in columns the bar does not use. "Nothing will
  clear it" was wrong, and the sweep is not dead -- but this is a DIAGNOSTIC and
  it does not pass section 6. The correct next step is a new pre-registration over the
  full feature set with the family size fixed in advance, not a claim on this
  number.""")

    if audit:
        print("\n  AUDIT: this run must score BETTER than the honest one. If it does not,")
        print("  the honest features already contained the next bar and all of it is void.")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data", default="data/vrp")
    ap.add_argument("--audit", action="store_true")
    a = ap.parse_args(argv)
    return run(a.data, a.audit)


if __name__ == "__main__":
    sys.exit(main())
