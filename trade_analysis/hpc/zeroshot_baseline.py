"""Zero-shot time-series foundation models against the same frame HAR is scored on.

WHY THIS RUNS BEFORE THE SWEEP
------------------------------
`trade_analysis/hpc/README.md` states the rule: *zero-shot foundation models are tried
FIRST, not last.* In 2024 the reflex here was to train a TFT. By 2026 a pretrained
forecaster is strong enough out of the box that training before checking one risks
reporting a trained model that a free forward pass would have beaten -- and you would
never discover that by running the sweep first, because the sweep never asks.

If a zero-shot model wins, section 9 of the pre-registration requires saying the problem
needed no bespoke model, not dressing the result up as a contribution.

WHAT IS FORECAST, AND WHY IT IS A SEVEN-STEP PROBLEM
-----------------------------------------------------
The frame is 5-minute bars, 60 origins per session, one bar apart. Thirty minutes is six
bars, so the obvious guess is that the target sits six steps ahead of the `rv_30m` series.
It is SEVEN, because `build_vrp_dataset` leaves a one-bar hole on purpose:

    hist = bars.index <  t   ->  rv_30m[t]     covers the returns ending at t-1
    fwd  = bars.index >  t   ->  rv_fwd_30[t]  covers the returns t+1 .. t+6

The return AT bar t belongs to neither window. That gap IS the no-lookahead guarantee --
the bar stamped t spans [t, t+5min) and so contains information from after the origin.
Hence

    rv_fwd_30[t]  ==  rv_30m[t + 7]

Verified rather than assumed: exact on 40,541 of 40,541 in-session pairs, relative error
0.0. At k=6 it matches 3 of 41,310 pairs with a median relative error of 14%. Shipping the
obvious guess would have scored the model on a window offset one bar from the one it was
given -- and the null that produced would have read as the model's failure.

NO LOOKAHEAD
------------
Context for origin i is `rv_30m[0 .. i]` INCLUSIVE. That is not a leak: `rv_30m[i]` is
backward-looking over the thirty minutes already elapsed at time i, so every value in the
context is realised and observable at the moment the forecast is made. The target is the
next seven steps, which never enters the context. `--audit` shifts the context one origin
into the future to prove the honest path did not already contain it.

THE BIAS TRAP -- AND WHY THIS FILE FEEDS VARIANCE, NOT LOG VARIANCE
--------------------------------------------------------------------
QLIKE scores a forecast of the conditional MEAN. Variance is strongly right-skewed, so
its mean sits well above its median, and a model's point forecast is usually the median.
Getting this wrong already turned "implied variance beats HAR, p=0.023" into p=0.583 in
this project, and both `har_forecast.py` and `har_baseline.py` carry an explicit exp(s2/2)
factor because of it.

The first version of THIS file walked into the same trap while documenting it. It fed log
variance to the model and then exponentiated the returned mean -- but exp(E[log v]) is the
geometric mean, which sits below the median, let alone the mean. Feeding logs makes a
correct answer HARDER, not easier: recovering E[v] from a log-space predictive
distribution needs the whole distribution, and the quantile grid spans only 0.1 to 0.9.
Integrating that misses the entire right tail, which for variance is where the mean lives.

So the default is to feed **variance in levels**. Chronos instance-normalises its input,
so scale is not a problem.

And then the pipeline's returned "mean" cannot be trusted either. Measured 2026-09-09 on
14,940 origins: chronos-bolt's `mean` and its own 0.5 quantile scored IDENTICALLY to six
decimal places -- same QLIKE, same RMSE, same Diebold-Mariano t. The library returns a
MEDIAN under the name mean. That is the same defect a third time, now arriving from a
dependency rather than from this repository.

What is scored instead is a lognormal fitted to the model's OWN returned quantiles:
log(q_p) = mu + sigma * z_p, hence E[v] = exp(mu + sigma^2 / 2). Lognormal is not an
arbitrary pick -- it is exactly what both HAR variants here already assume, so the neural
arm is not quietly handed a different error model than the benchmark it must beat.

The median is always scored alongside, and the run prints how often the library's "mean"
coincides with its median, so this particular artifact cannot hide a second time.

CALIBRATION, STATED BEFORE THE RERUN. A forecast uniformly k times too low scores
QLIKE = k - ln(k) - 1. The uncorrected run read 1.046465, which is k ~ 3.2. HAR's own
lognormal factor is 1.423, worth about 0.07 of QLIKE. So this correction is expected to
improve the number and NOT to close a gap of 0.62 -- which would make the loss a real
result rather than an artifact. Writing that down first is the point.
"""
from __future__ import annotations

import argparse
import sys
import time

import numpy as np
import pandas as pd

from .har_baseline import (TARGET, EPS, load, clean, split, qlike, dm_test, evaluate,
                           build_predictions)

CONTEXT = 512          # origins of history; ~8.5 sessions at 60 origins/session
HORIZON = 7            # NOT 6 -- see the module docstring. The extra step is the
                       # one-bar no-lookahead hole, and it is verified exact.
SERIES = "rv_30m"
# Chronos-Bolt is trained on exactly these levels. Asking for 0.01 or 0.99 would make it
# extrapolate past anything it saw in training, so the grid is left where the model is.
QUANTILES = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


def build_contexts(df: pd.DataFrame, idx: np.ndarray, audit: bool = False):
    """One context window per evaluation origin, ordered by (date, t).

    `df` must be the FULL frame, not the validation slice: an origin early in 2023 needs
    its history from 2022, and that history is legitimately available at the time.
    """
    s = df[SERIES].to_numpy(float)
    s = np.where(np.isfinite(s) & (s > 0), s, np.nan)
    out = []
    for i in idx:
        end = i + 1 + (1 if audit else 0)   # audit: one origin of genuine future
        lo = max(0, end - CONTEXT)
        w = s[lo:end]
        w = w[np.isfinite(w)]
        if len(w) < 32:
            w = np.array([np.nanmedian(s[:end]) if end else 1e-4])
        out.append(w)
    return out


def resolve_predictor(pipe):
    """Probe the pipeline's real signature on two toy series before spending the job.

    chronos-forecasting renamed `context` to `inputs` between its 1.x and 2.x lines. That
    cost a V100 allocation and forty-six seconds to discover at origin 0 of 14,940, with
    the model already loaded and the data already read. A probe on two fake series turns
    a library API change into a printed line.

    Returns (call, description, returns_mean).
    """
    import torch
    probe = [torch.linspace(0.01, 0.02, 64), torch.linspace(0.03, 0.05, 64)]
    tried = []
    candidates = [
        ("predict_quantiles(inputs=...)",
         lambda c: pipe.predict_quantiles(inputs=c, prediction_length=HORIZON,
                                          quantile_levels=QUANTILES)),
        ("predict_quantiles(positional)",
         lambda c: pipe.predict_quantiles(c, prediction_length=HORIZON,
                                          quantile_levels=QUANTILES)),
        ("predict_quantiles(context=...)",
         lambda c: pipe.predict_quantiles(context=c, prediction_length=HORIZON,
                                          quantile_levels=QUANTILES)),
    ]
    for name, fn in candidates:
        try:
            out = fn(probe)
        except (TypeError, AttributeError, NotImplementedError) as exc:
            tried.append(f"    {name}: {type(exc).__name__}: {exc}")
            continue
        has_mean = isinstance(out, (tuple, list)) and len(out) == 2
        q = out[0] if has_mean else out
        print(f"  resolved: {name} -> quantiles {tuple(q.shape)}"
              f"{', mean returned' if has_mean else ', NO mean returned'}", flush=True)
        return fn, name, has_mean
    print("  predict_quantiles did not resolve. Attempts:")
    for line in tried:
        print(line)
    raise SystemExit("no usable predict_quantiles signature -- check the chronos version")


def _quantile_mean(q: np.ndarray) -> np.ndarray:
    """Mean integrated over the quantile function. BIASED LOW -- only spans 0.1..0.9."""
    lvl = np.array([QUANTILES[0]] + QUANTILES + [QUANTILES[-1]])
    padded = np.pad(q, ((0, 0), (1, 1)), mode="edge")
    return np.trapezoid(padded, lvl, axis=1) / (lvl[-1] - lvl[0])


def _norm_ppf(p: np.ndarray) -> np.ndarray:
    """Inverse standard normal CDF. Acklam's rational approximation, |err| < 1.15e-9.

    Written out rather than imported so this file does not acquire a scipy dependency
    for one call -- and because dm_test already had a scipy-absent path that turned out
    to be broken, which is a poor advertisement for assuming the import.
    """
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00]
    p = np.asarray(p, dtype=float)
    out = np.empty_like(p)
    lo, hi = p < 0.02425, p > 1 - 0.02425
    mid = ~(lo | hi)
    q = np.sqrt(-2 * np.log(np.where(lo, p, 0.5)))
    out = np.where(lo, (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5])
                   / ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1), out)
    q = np.sqrt(-2 * np.log(np.where(hi, 1 - p, 0.5)))
    out = np.where(hi, -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5])
                   / ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1), out)
    q = np.where(mid, p, 0.5) - 0.5
    r = q * q
    out = np.where(mid, (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q
                   / (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1), out)
    return out


_Z = _norm_ppf(np.array(QUANTILES))


def _lognormal_mean(q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """E[v] and sigma, from a lognormal fitted to the model's OWN returned quantiles.

    WHY THIS EXISTS. chronos-bolt's `predict_quantiles` returns (quantiles, mean) and
    that second value is its 0.5 quantile -- measured 2026-09-09, the "mean" and the
    median scored identically to six decimal places on 14,940 origins, RMSE and DM
    t-statistic included. The library calls a median a mean.

    QLIKE scores a forecast of the MEAN, and for a right-skewed variable the median sits
    well below it. Scoring the median against QLIKE is the same defect that has now bitten
    this project three times: once in har_baseline (my exp(X@beta) with no exp(s2/2)),
    once in the first draft of this file (exp of a mean of logs), and now arriving from
    a dependency.

    THE FIT. If v is lognormal then log(q_p) = mu + sigma*z_p, with z_p the standard
    normal quantile. So regressing the log of the returned quantiles on z recovers mu
    and sigma, and E[v] = exp(mu + sigma^2/2) -- the same correction har_forecast.py and
    har_baseline.py already apply to a regression residual, applied instead to the
    model's own predictive distribution.

    Lognormal is not an arbitrary choice here: it is the distribution both HAR models in
    this project already assume, so using it keeps the comparison on one footing rather
    than handing the neural arm a different error model.

    z is symmetric about zero for a symmetric level grid, so mean(z) = 0 and the
    intercept is simply the mean of the logs.
    """
    y = np.log(np.maximum(q, EPS))
    mu = y.mean(axis=1)
    sigma = np.maximum((y * _Z).sum(axis=1) / (_Z ** 2).sum(), 0.0)
    return np.exp(mu + sigma ** 2 / 2.0), sigma


def _predict(pipe, contexts, batch: int, log_space: bool):
    """Returns (mean, median) forecasts of the target, in VARIANCE levels.

    In levels mode the pipeline's own mean is already E[v], which is what QLIKE scores.
    In log mode the quantiles are transformed with exp -- valid, because quantiles are
    equivariant under a monotone transform -- and the mean is integrated over them and
    reported as biased low, rather than faked with exp(mean of logs).
    """
    import torch
    call, _, has_mean = resolve_predictor(pipe)
    lib, med, lnm, sig = [], [], [], []
    imed = QUANTILES.index(0.5)
    t0 = time.time()
    for b0 in range(0, len(contexts), batch):
        chunk = contexts[b0:b0 + batch]
        tens = [torch.tensor(np.log(np.maximum(c, EPS)) if log_space else c,
                             dtype=torch.float32) for c in chunk]
        out = call(tens)
        # (batch, horizon, n_quantiles). The final step is the window the target is
        # measured over; the six before it are the no-lookahead hole plus the window's
        # own interior, and are not scored.
        q = (out[0] if has_mean else out)[:, HORIZON - 1, :].float().cpu().numpy()
        if log_space:
            q = np.exp(q)                    # exact: quantiles survive a monotone map
        m_ln, s = _lognormal_mean(q)
        lnm.append(m_ln)
        sig.append(s)
        med.append(q[:, imed])
        if has_mean:
            v = out[1][:, HORIZON - 1].float().cpu().numpy()
            lib.append(np.exp(v) if log_space else v)
        done = min(b0 + batch, len(contexts))
        if b0 % (batch * 10) == 0 or done == len(contexts):
            print(f"  {done}/{len(contexts)} origins  ({time.time()-t0:.0f}s)", flush=True)
    cat = lambda xs: np.maximum(np.concatenate(xs), EPS)
    return {"median": cat(med), "mean_lognormal": cat(lnm),
            "library_mean": cat(lib) if lib else None,
            "sigma": np.concatenate(sig)}


def run(data_dir: str, model_id: str, batch: int, log_space: bool,
        audit: bool, block: str) -> int:
    import torch
    from chronos import BaseChronosPipeline

    # SAME cleaning as the baseline, from the baseline's own function. Without it this
    # arm would score on rows HAR never saw, and the two QLIKE numbers would not be
    # comparable -- which is the sort of difference that gets read as a model result.
    df = clean(load(data_dir))
    blocks = split(df)
    test, train = blocks[block], blocks["train"]
    if test.empty:
        print(f"block '{block}' is empty")
        return 2

    print(f"frame: {len(df):,} origins over {df['date'].nunique()} sessions")
    print(f"scoring on '{block}': {len(test):,} origins / {test['date'].nunique()} sessions")
    print(f"model: {model_id}   context={CONTEXT}  horizon={HORIZON}  input="
          f"{'LOG variance (mean tail-truncated)' if log_space else 'variance levels'}")
    if audit:
        print("AUDIT: context shifted one origin into the future ON PURPOSE")

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {dev}")
    pipe = BaseChronosPipeline.from_pretrained(
        model_id, device_map=dev,
        torch_dtype=torch.bfloat16 if dev == "cuda" else torch.float32)

    contexts = build_contexts(df, test.index.to_numpy(), audit=audit)
    fc = _predict(pipe, contexts, batch, log_space)

    preds = build_predictions(train, test)
    tag = model_id.split("/")[-1]
    preds[f"zeroshot::{tag}::median"] = fc["median"]
    preds[f"zeroshot::{tag}::mean_lognormal"] = fc["mean_lognormal"]

    # SAY IT OUT LOUD when the library's "mean" is really its median. On 2026-09-09 the
    # two scored identically to six decimals on 14,940 origins and the run reported a
    # confident-looking loss to HAR that was substantially this artifact.
    lib = fc["library_mean"]
    if lib is not None:
        same = float(np.mean(np.isclose(lib, fc["median"], rtol=1e-6)))
        print(f"\n  library 'mean' equals its own 0.5 quantile on {same:.1%} of origins"
              + ("  <-- it is a MEDIAN. Not scored as a mean." if same > 0.99 else ""))
        if same <= 0.99:
            preds[f"zeroshot::{tag}::mean_library"] = lib
    s = fc["sigma"]
    corr = np.exp(s ** 2 / 2.0)
    print(f"  predictive sigma: median {np.median(s):.3f}  -> lognormal mean/median "
          f"ratio {np.median(corr):.3f}x  (HAR's own factor is 1.423)")

    table = evaluate(test, preds)
    print(f"\n=== QLIKE on {block} (lower is better) ===")
    print(table.to_string(index=False))

    y = test[TARGET].to_numpy(float)
    dates = test["date"].to_numpy()
    ref = "HAR-RV"
    if ref not in preds:
        print(f"\n{ref} unavailable; no DM test")
        return 0
    base_loss = qlike(y, preds[ref])
    print(f"\n=== Diebold-Mariano vs {ref}, clustered by SESSION ===")
    print("negative mean_diff = the model beat HAR")
    rows = [{"model": n, **dm_test(qlike(y, p), base_loss, dates)}
            for n, p in preds.items() if n != ref]
    print(pd.DataFrame(rows).to_string(index=False))

    zs = float(table.loc[table["model"] == f"zeroshot::{tag}::mean_lognormal",
                         "QLIKE"].iloc[0])
    har = float(table.loc[table["model"] == ref, "QLIKE"].iloc[0])
    print(f"\nzero-shot mean {zs:.6f}  vs  {ref} {har:.6f}  -> "
          f"{'zero-shot ahead' if zs < har else ref + ' ahead'} in level")
    print("Level is not the finding. The DM p-value above is; at ~250 sessions a gap of a"
          "\nfew thousandths of QLIKE is well inside session-to-session noise.")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--data", default="data/vrp")
    ap.add_argument("--model", default="amazon/chronos-bolt-base",
                    help="HF repo id. Cache it with fetch_zeroshot.sh on a COMPUTE node.")
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--block", default="validation", choices=["train", "validation"],
                    help="'heldout' is deliberately not offered; section 6 spends it once.")
    ap.add_argument("--log", action="store_true",
                    help="feed log variance. OFF by default: recovering E[v] from a "
                         "log-space predictive distribution needs tails the quantile "
                         "grid does not have. See the module docstring.")
    ap.add_argument("--audit", action="store_true")
    a = ap.parse_args(argv)
    return run(a.data, a.model, a.batch, a.log, a.audit, a.block)


if __name__ == "__main__":
    sys.exit(main())
