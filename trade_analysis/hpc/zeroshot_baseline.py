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
so scale is not a problem, and the pipeline's own returned mean is then already E[v] --
exactly what QLIKE scores, with no transform and no correction. `--log` is kept for
comparison and reports its mean estimate as what it is: tail-truncated and biased low.
Both the mean and the median forecast are always scored, so the gap between them is a
number in the output rather than an assumption in the code.
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
    """Mean of the predictive distribution, integrated over the quantile function.

    Only spans 0.1..0.9, so the tails are missing and this is BIASED LOW for a
    right-skewed variable. Used only where the pipeline hands back no mean of its own;
    the alternative -- exponentiating a mean of logs -- is biased low too and is not
    even an estimator of the right quantity.
    """
    lvl = np.array([QUANTILES[0]] + QUANTILES + [QUANTILES[-1]])
    padded = np.pad(q, ((0, 0), (1, 1)), mode="edge")
    return np.trapezoid(padded, lvl, axis=1) / (lvl[-1] - lvl[0])


def _predict(pipe, contexts, batch: int, log_space: bool):
    """Returns (mean, median) forecasts of the target, in VARIANCE levels.

    In levels mode the pipeline's own mean is already E[v], which is what QLIKE scores.
    In log mode the quantiles are transformed with exp -- valid, because quantiles are
    equivariant under a monotone transform -- and the mean is integrated over them and
    reported as biased low, rather than faked with exp(mean of logs).
    """
    import torch
    call, _, has_mean = resolve_predictor(pipe)
    means, medians = [], []
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
            m = _quantile_mean(q)
        elif has_mean:
            m = out[1][:, HORIZON - 1].float().cpu().numpy()
        else:
            m = _quantile_mean(q)
        means.append(m)
        medians.append(q[:, imed])
        done = min(b0 + batch, len(contexts))
        if b0 % (batch * 10) == 0 or done == len(contexts):
            print(f"  {done}/{len(contexts)} origins  ({time.time()-t0:.0f}s)", flush=True)
    return (np.maximum(np.concatenate(means), EPS),
            np.maximum(np.concatenate(medians), EPS))


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
    mean_hat, med_hat = _predict(pipe, contexts, batch, log_space)

    preds = build_predictions(train, test)
    tag = model_id.split("/")[-1]
    preds[f"zeroshot::{tag}::mean"] = mean_hat
    preds[f"zeroshot::{tag}::median"] = med_hat

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

    zs = float(table.loc[table["model"] == f"zeroshot::{tag}::mean", "QLIKE"].iloc[0])
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
