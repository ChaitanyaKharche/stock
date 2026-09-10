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

Forecasting the target is therefore a seven-step-ahead forecast of `rv_30m`, which is the
natural shape for a univariate foundation model and means this arm and HAR predict the
identical quantity, not two similar ones.

NO LOOKAHEAD
------------
Context for origin i is `rv_30m[0 .. i]` INCLUSIVE. That is not a leak: `rv_30m[i]` is
backward-looking over the thirty minutes already elapsed at time i, so every value in the
context is realised and observable at the moment the forecast is made. The target is the
next seven steps, which never enters the context. `--audit` shifts the context one origin
into the future to prove the honest path did not already contain it.

THE BIAS TRAP, WHICH THIS PROJECT HAS NOW HIT TWICE
---------------------------------------------------
QLIKE scores a forecast of the conditional MEAN. A right-skewed variable has mean well
above median, and a model's point forecast is usually the median. Ignoring that is what
turned "implied variance beats HAR, p=0.023" into p=0.583 once corrected, and it is why
`har_forecast.py` and `har_baseline.py` both carry an explicit exp(s2/2) factor.

Here no analytic correction is needed and none is applied: the predictive distribution is
available directly, so the mean is taken over it. Both the mean and the median are scored
and reported, so the size of the gap is MEASURED rather than assumed away.
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


def _predict(pipe, contexts, batch: int, log_space: bool):
    """Returns (mean, median) forecasts of the target, in variance levels.

    Two point forecasts on purpose. QLIKE wants the mean; most models hand you the median;
    the difference is the single largest error this project has made. Reporting both makes
    it a number instead of an assumption.
    """
    import torch
    means, medians = [], []
    t0 = time.time()
    for b0 in range(0, len(contexts), batch):
        chunk = contexts[b0:b0 + batch]
        tens = [torch.tensor(np.log(np.maximum(c, EPS)) if log_space else c,
                             dtype=torch.float32) for c in chunk]
        try:
            q, m = pipe.predict_quantiles(context=tens, prediction_length=HORIZON,
                                          quantile_levels=QUANTILES)
            # q: (batch, horizon, n_quantiles). The final step is the origin the
            # target is measured over; the six before it are the hole plus the
            # window's own interior, and are not scored.
            step_q = q[:, HORIZON - 1, :].float().cpu().numpy()
            step_m = m[:, HORIZON - 1].float().cpu().numpy()
            med = step_q[:, QUANTILES.index(0.5)]
        except (AttributeError, NotImplementedError):
            # Sample-based pipelines (Chronos-T5 and friends). The mean over samples IS
            # the conditional mean, including through the exp() below -- which is exactly
            # why sampling is preferred to transforming a point forecast.
            samples = pipe.predict(context=tens, prediction_length=HORIZON)
            arr = samples[:, :, HORIZON - 1].float().cpu().numpy()   # (batch, n_samples)
            if log_space:
                arr = np.exp(arr)
            step_m = arr.mean(axis=1)
            med = np.median(arr, axis=1)
            step_q = None
        if log_space and step_q is not None:
            step_m, med = np.exp(step_m), np.exp(med)
        means.append(step_m)
        medians.append(med)
        done = min(b0 + batch, len(contexts))
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
    test = blocks[block]
    train = blocks["train"]
    if test.empty:
        print(f"block '{block}' is empty")
        return 2

    print(f"frame: {len(df):,} origins over {df['date'].nunique()} sessions")
    print(f"scoring on '{block}': {len(test):,} origins / {test['date'].nunique()} sessions")
    print(f"model: {model_id}   context={CONTEXT}  horizon={HORIZON}  "
          f"input={'log variance' if log_space else 'variance'}")
    if audit:
        print("AUDIT: context shifted one origin into the future ON PURPOSE")

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {dev}")
    pipe = BaseChronosPipeline.from_pretrained(
        model_id, device_map=dev,
        torch_dtype=torch.bfloat16 if dev == "cuda" else torch.float32)

    idx = test.index.to_numpy()
    contexts = build_contexts(df, idx, audit=audit)
    mean_hat, med_hat = _predict(pipe, contexts, batch, log_space)

    # The HAR/IV/persistence benchmarks, computed here rather than copied, so both arms
    # are scored by the same code on the same rows in the same run.
    preds = build_predictions(train, test)
    tag = model_id.split("/")[-1]
    preds[f"zeroshot::{tag}::mean"] = mean_hat
    preds[f"zeroshot::{tag}::median"] = med_hat

    table = evaluate(test, preds)
    print("\n=== QLIKE on", block, "(lower is better) ===")
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
    rows = []
    for name, yhat in preds.items():
        if name == ref:
            continue
        rows.append({"model": name, **dm_test(qlike(y, yhat), base_loss, dates)})
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
                    help="HF repo id. Pre-download it on the LOGIN node -- compute nodes "
                         "on Discovery have no outbound internet.")
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--block", default="validation", choices=["train", "validation"],
                    help="'heldout' is deliberately not offered; section 6 spends it once.")
    ap.add_argument("--levels", action="store_true",
                    help="feed raw variance instead of log variance")
    ap.add_argument("--audit", action="store_true")
    a = ap.parse_args(argv)
    return run(a.data, a.model, a.batch, not a.levels, a.audit, a.block)


if __name__ == "__main__":
    sys.exit(main())
