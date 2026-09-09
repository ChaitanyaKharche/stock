"""Train a variance forecaster on the VRP frame. GPU optional.

    python -m trade_analysis.hpc.train_vrp --data data/vrp --model mlp --epochs 60
    python -m trade_analysis.hpc.train_vrp --data data/vrp --model mlp --optimizer muon

Deliberately small. The point of this file is NOT to be a big model -- the pre-registration
says the question is whether ANYTHING beats HAR, and a 400k-parameter network is already
far more capacity than ~368 training sessions can support. The existing TFT checkpoint has
398,854 parameters and collapsed to its prior; more parameters was never the problem.

QLIKE IS THE LOSS, not MSE. Training on MSE and reporting QLIKE optimises one thing and
scores another. QLIKE is minimised at yhat == y and is robust to the noise in the realised
variance proxy, so it is what both the training objective and the evaluation use.

Predictions are made in LOG VARIANCE and exponentiated. This makes a negative variance
unrepresentable rather than merely unlikely, which matters because QLIKE is undefined for
one.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .har_baseline import (HAR_FEATURES, HAR_J_EXTRA, TARGET, dm_test, evaluate, load,
                           qlike, split, build_predictions)

EPS = 1e-12
EXTRA_FEATURES = [
    "rv_5m", "quarticity_30m", "minutes_since_open", "minutes_to_close", "dow",
    "iv_var_atm", "risk_reversal", "butterfly", "straddle_spread_bp", "vix_prev_close",
]


def feature_columns(df: pd.DataFrame) -> list[str]:
    cols = HAR_FEATURES + HAR_J_EXTRA + EXTRA_FEATURES
    return [c for c in cols if c in df.columns and df[c].notna().any()]


def to_xy(df: pd.DataFrame, cols: list[str]):
    X = df[cols].to_numpy(np.float32)
    # Log-transform the strictly-positive variance columns; leave calendar features alone.
    for i, c in enumerate(cols):
        if c.startswith(("rv_", "bipower", "quarticity", "iv_var")):
            X[:, i] = np.log(np.maximum(X[:, i], EPS))
    y = np.log(np.maximum(df[TARGET].to_numpy(np.float32), EPS))
    return X, y


def make_optimizer(params, name: str, lr: float):
    """AdamW, or Muon where available.

    Muon (orthogonalised updates via Newton-Schulz) landed in PyTorch 2.9 and reports
    roughly 2x compute efficiency against AdamW on compute-optimal LLM training. On a
    problem this small the wall-clock saving is irrelevant -- it is here because if the
    cluster's torch has it, there is no reason not to, and because a negative result is
    more credible when the optimiser was not the excuse.
    """
    import torch
    if name == "muon":
        try:
            from torch.optim import Muon                      # torch >= 2.9
            return Muon(params, lr=lr), "muon"
        except ImportError:
            pass
        try:
            from muon import Muon                             # standalone package
            return Muon(params, lr=lr), "muon(pkg)"
        except ImportError:
            print("  Muon unavailable in this torch; falling back to AdamW")
    return torch.optim.AdamW(params, lr=lr, weight_decay=1e-4), "adamw"


def train(args) -> int:
    import torch
    import torch.nn as nn

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    df = load(args.data)
    cols = feature_columns(df)
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=cols + [TARGET])
    parts = split(df)
    if parts["train"].empty or parts["validation"].empty:
        print("Not enough history for the frozen split.")
        return 1
    print(f"  device {dev} | {len(cols)} features | "
          f"train {len(parts['train'])} / val {len(parts['validation'])} origins")

    Xtr, ytr = to_xy(parts["train"], cols)
    Xva, yva = to_xy(parts["validation"], cols)
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-8
    Xtr, Xva = (Xtr - mu) / sd, (Xva - mu) / sd

    model = nn.Sequential(
        nn.Linear(len(cols), args.width), nn.GELU(), nn.Dropout(args.dropout),
        nn.Linear(args.width, args.width), nn.GELU(), nn.Dropout(args.dropout),
        nn.Linear(args.width, 1),
    ).to(dev)
    n_params = sum(p.numel() for p in model.parameters())
    opt, opt_name = make_optimizer(model.parameters(), args.optimizer, args.lr)
    print(f"  {n_params:,} parameters | optimiser {opt_name}")

    Xtr_t = torch.tensor(Xtr, device=dev)
    ytr_t = torch.tensor(ytr, device=dev).unsqueeze(1)
    Xva_t = torch.tensor(Xva, device=dev)
    yva_np = parts["validation"][TARGET].to_numpy(float)

    def qlike_loss(pred_log, target_log):
        """QLIKE in log space: exp(t-p) - (t-p) - 1, minimised at p == t."""
        d = target_log - pred_log
        return (torch.exp(d.clamp(-10, 10)) - d - 1.0).mean()

    best, best_state, patience = np.inf, None, 0
    n = len(Xtr_t)
    for epoch in range(args.epochs):
        model.train()
        perm = torch.randperm(n, device=dev)
        for i in range(0, n, args.batch):
            idx = perm[i:i + args.batch]
            opt.zero_grad(set_to_none=True)
            qlike_loss(model(Xtr_t[idx]), ytr_t[idx]).backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            pred = np.exp(model(Xva_t).squeeze(1).cpu().numpy())
        score = float(np.mean(qlike(yva_np, pred)))
        if score < best - 1e-6:
            best, best_state, patience = score, {k: v.clone() for k, v in
                                                 model.state_dict().items()}, 0
        else:
            patience += 1
            if patience >= args.patience:
                print(f"  early stop at epoch {epoch} (no val gain for {patience})")
                break
        if epoch % 10 == 0:
            print(f"    epoch {epoch:>3}  val QLIKE {score:.5f}   best {best:.5f}")

    if best_state:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        nn_pred = np.exp(model(Xva_t).squeeze(1).cpu().numpy())

    preds = build_predictions(parts["train"], parts["validation"])
    preds[f"nn_{opt_name}"] = nn_pred
    table = evaluate(parts["validation"], preds)
    print(f"\n  VALIDATION (2023)\n{table.to_string(index=False)}")

    bench = min((m for m in preds if m.startswith("HAR")),
                key=lambda m: table.set_index("model").loc[m, "QLIKE"], default=None)
    result = {"features": cols, "n_params": n_params, "optimizer": opt_name,
              "val_qlike": best, "table": table.to_dict("records"), "seed": args.seed}
    if bench:
        dates = parts["validation"]["date"].to_numpy()
        r = dm_test(qlike(yva_np, nn_pred), qlike(yva_np, preds[bench]), dates)
        result["dm_vs_benchmark"] = {"benchmark": bench, **r}
        beat = r.get("mean_diff", 0) < 0 and (r.get("p") or 1) < 0.05
        print(f"\n  vs {bench}: mean dQLIKE {r.get('mean_diff', float('nan')):+.5f} "
              f"t={r.get('t', float('nan')):+.2f} p={r.get('p')}  "
              f"({r.get('n_sessions')} sessions)")
        print(f"  {'BEATS' if beat else 'DOES NOT BEAT'} the benchmark before Holm "
              f"correction across the 7-model family.")
        if not beat:
            print("  Per section 6, the 2024 held-out block stays untouched.")

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(result, indent=1, default=str),
                                  encoding="utf-8")
        print(f"\n  wrote {args.out}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--data", default="data/vrp")
    ap.add_argument("--model", default="mlp", choices=["mlp"])
    ap.add_argument("--optimizer", default="adamw", choices=["adamw", "muon"])
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch", type=int, default=4096)
    ap.add_argument("--width", type=int, default=64)
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--patience", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="")
    return train(ap.parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
