"""Hansen's SPA, studentized, with the max computed so it can be reduced across shards.

WHY THIS IS NOT `from arch.bootstrap import SPA`
------------------------------------------------
`arch` 8.0.0 is maintained and its `SPA` class is widely used, and its `studentize=True`
flag DOES NOTHING. Verified two ways on 2026-09-13 against the installed copy:

  * SOURCE: `self.studentize` is read at exactly one place in the compute path
    (multiple_comparison.py line 604) and it selects a STRING -- "bootstrap"/"asymptotic"/
    "none" -- that is stored in `self._info` for display. `_simulate_values()` computes
    `loss_diff_star.mean(0) - mean`, the raw recentred mean, and never divides by a
    variance. `_compute_variance()` is called, but its result feeds
    `_check_column_validity()` -- the sqrt(2 log log n) recentring threshold -- not the
    statistic.
  * EXPERIMENT: 40 trials on Hansen's own Example 4 design (one good model, k-1 junk models
    with 4x the standard deviation). p-values with `studentize=True` and `studentize=False`
    were IDENTICAL in 40 of 40, to zero tolerance.

So `arch.SPA` is White's Reality Check with Hansen's recentring. That is a fine test; it is
just not the one whose name it carries, and the difference is the entire reason to prefer
SPA here. Hansen's Table 2 shows RC losing an order of magnitude of power precisely when the
comparison set is dominated by poor models -- and an exhaustive parameter grid is the purest
possible example of a comparison set dominated by poor models. Roughly 99% of a million
cells are junk by construction.

WHAT THE LOSS DIFFERENTIALS ARE HERE
------------------------------------
The benchmark is NOT TRADING. Its loss is 0. A cell's loss is its negative P&L. So
`d_k = L_benchmark - L_k = pnl_k`, and the matrix of loss differentials is literally the
per-session P&L matrix. The null is `max_k E[d_k] <= 0`: no cell in the entire grid makes
money.

THE SHARD REDUCTION, WHICH IS WHAT MAKES A MILLION CELLS POSSIBLE
-----------------------------------------------------------------
The SPA statistic is a MAXIMUM over cells, and a maximum is associative. Storing the full
(3,815 sessions x 1,038,800 cells) matrix would be ~200 GB in float32 and is not worth
having. Instead each shard computes, for each of B bootstrap resamples, the max over ITS
OWN cells, and the collector takes the max across shards. Each shard stores a (B,) vector.

This is only valid if every shard bootstraps the SAME session indices. `boot_indices()`
derives them from a fixed seed and the session count alone, so shards never communicate and
still agree exactly. If a shard ever sees a different session list, the reduction is
silently wrong -- so `spa_shard_max` stores a hash of the session labels for the collector
to check.
"""
from __future__ import annotations

import hashlib

import numpy as np


def optimal_block(x: np.ndarray) -> float:
    """Politis-White block length WITH the Patton (2009) correction, via arch.

    Patton's correction is not optional: PW2004 printed `D_SB = 4g^2(0) + (2/pi)*int(...)`
    where the right quantity is `2g^2(0)`. Since `b_opt` scales as `D_SB^(-1/3)`, the
    uncorrected formula returns blocks that are too SHORT, which under-corrects for serial
    dependence and makes the test anti-conservative. arch implements the corrected form.

    arch's docstring advertises columns `b_sb`/`b_cb`; the DataFrame it actually returns is
    keyed `stationary`/`circular`. Indexing the documented names raises KeyError, so the
    positional column is used and the name is only checked.
    """
    from arch.bootstrap import optimal_block_length
    x = np.asarray(x, dtype=float).ravel()
    tab = optimal_block_length(x)
    col = "stationary" if "stationary" in tab.columns else tab.columns[0]
    return float(np.asarray(tab[col])[0])


def boot_indices(n: int, reps: int, block: float, seed: int = 20260913) -> np.ndarray:
    """Stationary-bootstrap (Politis-Romano 1994) session indices, shape (reps, n).

    Geometric block lengths with mean `block`, wrapping circularly. Derived from `seed` and
    `n` ONLY, so every shard independently generates the identical resamples -- which is the
    precondition for the cross-shard max reduction being the true grid-wide max.

    Vectorised. The textbook form is a double loop over (reps, n), which at 10,000 reps x
    3,815 sessions is 38 million Python iterations PER SHARD and would cost more than the
    backtest it is testing. The identity used instead: within a block, `idx` advances by one
    each step from the block's start, so

        idx[b, t] = (start_value_of_current_block + steps_since_block_started) mod n

    `np.maximum.accumulate` over the positions where a new block begins forward-fills which
    block each t belongs to, and the rest is arithmetic.
    """
    rng = np.random.default_rng(seed)
    p = 1.0 / max(block, 1.0)
    starts = rng.integers(0, n, size=(reps, n)).astype(np.int32)
    newblk = rng.random((reps, n)) < p
    newblk[:, 0] = True                                   # every path starts a block at t=0

    t = np.arange(n, dtype=np.int32)[None, :]
    last = np.maximum.accumulate(np.where(newblk, t, 0), axis=1)   # t of current block start
    anchor = np.take_along_axis(starts, last, axis=1)              # its drawn start value
    return ((anchor + (t - last)) % n).astype(np.int32)


def studentize(D: np.ndarray, idx: np.ndarray
               ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-cell mean and bootstrap standard error of the mean.

    `omega_k` is the standard deviation of the BOOTSTRAPPED means, which is the quantity
    Hansen studentizes by. Estimating it from the same resamples used for the null keeps the
    block-dependence structure consistent between the two.
    """
    mean = D.mean(axis=0)
    bm = np.empty((idx.shape[0], D.shape[1]), dtype=np.float64)
    for b in range(idx.shape[0]):
        bm[b] = D[idx[b]].mean(axis=0)
    omega = bm.std(axis=0, ddof=1)
    omega[omega <= 0] = np.inf          # a cell that never trades cannot be the max
    return mean, omega, bm


def spa_shard_max(D: np.ndarray, idx: np.ndarray, sessions: list[str]) -> dict:
    """Observed max and per-resample max over THIS shard's cells.

    Returns the three Hansen recentrings. `consistent` is the one to report; `lower` and
    `upper` bracket it and a large gap between them is itself a diagnostic that the result
    is sensitive to how borderline cells are treated.
    """
    n, k = D.shape
    mean, omega, bm = studentize(D, idx)
    sq = np.sqrt(n)
    t_obs = sq * mean / omega

    # Hansen's threshold for which cells are "poor enough" to be recentred to zero:
    #     keep mu_k = mean_k   iff   mean_k >= -sqrt(omega_k^2 * 2 log log n / n)
    # Studentised by sqrt(n)/omega_k, every omega cancels and it collapses to a SCALAR,
    # -sqrt(2 log log n). Written out as the per-cell expression it also evaluates to
    # inf/inf = NaN for cells that never trade (omega = inf), and a NaN threshold silently
    # makes the comparison False and recentres a live cell to zero.
    thresh = -np.sqrt(2.0 * np.log(np.log(n)))
    mu = {
        "upper": mean.copy(),                                  # recentre everything
        "consistent": np.where(t_obs >= thresh, mean, 0.0),
        "lower": np.maximum(mean, 0.0),                        # recentre only winners
    }
    out = {
        "n_cells": k, "n_sessions": n,
        "t_obs_max": float(np.max(t_obs)),
        "argmax": int(np.argmax(t_obs)),
        "mean_at_argmax": float(mean[int(np.argmax(t_obs))]),
        "session_hash": hashlib.sha256("|".join(sessions).encode()).hexdigest()[:16],
    }
    for name, m in mu.items():
        stat = sq * (bm - m[None, :]) / omega[None, :]
        out[f"boot_max_{name}"] = np.max(stat, axis=1).astype(np.float64)
    return out


def spa_pvalue(t_obs_max: float, boot_max: np.ndarray) -> float:
    """P(max under the null > observed). Reduced across shards before being called."""
    return float(np.mean(boot_max > t_obs_max))


def effective_trials(D: np.ndarray) -> float:
    """How many INDEPENDENT bets does a correlated grid actually represent?

    This exists because "just add more cells, the bootstrap handles it" is only half true.
    The bootstrap does handle cross-cell dependence correctly, so near-duplicate cells cost
    almost nothing. But STRUCTURALLY different cells -- a different window, the opposite
    direction -- are close to genuinely new trials and they do cost.

    Reported via the participation ratio of the eigenvalue spectrum of the cell correlation
    matrix, `(sum lambda)^2 / sum(lambda^2)`. For k identical cells it returns 1; for k
    orthogonal cells it returns k. It is a descriptive number, NOT an input to the p-value
    -- the SPA bootstrap already accounts for the dependence -- but it is the honest answer
    to "how big was the search really", and it is what should be compared against the
    minimum-backtest-length literature rather than the nominal cell count.
    """
    A = D - D.mean(axis=0, keepdims=True)
    sd = A.std(axis=0, ddof=1)
    keep = sd > 0
    if keep.sum() < 2:
        return float(keep.sum())
    A = A[:, keep] / sd[keep]
    C = (A.T @ A) / (A.shape[0] - 1)
    lam = np.linalg.eigvalsh(C)
    lam = lam[lam > 0]
    return float(lam.sum() ** 2 / (lam ** 2).sum())
