"""Checks for stats.py. Run it before trusting any sweep verdict.

    python -m trade_analysis.momo_sweep.stats_test

Four properties, in order of how badly a failure would hurt:

  1. SHARD REDUCTION IS EXACT. max-over-shards must equal max-over-all-cells. The entire
     storage design depends on this; if it is wrong the sweep reports a p-value computed
     over a subset of the grid while claiming the whole grid, which is anti-conservative in
     exactly the direction that manufactures a false discovery.
  2. STUDENTIZATION RECOVERS POWER. This implementation exists only because arch's does not
     studentize. If mine does not beat arch on Hansen's own design, it has the same bug and
     there was no point writing it.
  3. The vectorised stationary bootstrap has the right block structure.
  4. effective_trials is 1 for duplicated cells and ~k for independent ones.
"""
from __future__ import annotations

import sys

import numpy as np

from .stats import boot_indices, effective_trials, spa_pvalue, spa_shard_max


def t_shard_reduction() -> bool:
    rng = np.random.default_rng(11)
    n, k = 200, 60
    D = rng.standard_normal((n, k)) * rng.uniform(0.5, 3.0, k) + rng.uniform(-0.2, 0.2, k)
    sess = [f"d{i}" for i in range(n)]
    idx = boot_indices(n, 200, 5.0)

    whole = spa_shard_max(D, idx, sess)
    shards = [spa_shard_max(D[:, a::4], idx, sess) for a in range(4)]

    ok = True
    for key in ("boot_max_consistent", "boot_max_upper", "boot_max_lower"):
        red = np.max(np.vstack([s[key] for s in shards]), axis=0)
        if not np.allclose(red, whole[key], rtol=0, atol=1e-12):
            print(f"    FAIL {key}: max mismatch {np.abs(red - whole[key]).max():.3e}")
            ok = False
    red_obs = max(s["t_obs_max"] for s in shards)
    if not np.isclose(red_obs, whole["t_obs_max"], rtol=0, atol=1e-12):
        print(f"    FAIL t_obs_max: {red_obs} vs {whole['t_obs_max']}")
        ok = False
    if len({s["session_hash"] for s in shards} | {whole["session_hash"]}) != 1:
        print("    FAIL session_hash differs across shards")
        ok = False
    print(f"  1. shard reduction exact ........................ {'PASS' if ok else 'FAIL'}")
    return ok


def t_studentization_power() -> bool:
    """Hansen Example 4: one good model, k-1 poor models with 4x the standard deviation.

    Studentization is supposed to stop the high-variance junk from dominating the null
    distribution of the maximum. Without it, the junk sets the bar and the good model cannot
    clear it.
    """
    try:
        from arch.bootstrap import SPA as ArchSPA
    except ImportError:
        print("  2. studentization power ......................... SKIP (arch absent)")
        return True

    rng = np.random.default_rng(5)
    n, k, trials = 400, 20, 60
    sd = np.r_[1.0, np.full(k - 1, 4.0)]
    mu = np.r_[2.5 / np.sqrt(n), np.full(k - 1, -3.0 / np.sqrt(n))]

    idx = boot_indices(n, 400, 4.0)
    sess = [f"d{i}" for i in range(n)]
    hit_mine = hit_arch = 0
    for _ in range(trials):
        D = rng.standard_normal((n, k)) * sd + mu
        r = spa_shard_max(D, idx, sess)
        if spa_pvalue(r["t_obs_max"], r["boot_max_consistent"]) < 0.05:
            hit_mine += 1
        a = ArchSPA(np.zeros((n, 1)), -D, reps=400, block_size=4, seed=3)
        a.compute()
        if float(np.asarray(a.pvalues)[1]) < 0.05:      # 'consistent' row
            hit_arch += 1

    pm, pa = hit_mine / trials, hit_arch / trials
    ok = pm > pa
    print(f"  2. studentization power ......................... "
          f"{'PASS' if ok else 'FAIL'}  mine {pm:.2f} vs arch {pa:.2f}")
    if not ok:
        print("     Studentised SPA did NOT beat arch's unstudentised one on the design")
        print("     built to show the difference. Suspect this implementation, not arch.")
    return ok


def t_block_structure() -> bool:
    n, reps, block = 500, 400, 10.0
    idx = boot_indices(n, reps, block)
    ok = bool(((idx >= 0) & (idx < n)).all()) and idx.shape == (reps, n)
    # a "continuation" is a step where the index advanced by exactly one (mod n)
    cont = ((idx[:, 1:] - idx[:, :-1]) % n) == 1
    mean_block = 1.0 / (1.0 - cont.mean())
    ok = ok and abs(mean_block - block) / block < 0.25
    print(f"  3. stationary bootstrap block structure ......... "
          f"{'PASS' if ok else 'FAIL'}  mean block {mean_block:.1f} (target {block:.0f})")
    return ok


def t_effective_trials() -> bool:
    rng = np.random.default_rng(3)
    n = 400
    base = rng.standard_normal((n, 1))
    dup = np.repeat(base, 50, axis=1) + rng.standard_normal((n, 50)) * 1e-9
    indep = rng.standard_normal((n, 50))
    e_dup, e_ind = effective_trials(dup), effective_trials(indep)
    ok = e_dup < 1.5 and e_ind > 35
    print(f"  4. effective_trials ............................. "
          f"{'PASS' if ok else 'FAIL'}  duplicated {e_dup:.2f}, independent {e_ind:.1f}")
    return ok


def t_statistic_scale() -> bool:
    """t_obs must BE a t-statistic, not merely be monotone in one.

    This check exists because every other test here is scale-invariant and all of them
    passed while `t_obs` was inflated by sqrt(n) -- a factor of 39 on this sample. The
    p-value is unaffected (observed and null scale together), so nothing that compares
    p-values can see it. What IS affected is Hansen's recentring threshold, which is a fixed
    -sqrt(2 log log n) ~ -1.9: against an inflated t every cell clears it, `consistent`
    degenerates into `upper`, and the protection against a grid full of poor models
    quietly stops existing.

    On i.i.d. data with a known mean and SD, `mean / (sd/sqrt(n))` is the answer, and
    `t_obs` must match it to within bootstrap noise.
    """
    rng = np.random.default_rng(77)
    n = 1500
    x = rng.standard_normal((n, 1)) * 2.0 + 0.15
    idx = boot_indices(n, 800, 1.0)              # block 1 == i.i.d. resampling
    r = spa_shard_max(x, idx, [f"d{i}" for i in range(n)])
    analytic = float(x.mean() / (x.std(ddof=1) / np.sqrt(n)))
    got = r["t_obs_max"]
    ok = abs(got - analytic) / abs(analytic) < 0.10
    print(f"  5. t_obs is on the t scale ...................... "
          f"{'PASS' if ok else 'FAIL'}  got {got:.2f}, analytic {analytic:.2f}")
    if not ok:
        r_ = got / analytic
        print(f"     ratio {r_:.1f}; sqrt(n) is {np.sqrt(n):.1f} -- if those match, the")
        print("     statistic has an extra sqrt(n) in it.")
    return ok


def main() -> int:
    print("\n  stats.py checks\n")
    res = [t_shard_reduction(), t_studentization_power(),
           t_block_structure(), t_effective_trials(), t_statistic_scale()]
    print()
    if all(res):
        print("  ALL PASS\n")
        return 0
    print(f"  {res.count(False)} FAILED -- do not run the sweep\n")
    return 1


if __name__ == "__main__":
    sys.exit(main())
