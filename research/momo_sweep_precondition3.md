# Pre-condition 3 (lookahead) does not pass as written — and the reason is power, not leakage

**Status: OPEN. Requires a decision before the sweep is submitted.** Written 2026-09-15,
before any sweep shard has been run on the cluster, so nothing here is informed by the
sweep's outcome.

`research/momo_sweep_preregistration.md` §6 lists four mandatory pre-conditions and says
the sweep does not run unless all four pass. Three pass. This is the fourth.

## What §6.3 says, and what it does

> **3. Lookahead audit.** A run with the signal shifted one bar into the future must score
> **materially BETTER**. If peeking does not help, the honest path already contains future
> information and every number is void.

Implemented in `trade_analysis/momo_sweep/audit.py`. Measured on QQQ 2016–2021 (1,497
sessions), V0 plus a seeded random sample of grid cells:

| | honest | +1 bar | paired |
|---|---|---|---|
| **V0 anchor** | −0.7118 | **+2.2013** | **+2.9130** |
| trading cells, mean | −0.1046 | +0.0530 | +0.1687 |
| trading cells, median | | | −0.0353 |
| trading cells improved | | | **48%** |

The anchor passes overwhelmingly — a sign flip from loss to profit. The population does
not: fewer than half of cells improve, and the median is negative while the mean is
positive, i.e. heavily right-skewed.

## The two competing explanations, and the evidence that separates them

The §6.3 inference — "peeking doesn't help, therefore the honest path already peeks" — is
only one of two readings. The other is that one bar carries too little information to
detect on most cells. These make **different, testable predictions**, which is what makes
this resolvable rather than a matter of interpretation.

**Prediction 1 — leakage is cell-independent.** All 2.8M cells execute the identical code
path. If that path leaked the future, peeking would fail to help *regardless of how much a
cell trades*.

**Prediction 2 — reselection noise is symmetric.** Shifting `close` forward does not merely
add information; it changes which bars clear `abs(z) >= sigma_mult`, so it reselects the
trade population. For a cell with no signal that is a reshuffle: ~50% improve, paired mean
~0, again independent of trade count.

Both predict a flat relationship with trade count. Observed, on two independent cell
samples drawn with different seeds and sizes:

| trades | 151-cell sample | 121-cell sample |
|---|---|---|
| 1–50 | 39%, −0.0329 | 38%, −0.0406 |
| 50–200 | 33%, −0.1779 | 22%, −0.2150 |
| 200–1,000 | 47%, −0.0237 | 47%, −0.0626 |
| 1,000–5,000 | **56%, +0.2150** | **59%, +0.4839** |
| 5,000+ | **67%, +3.1621** | **67%, +3.1621** |

**Monotone in both samples.** Peeking helps precisely the cells that trade enough for the
effect to be measurable, and the benefit grows with the count. Neither competing prediction
survives that. Bin edges were fixed before the numbers were seen, and reported as the whole
curve rather than as a threshold, specifically so the conclusion cannot rest on a cutoff
chosen after the fact.

## The audit is proven capable of detecting a lookahead

The above would still be unsafe to act on if the audit simply could not see leakage at all.
So `audit.py` includes a power check: each cell is handed `force_dir`, the sign of its own
trade's outcome (`open[k+1+T] − open[k+1]`, exactly as `run_cell` computes entry and exit).
That is a perfect oracle, and an honest harness must convert it into large profits.

| | honest | oracle |
|---|---|---|
| V0 anchor | −0.7118 | **+34.0021** |
| trading cells profitable | | **113/113 (100%)** |
| mean | −0.1120 | **+11.1645** |

**100% of trading cells, against a pre-stated bar of 95%.** The audit detects an
unambiguous lookahead without exception. Therefore the one-bar result is a statement about
the information content of one bar, **not** evidence that the honest path is already
peeking.

## Two errors of mine on the way here, recorded because the method matters

1. **The first criterion was arithmetically invalid.** It required
   `peek_mean > 3 * abs(honest_mean)`. With a negative baseline that demands the peeking
   mean be positive and large, so the test would fail even if peeking improved every cell
   without bound. A test that cannot be passed by an arbitrarily large improvement is not
   measuring improvement. Demonstrable from the algebra alone, independent of the outcome.
2. **The first power check was not an oracle.** It shifted the whole `close` array forward
   by the holding period, which moves *both* ends of the momentum window: `raw` became the
   return from `k−L+T+1` to `k+T+1` while the trade still spanned `k+1` to `k+1+T`. Only
   weakly correlated with `exit − entry`, it scored 53% and read as "the audit is blind",
   when the truth was that the oracle was not an oracle. Fixed by overriding the direction
   outright via `force_dir`; 53% → 100%.

Both are recorded because a pre-condition rewritten twice by the person who wants it to
pass is exactly the pattern a pre-registration exists to prevent. The defence is not that
the conclusion is convenient but that each defect is demonstrable without reference to
which way the run came out.

## Pre-condition status

| # | condition | status |
|---|---|---|
| 1 | V0 reproduces `momo_v2.py` at n = 8,598 | **PASS** — exact, re-verified after both engine hooks |
| 2 | `stats_test.py` passes | **PASS** — 6/6 (test 2 SKIPs: `arch` absent from this venv) |
| 3 | lookahead audit | **fails as written**; audit proven powered; see above |
| 4 | direction placebo | **PASS** — z = −0.10 across 4 draws |

## The decision required

§6.3's *intent* is to establish that the engine does not see the future. The evidence says
it does not: a perfect oracle is detected 100% of the time, the validated anchor improves
by +2.91 under a one-bar peek, and the benefit is monotone in trade count across two
independent samples. §6.3's *letter* — a grid-wide improvement — is not met, because the
grid is mostly cells too thin to measure.

Options, in the order I would rank them:

1. **Amend §6.3 to be evaluated on the anchor plus the oracle power check**, citing this
   document, dated, before the run. The amendment is on a stated principle — an audit must
   be read where it has power — and not on an observed outcome.
2. **Amend §6.3 to add a minimum-trades qualifier** (e.g. cells with ≥1,000 trades), which
   the curve supports but which involves choosing a cutoff after seeing the curve.
3. **Leave §6.3 as written and do not run the sweep.** Defensible, and the cost is that a
   177-CPU-hour job that is otherwise ready stays unrun on a criterion whose own power
   check says it cannot answer the question asked of it.

**I am not treating this as mine to close.** The criterion has already been rewritten once
in this session, and the value of a pre-registration comes from the person who wants the
answer not being the one who keeps re-specifying the test.

## Unrelated but in the same commit: the job would have failed 500/500 tasks

`submit_sweep.sbatch` activated a conda env named `vrp`. There is no such env on Discovery
(`trade-venv`, `trade-env`, `legal-env`, `8674-env`), and with no `uv.lock` or
`pyproject.toml` in the repo the uv branch was never taken, so every array task would have
reached `conda activate vrp` and died. Fixed, and the interpreter now has to prove it can
`import numpy, trade_analysis.momo_sweep.engine` before `srun`. The dependency surface is
numpy alone: `trade_analysis/__init__.py` has no imports, so the `config.py`-raises-on-import
chain is never entered, and Hansen's SPA is local rather than from `arch`.

Also measured, replacing the sbatch header's estimate: **0.151 ms per cell-session**, so
2,822,400 cells x 1,497 sessions is **~177 CPU-hours**, not the ~300 assumed — about 21
minutes per task at 500 shards, inside a 6 h wall by a wide margin. On `short` (50
concurrent) that is 10 waves, roughly 3.5 h of wall clock.
