# MOMO_CHASE sweep — result: null. The form is closed.

Run 2026-09-15/16 on Northeastern Discovery, array job `10374118`, 500 tasks.
Pre-registration: `research/momo_sweep_preregistration.md`. Reported under §7, which
requires the full distribution regardless of outcome.

## Verdict

**Hansen SPA, studentized, p(consistent) = 0.8227.** The null — *no cell in the grid beats
not trading, net of costs* — is not rejected, and is not close to being rejected.

| recentring | p |
|---|---|
| lower | 0.7276 |
| **consistent (governs)** | **0.8227** |
| upper | 0.8500 |

All three agree, which matters: the printed "wide gap" note refers to a 0.1224 spread, but
when every recentring sits between 0.73 and 0.85 the verdict is not sensitive to how
borderline cells are treated. Observed max studentized statistic **t = +4.0962**.

**§8 row 1 fires: no parameterization of the momo-chase form beats not trading, net of
costs, anywhere in 2,822,400 cells. MOMO_CHASE is closed as a standalone strategy. The
2022–2026 held-out set was never touched and must not be.**

## The reduction is legitimate

| check | value |
|---|---|
| shards | **500 of 500** — none missing |
| session hash | `ad6011cfb4c6128d`, **identical in every shard** |
| cells reduced | **2,822,400** — exactly the grid |
| sessions | 1,497 (5.94 yr), 2016-01-06 → 2021-12-31, QQQ only |
| cost charged | $0.0100/share round trip |
| bootstrap | B = 10,000, stationary, block 5.0 |

The SPA verdict is **exact over the whole grid**, not an approximation: Hansen's statistic
is a maximum, a maximum is associative, and `stats_test.py` asserts that the shard-reduced
maximum equals the whole-grid maximum rather than assuming it. The identical session hash
is what makes pooling the 500 bootstrap-max vectors valid.

## The apparent paradox, which is the whole point of the exercise

Read naively the output looks like a discovery:

- best cell nets **+$2.3245/session** over 414 trades
- annualised Sharpe **+1.688**
- breakeven spread **$0.1550/share — 15x the $0.01 actually charged**
- **876,718 cells (31.1%)** have a positive net mean

And yet p = 0.82. Both things are true, and the reconciliation is the reason this was
pre-registered instead of eyeballed:

**With 2,822,400 cells searched, a maximum this large is exactly what the null produces.**
The bootstrap distribution of the max studentized statistic under the null puts 82% of its
mass above t = 4.0962. A t of 4.10 would be remarkable for one pre-specified strategy —
Harvey/Liu/Zhu's hurdles for this literature are 3.39–3.78 — but those hurdles are for a
few hundred trials, not millions.

**§5 predicted this almost exactly.** It stated before the run that
`E[max Sharpe | true Sharpe = 0] ≈ 1.65 annualised at 5,000 independent trials`. The
observed winner: **1.688**. The best result in 2.8M cells landed within 2% of what the
pre-registration said pure noise would produce.

Note also *what* the best cell is: `fade`, not `chase` — afternoon window 13:00–15:55,
σ-threshold 1.25, ADX ≥ 30, all other gates off. The one-bit hypothesis flip built into the
grid means the grid's best-looking cell is mean reversion rather than momentum. It still
does not survive. So this null covers the reversal form too, not merely the chase form.

## Second gate — moot, and leaning the same way

| | |
|---|---|
| effective trials (measured, 1,500 retained cells) | 31.4 |
| floor at measured n_eff (**governs**, §4b) | Sharpe ≥ 0.858 |
| floor at nominal 2,822,400 | Sharpe ≥ 2.080 |
| winner's Sharpe | **+1.688** |

The winner **clears the governing floor but fails the conservative bracket** — flagged in
the output as "borderline, not a clean pass". §4b requires *both* gates, and SPA already
failed, so this changes nothing. It is recorded because it leans the same direction rather
than against.

The measured `n_eff` of 31.4 is itself informative: 1,500 top cells behave like ~31
independent bets. The grid is enormously redundant, which is exactly why the nominal count
is the wrong number to feed MinBTL and why SPA's bootstrap — which prices that dependence
directly — is the governing test.

## Distribution across all 2,822,400 cells (§7)

| percentile | net $/session |
|---|---|
| p0 | −44.9223 |
| p1 | −4.3161 |
| p5 | −2.0944 |
| p25 | −0.6905 |
| **p50** | **−0.1349** |
| p75 | +0.0502 |
| p95 | +0.6343 |
| p99 | +1.2624 |
| p100 | +5.6007 |

The median cell loses money. 31.1% are net positive, and that same 876,718 is also the
count whose breakeven beats the real spread — an identity (net > 0 ⟺ gross/share > spread),
and a useful consistency check that the cost accounting is coherent.

## The frozen V0 is in the bottom 6% of its own family

**Cell #928,056 — the exact parameterization running in the live lab — ranks 2,656,847 of
2,822,400 by net mean. It beats only 5.9% of the grid.**

- net **−$1.9352/session**
- breakeven **−$0.0058/share**, i.e. **negative**: on this window it loses money *gross*,
  before any cost is charged at all

The collector originally printed this as "94.1th pct", which reads as the opposite of what
it means; fixed in commit `8860e1f`.

**Four caveats, because this is the finding most likely to be over-read:**

1. The sweep is **QQQ only, 2016–2021**. The live lab trades 15 symbols in 2026.
2. The sweep's V0 uses `max_per_day=0` — reproducing `momo_v2.py`, which does not apply the
   frozen setup's 3-per-day cap. The live setup does.
3. `sweep --validate` over **QQQ+SPY across all years** gives V0 **+$0.0339/trade gross,
   n = 8,598**. The negative figure here is on a strict subsample. Both are correct; they
   are different samples.
4. The live lab measures something this cannot: prospective performance with real fills.

So this **weakens the prior for MOMO_CHASE substantially without settling the live test.**

## What this does and does not change for the live lab

**It changes nothing operationally, and that is deliberate.** §9 prohibits any addition to
the frozen live lab under any outcome, and the live lab's own freeze says the clock never
resets. Pulling a setup mid-test because a retrospective search came back unfavourable is
precisely the intervention a forward-test freeze exists to prevent — it would convert a
prospective test into a retrospective one.

What it does change is the **prior the 200-trade-per-setup checkpoint should be read
against**, and that is why this is recorded now, before that checkpoint arrives, rather
than afterwards. MOMO_CHASE shares stands at 163/200.

## Prohibited from here (§9)

No cell added, removed or re-ranged now that the result is known. No re-run at a different
cost assumption to rescue a cell — the breakeven column is published precisely so cost
sensitivity is visible without re-running. No promotion of a DSR or PBO figure over this
verdict. **No use of the 2022–2026 holdout: §8 row 1 fired, so it is never touched.** No
addition to the frozen live lab.

## What it cost, and what it bought

~700 CPU-hours across 500 tasks. It bought a closed question: the momo-chase form — and its
fade mirror — does not contain a profitable parameterization on QQQ 2016–2021 net of a
realistic penny spread, and the best thing 2.8M attempts produced is statistically
indistinguishable from what searching 2.8M times through noise produces.

This is the **ninth null** in the programme (`research-programme-verdict`), and the first
one established by exhausting a hypothesis class rather than by testing a single member of
it.
