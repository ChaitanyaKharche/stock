# Arm A — failed replication, and the reason is in my specification

*Run 2026-09-11. `submit_arm_a.sbatch` / `collect_arm_a.py`, 12 entry times, git `915a5125`,
numpy 2.2.6 / pandas 2.3.2, θ = 0, min_history 250. Vilkov SPXW panel, 1,397 sessions
2016-09 → 2024-05, of which 1,147 usable after the history burn-in.*

**This is not evidence about Almeida/Freire/Hizmeri's result.** §4 of the pre-registration
committed the reading in advance: *"Arm A fails to reproduce → stop and debug the
implementation against their reported Sharpe. A failed replication is about our code until
proven otherwise."* That branch is the one we are in.

## The table

13,673 trades over 1,147 sessions. Buy share 78.4% overall.

| slot | n | buy% | gross | net | grossHdg | netHdg | netHdg t |
|---|---|---|---|---|---|---|---|
| 10:00 | 1146 | 95% | +0.000024 | −0.000062 | +0.000012 | −0.000073 | −0.94 |
| 10:30 | 1146 | 99% | +0.000022 | −0.000029 | −0.000048 | −0.000099 | −1.37 |
| 11:00 | 1147 | 100% | +0.000088 | +0.000041 | −0.000003 | −0.000049 | −0.67 |
| 11:30 | 1147 | 98% | +0.000013 | −0.000032 | −0.000054 | −0.000099 | −1.45 |
| 12:00 | 1147 | 91% | −0.000016 | −0.000059 | −0.000047 | −0.000090 | −1.42 |
| 12:30 | 1147 | 78% | +0.000017 | −0.000025 | −0.000033 | −0.000075 | −1.21 |
| 13:00 | 1144 | **21%** | −0.000074 | −0.000117 | +0.000015 | −0.000028 | −0.50 |
| 13:30 | 1132 | 74% | +0.000012 | −0.000028 | −0.000042 | −0.000082 | −1.55 |
| 14:00 | 1130 | 70% | +0.000124 | +0.000062 | +0.000045 | −0.000017 | −0.34 |
| 14:30 | 1131 | 57% | +0.000054 | +0.000012 | −0.000061 | −0.000103 | −2.14 |
| 15:00 | 1129 | 69% | +0.000027 | −0.000017 | −0.000026 | −0.000070 | −1.62 |
| 15:30 | 1127 | 88% | −0.000014 | −0.000060 | +0.000013 | −0.000032 | −1.01 |

Pooled: gross +0.000020 (t +0.28), net −0.000029 (t −0.42), gross-hedged −0.000020
(t −0.58), net-hedged −0.000069 (t **−2.06**).

Nothing here is readable as a test. The nominal t = −2.06 on net-hedged is across 12 entry
times tested, so Holm leaves it nowhere near significance — and more importantly it is the
P&L of trading on a biased signal, which is a measurement of the bias.

## Why it failed, precisely

| | |
|---|---|
| median `ep_physical / mid` | **1.0275** — the device sits 2.8% above market |
| mean `payoff / mid` | **0.9939** — options are genuinely rich by 0.6% (the premium) |

**The bias is 4.6× the signal it is meant to detect.** With θ = 0 the SSD band is degenerate
— a single point — so a 2.8% miscalibration decides the side on nearly every origin, and
the "rule" becomes "buy almost always."

**And the bias is not a constant, which is the useful part.** Buy share runs 95–100% at
10:00–11:30, collapses to **21% at 13:00**, and sits at 57% by 14:30. A level error would
shift every slot equally. What varies with the slot is the **horizon to settlement** — six
hours at 10:00, thirty minutes at 15:30 — and therefore the width and shape of the
distribution being integrated.

So this is a **tail-shape mismatch, not a level offset**: an empirical histogram of 250+
past settlement returns has fatter tails than the distribution the market is pricing, and
`E_P[max(R − K, 0)]` is a tail integral, so the overstatement grows with horizon. That
matters for what comes next, because **a tuned θ could not fix it** — a single constant
cannot absorb a bias that changes sign across the session.

## What this rules in and out

- **Rules out:** the bare physical expectation as a usable bound. My pre-registration chose
  θ = 0 for being parameter-free, and parameter-free turned out to mean degenerate.
- **Does not rule out:** the published effect. Almeida et al.'s bound is the sup/inf over
  concave utilities, which is a *band* whose width is derived rather than assumed. That is a
  linear program per origin and it is a materially different object from what ran here.
- **θ is not tuned.** §9 forbids moving a gate post-hoc, and fitting θ to a bias measured
  after the fact is exactly that. It would also not work, per the slot-dependence above.

## What did verify

- **`payoff` identity exact**: `max(sret − mnes_rel, 0)` matches the panel's `payoff` to
  **0.00e+00** over 49,968 call rows. The per-bar spot normalisation is understood.
- **16:00 rows correctly excluded** — they carry the *next* day's `sret` and exist on
  non-expiry days; including them would be a one-day lookahead on 522 non-sessions, and
  they are what inflates the session count from 1,397 to 1,919.
- **Provenance clean** across all 12 tasks after a re-run: one git SHA, one numpy/pandas
  pair, one parameter set.

## The instrument caught two things about itself

**Mixed SHAs.** The first full run had slot 12:00 at `d0de4c9` and the rest at `c2d74ab`.
The code was byte-identical — uncommitted, then committed unchanged — but the collector
cannot know that and correctly refused. Re-run at one SHA.

**A gate that always fires.** It also flagged a dirty tree on all 12 tasks, which is true
and always will be: the live lab writes `live_lab_data/` inside this repo, so it is dirty
during and after every session. A gate with a 100% firing rate trains the reader to skip
it. Provenance now records `git_dirty` (reported) and `git_dirty_code` (gated, excluding
the data paths). Uncommitted *code* and mismatched SHAs remain hard failures.

## Next

A new pre-registration specifying the bound properly — the concave-utility sup/inf rather
than a point expectation — with the band width derived and the horizon-dependence of the
empirical distribution handled explicitly. That version is a linear program per origin
across 13,673 origins, and it would be the first thing in this programme to justify the
cluster on compute rather than on provenance.
