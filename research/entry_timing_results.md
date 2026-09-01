# Results — Does his entry timing have forward edge?

**Executes `research/entry_timing_preregistration.md`.** Run 2026-08-26. Zero new API calls;
everything from `atm_paths.pkl` (197 ATM contracts) and `bars_long.pkl`.

> ## HEADLINE
>
> **The pre-registered C-skill test is VOID — I built lookahead into my own control arm.**
> Corrected with a clean control, **his entry timing is indistinguishable from random.**
> **C-money is unaffected and is a clean null:** a mechanical ATM fixed-hold on his own
> entries is worth **-$0.53 to +$4.93/week** at 1 contract. His actual discretionary tape
> runs **+$59.52/week (QQQ/SPY)** and **+$85.59/week (QQQ)**. The value is in the exits.

## Funnel

```
QQQ/SPY round trips                          357
  excluded entry_too_late                     17    (< 25 min of path before 15:45)
  excluded horizon_unreachable                 1
  evaluated                                  339
distinct days                                123      range 2025-05-01 -> 2026-02-20
median entry time 11:39   median ATM ask $1.18   median placebo pool 60 bars
calls 181  puts 158   QQQ 269  SPY 70
```

---

# PART 1 — C-skill is VOID (self-inflicted lookahead)

## What the raw run said

```
C-skill  5m   -1.96%  Holm 0.5718   [under-powered]
C-skill 10m   -5.96%  Holm 0.0192   CLEARS
C-skill 15m  -11.88%  Holm 0.0008   CLEARS
C-skill 25m  -20.86%  Holm 0.0008   CLEARS
```

Three clear Holm, all saying his entries are *worse* than random. **This result is not
reported as a finding, because the placebo arm is contaminated.**

## The tell

The placebo arm returned **+27.12% mean with a 74.3% win rate** for a 25-minute hold on a
0DTE ATM option. That is not a plausible number for buying premium. A control arm that prints
money is a broken control arm.

## The defect

**The ATM strike is selected from the underlying spot at HIS entry second T.** A placebo entry
at `i < T` therefore holds a contract chosen with information from the future — at time `i`
that strike was not yet ATM, and the "trader" could not have known to buy it. Because his
entries are momentum chases (prior-15-min move **+0.2029 sigma** in his direction, 89.3% of
the time), the underlying reliably travelled *toward* that strike between `i` and `T`. Every
pre-entry placebo gets a free ride on the move he entered late.

Compounding it, the +/-30 min window is **centred on a momentum event**, so the pre-entry half
of the pool inherits a systematic tailwind.

This is a **pre-registration defect, not an execution error.** The design in section 3 was
wrong when written. Section 9's leak audit asserted "placebo seconds drawn from the clock
only, never from outcomes" — that was true of the *seconds* and false of the *contract*.

## The diagnostic that establishes it

Split the placebo pool at his entry. Post-entry bars carry **no strike lookahead** (the
contract is still ~ATM there, and no future information selects it).

| horizon | PLACEBO pre-entry | HIS entry | PLACEBO post-entry |
|---|---|---|---|
| 5m | +9.19% (56.0% win) | +1.82% (43.5%) | -0.43% (43.4%) |
| 10m | +20.66% (63.1%) | +3.98% (46.2%) | +0.79% (44.0%) |
| 15m | +31.02% (67.3%) | +3.95% (45.9%) | +2.43% (44.4%) |
| 25m | **+49.30% (71.3%)** | +6.44% (47.3%) | +5.03% (43.5%) |

Win rates are **per-draw**, so they are comparable across arms. (The original run reported the
fraction of 20-draw *averages* above zero, which is mechanically inflated toward 50-100% and
was itself misleading — a second, separate reporting error.)

```
 5m  pre  - HIS   +7.37%  CI [ +4.42%, +10.42%]  p 0.0003
10m  pre  - HIS  +16.68%  CI [+12.41%, +20.98%]  p 0.0003
15m  pre  - HIS  +27.07%  CI [+21.55%, +33.09%]  p 0.0003
25m  pre  - HIS  +42.85%  CI [+34.70%, +51.25%]  p 0.0003

 5m  post - HIS   -2.25%  CI [ -5.35%,  +0.77%]  p 0.1535
10m  post - HIS   -3.19%  CI [ -7.32%,  +0.98%]  p 0.1245
15m  post - HIS   -1.53%  CI [ -7.09%,  +3.63%]  p 0.5825
25m  post - HIS   -1.41%  CI [ -8.62%,  +5.20%]  p 0.7115
```

The pre-entry half carries the entire effect. The clean half is **null at all four horizons.**

## Corrected conclusion — EXPLORATORY, not confirmatory

The post-entry-only comparison is a **post-hoc restriction of the control**, adopted to remove
a leak, not pre-registered. It cannot confirm a hypothesis. It can and does invalidate the
pre-registered one.

Against a lookahead-free control, **his entry timing is neither better nor worse than picking
a random moment in the following 30 minutes on the same contract.** All four CIs straddle
zero.

Two notes that tilt *in his favour* and still leave a null:

- Post-entry bars have **less time to expiry**, so they carry more theta decay per minute.
  That handicaps the control, and the control still ties.
- If he were systematically buying local price extremes, post-entry bars would buy cheaper and
  *beat* him. They do not.

This satisfies **section 7, falsifier row 1**: *his entry timing shows no measurable skill.*
It reaches that verdict through a different route than pre-registered, and is labelled
accordingly.

---

# PART 2 — C-money (UNAFFECTED, clean null)

C-money uses **only his entry second** and the ATM strike selected **at that second**. No
future information enters. The lookahead above does not touch it.

```
comparison            effect            95% CI           p       Holm    MDE(80%)
C-money  5m (vs 0)    +1.82%  [ -1.07%,  +4.77%]   0.2282   0.5718    +4.20%  [underpowered]
C-money 10m (vs 0)    +3.88%  [ -0.31%,  +8.34%]   0.0708   0.3540    +6.19%  [underpowered]
C-money 15m (vs 0)    +3.82%  [ -1.83%, +10.15%]   0.2040   0.5718    +8.57%  [underpowered]
C-money 25m (vs 0)    +6.26%  [ -1.29%, +15.17%]   0.1108   0.4432   +11.82%  [underpowered]
```

**0 of 4 clear.** All four point estimates are positive; all four CIs include zero; all four
are underpowered for their own effect. Holm is applied at **m = 8** per section 11, which is
not relaxed because four family members were voided — at m = 4 the best would still be 0.283.

### Dollars — the answer to the actual question

At 1 contract, net of the real **$0.0808/contract round-trip fee**, 8.0 trades/week:

| horizon | $/trade | $/week | 95% CI per trade |
|---|---|---|---|
| 5m | -$0.07 | -$0.53 | [-$4.08, +$4.08] |
| **10m** | **+$0.61** | **+$4.93** | [-$5.51, +$6.76] |
| 15m | -$0.41 | -$3.31 | [-$8.37, +$7.54] |
| 25m | -$1.76 | -$14.18 | [-$11.25, +$8.54] |

**The stated objective is $200-500/week.** The best horizon's *upper* confidence bound is
+$6.76/trade = **$54/week at 1 contract** — still 4x short of the floor of the target, at the
optimistic end of the interval. The point estimate is $4.93/week.

Scaling size scales the CI with it. This is a coin flip multiplied by contract count, not an
edge that compounds.

---

# PART 3 — What his actual tape says (context, not a test)

Day-clustered bootstrap, 20,000 reps, on realised broker P&L:

```
all symbols     n=443  days=158  mean  +$3.28/trade  CI [-$13.09, +$19.19]  p 0.6801
QQQ/SPY         n=357  days=126  mean  +$7.03/trade  CI [ -$9.69, +$23.99]  p 0.4098
QQQ             n=279  days= 89  mean  +$8.72/trade  CI [ -$8.18, +$25.41]  p 0.3035
single names    n= 86  days= 56  mean -$12.28/trade  CI [-$57.35, +$29.16]  p 0.6089
```

| cut | net | span | rate |
|---|---|---|---|
| all symbols | +$1,452.82 | 75.3 wk | **+$19.30/wk** |
| QQQ/SPY | +$2,508.54 | 42.1 wk | **+$59.52/wk** |
| QQQ | +$2,433.22 | 28.4 wk | **+$85.59/wk** |
| single names | **-$1,055.72** | — | — |

**Positive at every cut. Significant at no cut.**

### The single-name split is POST-HOC — treat with suspicion

Dropping single names would have raised his total by 73%. But this is a two-way subset split
chosen *after* seeing outcomes, which is precisely the move the user's own DO-NOT list
forbids, and it does not clear on its own terms (p = 0.61 across 86 trades, 11 tickers, CI
spanning -$57 to +$29).

There is a **non-statistical** reason to prefer QQQ/SPY that does not depend on this split:
no options data entitlement for single names, so the entire research programme is scoped to
the indices regardless. That reason stands on its own. The +73% figure does not.

### Discretion vs mechanism — the sharpest comparison available

| | $/week |
|---|---|
| mechanical ATM fixed-hold on his own entries (best horizon) | **+$4.93** |
| his actual discretionary QQQ/SPY tape | **+$59.52** |
| his actual discretionary QQQ tape | **+$85.59** |

A ~12-17x gap, on overlapping trades. Combined with the construction result — **his
discretionary exit beats a mechanical +25% target by 8.97pp, Holm 0.027**, the only comparison
in that family that cleared — the evidence consistently locates whatever he has in **exit
management**, not entry selection.

Neither figure is statistically established. The direction is consistent across two
independent tests.

### How much more data would settle it

QQQ per-trade SE is ~$8.57 against a mean of $8.72, i.e. **t ~ 1.0**. Reaching t = 2 requires
roughly **4x the observations** — about **356 more QQQ trading days**, ~1.5 years at his
current rate.

**No further analysis of the existing data can resolve this.** The sample is the binding
constraint, not the method. This is the strongest available argument for ending the research
programme rather than extending it.

---

## Section 8 read-out — DESCRIPTIVE ONLY, no p-values, no claims

Signed underlying move in his traded direction, in that session's sigma, at his entries:

| horizon | mean | median | frac > 0 |
|---|---|---|---|
| 5m | **-0.0581** | -0.0244 | 48.4% |
| 10m | +0.0214 | +0.0396 | 51.6% |
| 15m | +0.0536 | +0.0890 | 53.4% |
| 25m | +0.0739 | +0.1333 | 56.0% |

Per section 8 this carries **no inferential apparatus** and **no conclusion above may be
revised on the basis of it.**

## Errors in this experiment, recorded

1. **Strike-selection lookahead in the placebo arm** (design defect, present in the
   pre-registration as written). Voids C-skill. Found by inspection of an implausible control
   return, confirmed by the pre/post split.
2. **Placebo win rate computed on 20-draw averages**, not per draw — mechanically inflated and
   not comparable to the single-draw HIS win rate. Corrected to per-draw above.

Neither error affects C-money or Part 3.
