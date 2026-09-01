# Pre-registration — Observable Pre-Entry State at Execution Timestamps

**Status: PRE-REGISTERED, NOT YET RUN.** Written 2026-08-20 after (a) reading the full
research record, (b) verifying the coordinator's ESTABLISHED claims, and (c) sizing the data
fetches — and **before any endpoint below was computed.** Descriptive quantities already
computed and reported in the audit (P&L, latency, limit-price taxonomy, entry-time
distribution) are **inputs**, not endpoints; none of them is an outcome regression.

Deviations must be appended as dated amendments, never edited in place.

---

## 0. Two corrections to the brief, established before pre-registering

1. **`client_bid/ask_at_submission` is sign-flipped on 289 of 1,037 orders**, all sells.
   Repair: where `client_ask < 0`, set `true_bid = -client_ask`, `true_ask = -client_bid`.
   After repair the median sell fills at exactly the true bid. The claim "the sell-side
   spread metric is broken, do not use it" is **withdrawn** — it was a sign convention.
2. **"Buys land at median position 1.000 in the submission spread (full ask)" is a
   tick-quantization artifact.** 64.9% of buy submissions have a **1-cent** spread; 86.8%
   have <= 2 cents. Dividing by a one-tick denominator makes the ratio an integer tick count
   with observed range [-18, +12]. In absolute units the buy fill is at the displayed ask
   (median deviation 0c) and 0.64c inside it on average. **Report cents, never the ratio.**

Neither correction is used as an endpoint. Both change how endpoints are specified.

---

## 1. Sample

Source `research/round_trips.csv` (443 round trips), verified against
`research/order_history_raw.json` by independent replay: 1,222 executions, 405 contracts,
**zero** contracts with non-zero terminal position, **zero** negative running positions, and
execution-derived net = **$1,452.82**, matching the CSV to the cent.

Timestamps: `legs[0].executions[].timestamp`, UTC, converted to America/New_York.
**Decision time is the execution timestamp, not the order's `created_at`** — except where the
endpoint is explicitly about submission (H2), where `created_at` is used and stated.

| endpoint | universe | n |
|---|---|---|
| **P (primary)** | QQQ+SPY, entry and exit same session, hold >= 2.0 min | **329** |
| H1 | all 13 symbols, same-session | 427 |
| H2 | exit limit orders priced above the true ask at submission | 82 |

Hold >= 2.0 min on P because a duration-matched enumeration at H < 2 minutes has almost no
comparison mass and is dominated by quote staleness. 17 ETF trades are excluded on this rule;
the count is reported. **This floor is fixed now, before any percentile is seen.**

---

## 2. PRIMARY ENDPOINT — P: duration-matched entry-timing percentile

### Why this and not a prediction regression

Power, computed before choosing. On the 443 round trips: net dollar SD = **$156.70**,
day-clustered SE of the mean = **$8.24** (measured design effect **1.22**), so the MDE on any
dollar outcome is **+/-$23/trade** at 80% power against a realised mean of **+$3.28**. A
pre-entry predictor would have to move the outcome by seven times his entire average result
to be visible. **Every dollar-denominated prediction endpoint on this dataset is
uninformative by construction.** The percentile has SD 0.289 by definition, giving
SE ~ 0.0184 and **MDE ~ 0.051** at n=329 — it is the only properly powered endpoint
available, which is why it is primary.

### Specification

For each eligible round trip: fetch the 1-minute NBBO for **his exact contract** on **his
exact session** (ThetaData `/option/history/quote`, one chain request per
`(symbol, expiry, date, right)`; 190 requests). Valid minutes: `09:31 <= t <= 15:59`,
`bid > 0`, `ask > 0`, `ask >= bid`. Invalid minutes dropped, **never forward-filled**.

Let `H = round(hold_min)` clipped to `[2, 380]`. Enumerate every valid start minute `i` such
that `i + H` is also valid, and form

```
r(i)   = mid(i + H) / mid(i) - 1        mid = (bid + ask)/2
r_user = exit_vwap / entry_vwap - 1     (broker fill prices, gross of fees)
P      = ( #{r < r_user} + 0.5 * #{r = r_user} ) / N       (mid-rank; ties guaranteed)
```

**Primary statistic:** mean `P` over the 329 trips. **Null: 0.50.**

**Inference:** cluster bootstrap over **activity dates**, 10,000 replicates, percentile 95%
CI. Design effect measured on these data, not assumed.

### What holding duration fixed buys, and the one hazard it introduces

Fixing `H` at his **realised** duration removes the artifact that destroyed the previous
test's headline: there, the unconstrained comparison set averaged ~130-minute holds against
his ~25-minute holds, and a decaying 0DTE option makes early exit look like skill. Matching
duration also holds contract choice, day choice and direction fixed. What remains is
**entry-minute selection alone.**

**The hazard, stated in advance:** `H` is chosen *after* seeing the path, so conditioning on
it conditions on post-entry information. This *removes* his exit-timing skill from the
measurement (a trade that ran fast and was closed in 8 minutes is compared only against other
8-minute windows). The direction of this bias is toward 0.50; it cannot manufacture a
positive result. P therefore measures entry timing **net of** any exit skill, and a null on P
is not a null on his execution as a whole.

### Falsification condition, committed now

| result | conclusion |
|---|---|
| 95% CI within [0.45, 0.55] | **entry timing carries no detectable information.** Stop. |
| CI lower bound > 0.55 | entry timing carries information; proceed to one discriminating follow-up |
| CI upper bound < 0.45 | entry timing is systematically poor |
| otherwise | inconclusive; report as such, do not resolve via a secondary |

The bar is **0.55**, not the prior study's 0.53, because that study covered a different sample
and the duration-matched design has a tighter null; 0.55 is one MDE above 0.50 and is the
smallest bar this design can honestly defend.

### Relationship to the prior test — and how a contradiction will be handled

The prior pre-registered test returned mean P = **0.5057** [0.4528, 0.5584] at d=10 and
**0.5151** [0.4679, 0.5652] at d=20, cells that bracket the 22.7-minute ETF median hold.
This is **not a re-run of a failed experiment.** Three specific reasons the prior run could
not have detected the effect:

1. **It used a fixed duration grid applied to every trade**, because duration was
   unobservable. A trade actually held 6 minutes was scored against 20-minute windows. That
   mismatch attenuates toward 0.50 for every trade whose true H differs from the grid cell.
2. **Its sample was a different, smaller set** — 266 ETF round trips, 2025-05-05 to
   2025-11-26, from the date-only CSV. This sample is 329 eligible ETF trips spanning 2025-01
   to 2026-02, and roughly 60% of it lies outside that window.
3. **Its `r_user` used contract-level VWAP over an unknown number of fills with no
   timestamps**, so multi-tranche trades were unmatchable to any single duration.

**If P comes in materially above the prior d=20 cell, that is a result requiring explanation,
not a rescue.** The pre-committed explanation test is (2): recompute P restricted to the
2025-05-05..2025-11-26 overlap window AND to trips whose H falls in [15, 30] minutes. If P in
that intersection is consistent with the prior 0.5151, the difference is sample and
duration-matching; if P is elevated there too, the two studies genuinely disagree and the
prior study's `r_user` construction is the remaining suspect. This test is specified now.

### Pre-registered controls, run and reported BEFORE the primary

- **C1 calibration.** Draw a random valid start `i` for each trip, price `r(i)` under the
  primary convention, rank against the same set. 200 draws per trip. **Pass: grand mean in
  [0.49, 0.51].** Failure implies verdict invalid, no substantive conclusion.
- **C2 pricing-convention offset.** His fills are measured (section 0) at the displayed ask on
  entry (median 0c, mean -0.64c) and at the true bid on exit (median 0c, mean +1.34c). On a
  1-cent modal spread that is entry ~ mid + 0.5c and exit ~ mid - 0.5c, i.e. he pays roughly
  one tick of round-trip spread that the mid-to-mid enumeration does not charge. Quantify it:
  draw a random valid `(i, i+H)`, price it **ask->bid**, rank against the **mid->mid** set.
  The resulting mean is the offset `b`; **the primary is interpreted against `b`, not against
  0.50**, and `b` is reported whether or not it is convenient.
- **C3 support.** Count trips whose `r_user` lies outside `[min r, max r]`, and trips with
  fewer than 30 enumerable start minutes.

### Pre-registered secondaries (never promoted)

QQQ and SPY separately; ask->bid convention; single-fill-only subset (`n_buys==1 and
n_sells==1`); the overlap-window/duration-window intersection of the contradiction test;
median P; sign test.

---

## 3. SECONDARY ENDPOINT — H1: the one pre-entry state variable

**Exactly one** market-state variable is pre-registered. This is deliberate: the feature space
here is unbounded and I am the agent most able to generate it.

### Mechanism

The trader's stated setup is that price *clearing* a confluence of levels marks something.
Every mechanical test of that idea in this repo was run in a 09:30-09:45 window. His actual
median entry is **11:44** and only **5.6%** of entries fall in 09:30-09:45, so the coded rule
and the traded rule are different objects. Separately, his order flow is measured to be
**92.8% executable-on-arrival at entry** (249 market orders plus 194 of 250 limit orders
priced at or above the displayed ask; only 14 of 499 entry orders rest below the bid). A
trader who is 93% executable-on-arrival is, mechanically, **taking price that has already
moved** — he is not waiting at a level for price to come to him. The mechanism under test is
therefore momentum-chasing: he buys after the underlying has moved his way, and the question
is whether the size of that prior move relates to what happens next.

Participants and incentives: if the prior move is small, he is early relative to the crowd and
the option is cheap in vol terms; if the prior move is large, market makers have already
repriced the strike and he is paying for realised movement he then has to *exceed*. Same
premium, different amount of move already spent.

### Variable, and the timing convention

```
m = signed underlying return over the 15 minutes strictly BEFORE the entry second,
    in the direction of the option bought, scaled by trailing 20-session daily sigma:

    m    = sign * ( S(t_entry) / S(t_entry - 15min) - 1 ) / sigma_20
    sign = +1 for a call, -1 for a put
```

`S(t)` is the **close of the last completed 1-minute bar at or before `t_entry`**. A bar
labelled 11:44 spans [11:44, 11:45) and is not complete until 11:45, so for an entry at
11:44:37 the bar used is the one labelled **11:43**. This is the exact convention the lookahead
bug in `options_premium_backtest.py:152-158` violated and it is enforced here. `sigma_20` uses
only sessions strictly before the entry date. Window = **15 minutes**, fixed now; 5/30/60-minute
variants are a pre-declared sensitivity band, reported together, **none promotable**.

### Outcome and model

Outcome `Y = net / qty` (dollars per contract, net of fees actually paid) — per-contract so
that position size is not mechanically inside the outcome. Model `Y ~ m`, OLS point estimate,
day-clustered bootstrap (10,000 reps) for inference. Secondary scale: `pct`, reported
alongside, not promoted.

### Falsification and honest power

**Falsifier:** the 95% day-clustered CI on the slope contains zero.

**And it will almost certainly contain zero.** Per-contract `Y` SD is **$45.30**; at n=427 with
DEFF ~1.22 the SE on a per-SD slope is ~ **$2.7**, MDE ~ **+/-$7.5** per SD of `m`. That is
detectable only if the effect is more than twice his entire mean per-contract result. **This
endpoint is pre-declared underpowered for any plausible effect.** It is run because its
*descriptive* half is not underpowered at all: the distribution of `m` answers "is he chasing
or fading?" with near-zero measurement error, and that question has never been answerable.
**The descriptive half is the deliverable; the regression is a formality whose null result must
not be reported as evidence of no effect.**

---

## 4. EXPLORATORY — H2: cancelled profit targets

**Exploratory. Produces no p-value anyone may quote.** Listed so that it is on the record as
exploratory before it is run, rather than presented later as confirmatory.

**Mechanism and why it is not outcome-biased.** 82 of 286 exit limit orders are priced **above**
the true ask at submission: these are resting profit targets, and the limit price is a number he
typed before knowing the outcome. It is the only recorded intent in any data source in this
project. Their fill rate is **44/82 = 53.7%**, versus 182/185 = 98.4% for orders priced at or
below the bid. So roughly half of his stated targets are abandoned.

**Question:** for each of the 38 cancelled targets, did the contract's NBBO subsequently reach
the target price before 15:59 the same session? Target price, placement time and cancellation
time are all recorded ex ante; "did the market get there" is a fact about the world.

**Descriptive statistic:** share of cancelled targets subsequently reachable, and the dollar
difference between the target and what he actually realised. No inference. n=38 supports no
test; any pattern here is a hypothesis for a future confirmatory study on future trades.

---

## 5. Multiplicity, declared now

Confirmatory family = **exactly two** endpoints, P and H1. Holm across those two p-values. H2 is
exploratory and outside the family. The pre-declared sensitivity bands (P: 6 secondaries; H1: 3
alternate windows plus a percent scale) are **not** family members and are reported as bands,
never as tests.

**Specifications examined and discarded before pre-registering**, so the count is on the record:
distance-to-nearest-level (rejected — requires choosing among >= 8 level definitions, an
unbounded fork); time-of-day bucket vs outcome (rejected — 7 buckets, n=34-97 each, guaranteed
to produce one significant cell); VIX/VIX1D regime (rejected — no intraday index entitlement,
would 403); order-type vs outcome (rejected — market/limit is 93% collinear with
executable-on-arrival, so it is one variable not two); first-side-of-day direction accuracy
(rejected — subsumed by H1's sign and underpowered at n~59); averaging-down (rejected — leaks,
per the standing inventory). **Six discarded, two retained.**

## 6. Prohibited

No parameter tuning. No threshold search. No promotion of a secondary, a sensitivity cell or an
exploratory result to primary. No winner/loser split as an endpoint — the percentile and the
outcome are the same object, so conditioning on profitability is circular. No use of
`created_at` where the execution timestamp is the decision time, or vice versa, other than as
specified. No re-specification after a result is seen.
