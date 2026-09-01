# Pre-registration — Does his entry timing have forward edge?

**Status: PRE-REGISTERED, NOT YET RUN.** Written 2026-08-26, before any computation.

## 0. Why this is not an eighth repeat of the same null

The seven prior prediction nulls all asked: **does some observable forecast the outcome?**
MACD state, ADX, EMA pullback, RVOL, strike distance, position size, within-day ordinal, a
19-feature ceiling model, his own declared indicator stack, ten external setups. All null.

**None of them asked whether HE forecasts the outcome.** Discretionary skill, if it exists,
lives in screen-reading that no column in this dataset encodes. It is entirely consistent
with the seven nulls for his entry seconds to be better than random seconds on the same day
and for no measurable feature to explain why.

This test asks that question directly, and it is the last question in the programme that has
not been asked.

## 1. Sample

QQQ/SPY round trips from `research/round_trips.csv` that have an ATM contract quote path for
the entry session in `atm_paths.pkl` (197 contracts). Single names excluded — no options
entitlement, and the exclusion is pre-existing, not chosen here.

**Constant-sample rule:** a trade is included only if its entry second leaves **>= 25 minutes**
of forward quote path before 15:45. This makes the sample **identical across all four
horizons**, so horizons are comparable to each other. It drops late-day entries. That
selection uses only the clock, which is known at entry, so it is not lookahead — but it does
narrow the population, and the funnel is reported.

## 2. Instrument

**The ATM contract, not his contract.** ATM = listed strike nearest the underlying spot at
his entry second (median 0.042% from spot in the cached set). This is the instrument he has
said he is willing to trade, so the test measures the thing he would actually do.

Entry at that contract's **ask**. Exit at its **bid** N minutes later. No mid-price anywhere.

## 3. Arms

| arm | definition |
|---|---|
| **HIS** | buy at the ask at his actual entry second; sell at the bid exactly N minutes later |
| **PLACEBO** | **20** seconds drawn uniformly at random from the *same contract's* quote path, within **+/-30 minutes** of his entry, each also leaving 25 min of forward path; each priced ask to bid over the same N; the 20 returns **averaged** |

The placebo is drawn on the **same contract, same day, same expiry, same right**. IV, theta,
spread regime, session, and direction are therefore *identical* between arms. The only thing
that differs is **which second he chose**.

The +/-30 minute window matches time-of-day between arms, so the 0DTE theta profile is
common to both and cannot drive the difference. Averaging 20 placebo draws shrinks the
placebo-arm variance without biasing it.

Seed fixed at 20260826.

## 4. Comparisons and multiplicity

Horizons **N in {5, 10, 15, 25} minutes**, fixed now, taken from the user's own stated
question. No other horizon will be added or reported.

| # | comparison | asks |
|---|---|---|
| C-skill | HIS minus PLACEBO, per horizon (4 tests) | **is his timing better than a coin flip on the same contract** |
| C-money | HIS mean return vs 0, per horizon (4 tests) | **does it make money after spread and theta** |

**Family size m = 8. Holm across all eight.** These are two genuinely different questions and
both are confirmatory, so both are in the family. No test is demoted to "secondary" after
seeing a result.

**Statistic:** mean paired difference (C-skill) or mean level (C-money), **day-clustered
bootstrap, 10,000 reps**, resampling activity dates with replacement.

## 5. Dollar outcome

Reported at **1 contract per trade** — mean $ per trade, and $/week implied by his observed
trade rate over the sample window. The user's stated objective is $200-500/week, so a result
that is statistically real but worth $12/week is a **negative answer to his actual question**
and will be reported as one.

## 6. Power, committed now

The achieved MDE is computed from the realised bootstrap SE and reported next to every
effect. **If |effect| < MDE the result is labelled underpowered regardless of its p-value.**
This is the same rule applied in the construction tests and it is not relaxed here.

## 7. Falsifiers, committed now

| outcome | conclusion |
|---|---|
| C-skill CI includes 0 at **all four** horizons | **his entry timing shows no measurable skill.** His positive tape is exit management, not entry selection |
| C-skill > 0 clearing Holm at any horizon | his entry timing IS better than a same-contract coin flip — the first positive prediction result in the programme |
| C-money < 0 at all horizons | a mechanical fixed-hold on his entries loses money; only his discretionary exit rescues it |
| C-money > 0 clearing Holm | a fully mechanical rule on his own entries is profitable |

C-skill and C-money can disagree, and that combination is informative rather than awkward:
**skill > 0 with money < 0** means his read is real but the 0DTE structure eats it, which
points at the instrument, not at him.

## 8. Mechanism read-out — DESCRIPTIVE, NOT TESTED

The **signed underlying move** (in his traded direction, in that day's sigma) over each
horizon, at his entries and at placebo entries, is reported as **point estimates with no
p-values and no CIs**. It decomposes an option-level result into direction versus
theta/spread. It is deliberately given no inferential apparatus so that it cannot be
retrofitted into a claim. It is **not** in the Holm family and **no conclusion in section 7
may be revised on the basis of it.**

## 9. Leak audit

| | |
|---|---|
| ATM strike selection | underlying spot at the entry second — known at entry |
| placebo seconds | drawn from the clock only, never from outcomes |
| forward path | the *mechanism* being measured, applied identically to every trade and every placebo, with no per-trade selection |
| exit rule | a fixed clock offset. No path-dependent exit, so no policy can peek |
| outcome selection | none. Every qualifying trade enters every arm at every horizon |
| quote validity | bid > 0, ask > 0, ask >= bid; 09:31-15:59 |
| survivorship | his entry seconds come from filled orders. Unfilled limit intents are absent, which is a known and previously documented limitation (66 cancels, ~6-7 genuine abstentions) |

## 10. Known limitations, stated before the result

- **In-sample.** The cached window is May-Nov 2025, the same window as the construction
  tests. A positive C-skill result here is a hypothesis for out-of-sample testing, **not a
  validated edge**, and will be labelled as such.
- **Fill realism.** HIS is priced at the NBBO ask at his entry second, not his realised fill.
  Both arms face the same assumption, so the *difference* is fair; the *level* (C-money) is
  an approximation of what a market order would get.
- **His entries are not independent of his exits.** He may enter more aggressively when he
  intends to manage actively. This test cannot separate those.

## 11. Prohibited

No horizon added, removed, or re-reported after seeing results. No placebo window changed.
No per-symbol, per-year, or per-time-of-day cell promoted to primary. No switch from ATM to
his actual strike. Family size stays 8 even if a test fails to execute.

## 12. Stated prior

Given seven prediction nulls I expect **C-skill near zero**. I expect **C-money negative at
25 minutes** — midday 0DTE ATM theta plus a paid spread is a steep hurdle — and I am
genuinely uncertain at 5 minutes, where the spread is the dominant term and the sign could
go either way. I am recording this so that a null cannot later be described as expected all
along, and a positive cannot be described as predicted.

---

## AMENDMENT 2026-08-26 (post-run) — C-skill VOIDED for lookahead in the control arm

Appended after execution. **Nothing above this line has been edited.**

**Defect.** Section 3 specifies the placebo as random seconds on "the same contract." The
contract is the ATM strike **selected from spot at HIS entry second T**. A placebo entry at
`i < T` therefore holds a strike chosen with information from `T > i`. Since his entries are
momentum chases, the underlying reliably travelled toward that strike over `[i, T]`, handing
every pre-entry placebo a free ride.

Section 9 claimed "placebo seconds drawn from the clock only, never from outcomes." That was
true of the **seconds** and false of the **contract**. The leak audit checked the wrong object.

**Detection.** The placebo arm returned +27.12% mean / 74.3% win for a 25-minute 0DTE ATM
hold — implausible on its face. A pre/post split confirmed it: `pre - HIS` p = 0.0003 at all
four horizons, `post - HIS` p = 0.12/0.12/0.58/0.71, null at all four.

**Consequence.** All four **C-skill** comparisons are **VOID**. The apparent "3 of 8 clear"
is retracted and must not be cited.

**C-money is unaffected** — it uses only his entry second and a strike selected at that same
second. Its four results stand as reported, corrected at the pre-registered **m = 8**.

**Family size stays 8** per section 11. It is not reduced because members were voided;
lowering the bar after a sibling fails is not legitimate, and here the failure is my own.

**The post-entry-only comparison is EXPLORATORY.** It is a post-hoc restriction of the
control adopted to remove a leak. It may invalidate the pre-registered hypothesis, which it
does. It may **not** confirm one. Any confirmatory claim about entry-timing skill requires a
fresh pre-registration on out-of-sample data.

**Correct design for any future attempt.** Select the strike from spot at the **placebo**
entry second, not his — i.e. each placebo buys the contract that was ATM *for that trader at
that moment*. This requires a chain fetch per placebo second and was not affordable under the
cached data. Note that the Option Data subscription cancels 2026-09-05.
