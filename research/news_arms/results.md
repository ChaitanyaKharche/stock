# news_arms — discovery results (shares 2018–2022, options 2020–2022)

Status: **discovery read 2026-11-10. Exploratory. Holdout not read. Gate 2: FAIL on the
first pass (`audit_results.md`, 27 rows), corrected in place below; re-audit pending.**
Every number here is in `analysis_shares_discovery.json`, `analysis_options_discovery.json`,
`analysis_q6_discovery.json`; the tables are rendered from those files by
`trade_analysis/news_arms/report.py` into `results_tables.md` and are not typed by hand.
Pre-registration: `preregistration.md` (gate 1 PASS, owner approved; records in §18–§21).

**Open defect, found at gate 2:** 23 regular 2022 shares sessions are missing from the
replay. During a vendor outage the feed returned zero bars and `backfill.run` logged each as
"not a session"; the chunks exited 0. They are 2022-05-13, 07-15 → 07-25 (7), 08-04 → 08-09
(4), 10-03 → 10-12 (8) and 11-01 → 11-03 (3), and they include the Fed day 2022-11-02 and the
jobs reports of 2022-08-05 and 2022-10-07. Every shares number below rests on **1,226 of
1,249** regular sessions; the direction of the bias is unknown until they are re-replayed.
Fix landed the same day: a weekday with no bars that the NYSE calendar says was open now
raises and fails the chunk (`backfill.SessionLost`, tests fail on the previous code), every
skip is recorded in `RUN.json` with its reason, and `replay_chunks --dates` replays an
explicit list into a fill-in chunk. The options replay has no lost session (270 no-0DTE days
are excluded by rule and listed; 4 half days).

## The answer in plain words

**The frozen runners do not earn a detectably different return on calendar-release days.**
On the 231 shares sessions with a scheduled release (CPI, jobs, GDP, PCE or a Fed decision)
the shares arm made $143 a day less than its average day, and the 95% confidence interval runs
from −$510 to +$250 a day. The options arm: $94 a day less, interval −$270 to +$90. Neither
is distinguishable from zero. That is the discovery preview of the one confirmatory test (C1,
shares; C2, options) that the holdout will decide.

**One exception, and it is the Fed.** On the 38 shares sessions with a Fed decision the arm
lost **$902 a day more** than its average day (interval −$1,770 to −$30). The options arm on
its 22 Fed sessions: −$295 a day, same sign, interval includes zero. Of the shares excess,
$831 a day is incurred by trades entered **before 14:00**, so the loss precedes the
announcement. It sits beside the Fed-day range finding already settled in
`news_impact/results.md` (Fed days move 1.23× a normal day's range). It is the only row in
the study that a forward policy experiment could be built on, and it is not strong: it fails
the pre-registered multiple-testing correction, 2022 carries half of it, and a block
bootstrap that respects volatility clustering puts the upper bound at zero.

**Shock days are bad for the shares arm, and the volatility regime does not fully explain
it.** The 135 sessions with a geopolitical surprise lost $1,116 a day more than average.
Within the high trailing-range regime alone they still lose 0.72 σ more than other
high-regime days (CI [−1.13, −0.33]); in the low regime there is nothing (+0.11 σ), so the
row is **unstable** by the §2 rule. 2020 carries most of it: within the high regime without
2020 the effect is −0.41 σ with a CI that crosses zero. Geopolitical rows are descriptive by
design and this one stays descriptive.

**Nothing per setup survives.** 66 setup × category cells on shares, 20 on options; one passes
Benjamini–Hochberg (PDH_PDL_FailedBreak on Fed days, −$179 a day) and it is a fifth of the
pre-registered quarter-sigma bar. The shortlist for the holdout is **empty**. The holdout read
will consist of C1 and C2 and their sensitivity lines, nothing else.

**No arm diverges from the other** on any category (Q6).

Terms, once: *σ* is one standard deviation of the arm's daily net over discovery ($3,197 a
day for shares, $1,032 for options); *effect* is the condition's mean daily net minus the
mean over all sessions, in σ; *B0* is that all-sessions mean; *CI* is a 95% interval from
resampling whole sessions (day-clustered); *placebo rank* is where the real effect sits among
5,000 random conditions of the same size matched by year and weekday.

## Baseline first (section 5)

| arm | sessions | trades | B0 mean/day | B0 median/day | σ/day | per trade |
|---|---|---|---|---|---|---|
| shares | 1,226 | 222,363 | **−$288** | −$683 | $3,197 | −$1.59 = −1.6 bp of $10,000 notional |
| options (ATM, SPY, 0DTE days only) | 482 | 9,943 | **−$100** | −$214 | $1,032 | −$4.9 per contract |

Both arms lose on an ordinary day. "Worse on news" below means "loses more", never "stops
winning". 270 options sessions, Tuesdays and Thursdays without a 0DTE expiry before
2022-11-14, are excluded as pre-registered in 14.11; 39 Tuesdays and Thursdays that did have
an expiry (quarter ends, and every one from 2022-11-15) are kept.

## Findings, each with its four verdicts (13.9)

### F1. Calendar-release days as a whole (Q3 scheduled; discovery preview of C1/C2)

| arm | n | effect σ | 95% CI | block-10 CI (post hoc) | $/day | placebo rank, pre-reg. | placebo rank, condition in pool (sens.) | §2 |
|---|---|---|---|---|---|---|---|---|
| shares | 231 | −0.045 | [−0.16, +0.08] | [−0.15, +0.07] | −143 | 0.965 → "effect" | 0.854 → chance | unstable |
| options | 110 | −0.091 | [−0.26, +0.09] | [−0.24, +0.07] | −94 | 1.000 → "effect" | 0.925 → chance | unstable |

- **Statistical:** CI includes zero in both arms. The pre-registered placebo (13.3: the
  condition's own sessions left out of the pool) says "effect" at the 96th and 100th
  percentile, but that pool rule is biased: leaving the release days out of each year ×
  weekday pool pushes the null away from the real effect. The auditor measured the weekday
  mix as worth +0.012 σ of the +0.042 σ placebo shift (shares); the rest is the exclusion.
  With the condition left in the pool (a sensitivity, not pre-registered) the ranks fall to
  0.854 and 0.925: chance. The pre-registered confirmatory statistic is the CI against B0;
  it decides, and it says nothing. The sign also flips across regimes (mid +0.11 σ), so the
  row is unstable by §2.
- **Effect size:** −0.05 σ and −0.09 σ; −$143 and −$94 a day; −0.8 bp per $10,000 shares
  trade against a −1.6 bp average trade.
- **Replay fidelity:** both arms passed acceptance (§18: shares ρ 0.932, options 0.953).
- **Practical usefulness:** information set (i) could feed a pre-open rule; 46 sessions a
  year; the native effect is below one round-trip spread. **Not a candidate.**

### F2. Fed decision days (Q1 FOMC)

| arm | n | effect σ | 95% CI | block-10 CI | $/day | single-category | placebo (pre-reg. / in-pool) | BH (family 4) | §2 |
|---|---|---|---|---|---|---|---|---|---|
| shares | 38 | **−0.282** | **[−0.55, −0.01]** | [−0.54, 0.00] | **−902** | −0.254 [−0.52, +0.03] (n 33) | 0.983 / 0.960 → effect | fail (p 0.043 > 0.025) | stable |
| options | 22 | −0.286 | [−0.77, +0.22] | [−0.75, +0.18] | −295 | −0.147 [−0.63, +0.37] (n 19) | 0.932 / 0.866 → chance | reported only (n < 30) | stable |

By year, effect against that year's own baseline (σ): shares 2018 −0.29, 2019 −0.09,
**2020 +0.18**, 2021 −0.40, 2022 −0.84 — four of five years negative, not five. Without 2022
the effect halves to −0.16 σ [−0.30, −0.00]. One 2022 Fed session (11-02) is among the lost
23. Options: 2020 −0.02, 2021 −0.14, 2022 −0.64.

Metric block, shares, Fed days vs no-event days: win rate per day **29% vs 39%**, expectancy
−6.1 bp vs −0.5 bp per trade (3.8× the −1.6 bp average trade), 195 trades per session vs
180. Options: stops 41% of exits vs 35%, return on premium −5.4% vs −0.2%, underlying twin
−7.9 bp vs +2.1 bp.

Timing (Q5, descriptive, 14.4): on Fed days the shares arm's trades entered before 14:00
earned −$1,105 per session against −$275 on all sessions, an excess of **−$831 per session**;
trades entered after 14:00 lost $2.51 each (1,293 trades) against $6.87 for the morning
trades (6,113). The money is lost in the morning of a day the calendar had already named.

- **Statistical:** shares CI excludes zero on the pre-registered bootstrap; fails BH at
  q = 0.10 in a family of four; the post hoc block bootstrap touches zero; same sign in four
  of five years and in both arms; overlap-stable; stable by §2. Options underpowered (22 days).
- **Effect size:** −0.28 σ, **−$902 a day on 38 days = −$34,300 over five years**, −6.1 bp
  per trade on Fed days.
- **Replay fidelity:** arm acceptance PASS. Among the per-setup cells inside it (Q7),
  VWAP_Reclaim (79% one-sided replay) is second by p-value and ORB_5min (72%) seventh of
  fifteen; the arm-level number does not depend on either.
- **Practical usefulness:** information set **(i)**: "a Fed decision is due today" is known
  before the open. Eight sessions a year. A rule "shares arm stands aside before 14:00 on
  Fed days" would have forgone about $8,400 a year of raw morning loss, of which about
  **$6,300 a year is the Fed-specific excess** (−$831 × 38 ÷ 5); the whole-day excess is
  about $6,900 a year. That is an **association** (13.2): the runner never saw the calendar,
  so this study cannot say what the rule would do live. **Candidate for a forward policy
  experiment**, the only one in the study, and the owner may weigh it; it is not a lab change.

### F3. Geopolitical surprise days (Q3 unscheduled; descriptive + placebo, never tested)

Definition per the pre-registration text: every session carrying a `surprise = yes`
geopolitical row (135 shares, 62 options). The first version of this report used only the
sessions without a scheduled release as well (114 / 51); that variant is kept as a
sensitivity (−0.379 σ shares, −0.106 options).

| arm | n | effect σ | 95% CI | block-10 CI | $/day | placebo (pre-reg. / in-pool) | low regime | mid | high regime | §2 |
|---|---|---|---|---|---|---|---|---|---|---|
| shares | 135 | −0.349 | [−0.58, −0.13] | [−0.67, −0.09] | −1,116 | 1.000 / 1.000 | +0.11 | −0.26 | **−0.72 [−1.13, −0.33]** | unstable |
| options | 62 | −0.104 | [−0.39, +0.20] | [−0.39, +0.21] | −107 | 0.862 / 0.770 → chance | +0.36 | +0.17 | −0.26 | unstable |

By year, shares, effect vs that year's baseline: 2018 −0.05, 2019 +0.06, **2020 −0.99** (35
sessions), 2021 −0.02, 2022 −0.42. Within the high regime without 2020: −0.41 σ
[−0.84, +0.06], 33 sessions. financial_stress alone (never tested by rule): shares −1.30 σ on
17 days, five of them in 2020 at −3.63 σ (March 2020); options −0.34 σ on 12 days.

- **Statistical:** shares CI excludes zero under both bootstraps and both placebo rules.
  Removing each regime's own average moves the effect only from −0.35 to about −0.36 σ, so
  the regime is **not** the explanation the first version of this report gave; the effect is
  real within the high regime. It is, however, concentrated in 2020 and absent in the low
  regime, which makes it unstable by §2.
- **Effect size:** −0.35 σ, −$1,116 a day; 2020 alone −$3,170 a day.
- **Replay fidelity:** PASS at arm level; 2020 is outside the acceptance period and the
  shares arm's rules were written on 2026 microstructure (§9).
- **Practical usefulness:** a shock is known only when it breaks; 56 of these 135 sessions
  carry a break time inside the session (37) or an unknown time (19), so the information set is mixed
  between (i) overnight and (ii) intraday. The geopolitical list is salience-selected (§9)
  and the category is descriptive by design. **Not a candidate.**

### F4. CPI, jobs, GDP, PCE days (Q1)

| category | shares n | shares σ | shares CI | options n | options σ | options CI |
|---|---|---|---|---|---|---|
| CPI | 60 | +0.092 | [−0.17, +0.41] | 19 | −0.327 | [−0.70, +0.06] |
| NFP | 58 | +0.062 | [−0.15, +0.28] | 36 | −0.100 | [−0.39, +0.22] |
| GDP_ADV | 20 | −0.204 | [−0.59, +0.17] | 1 | — | — |
| PCE | 59 | −0.076 | [−0.33, +0.21] | 34 | +0.163 | [−0.21, +0.56] |

Every CI includes zero; every placebo verdict is "selection / chance". The one line worth a
glance is options on CPI days: 19 days, return on premium −13.9% against −0.2% on no-event
days, win rate per day 16%; it is under the 30-session minimum and reported only. All four
verdicts: no finding. Two of the lost 2022 sessions are jobs days (08-05, 10-07).

### F5. Setup × category (Q7)

Shares: 66 tested cells, one BH pass, **PDH_PDL_FailedBreak on Fed days, −0.056 σ [−0.09, −0.02],
−$179 a day**; it is 22% of the 0.25 σ bar. Options: 20 cells, no pass. **Shortlist: empty.**
By 15.7 a setup's cell is its net over the whole arm's σ, so one setup out of fifteen cannot
reach a quarter of the arm's daily variation unless it dominates the book; the bar was set
before the data and stands. The two low-fidelity setups (VWAP_Reclaim 79%, ORB_5min 72%
one-sided replay) are labelled wherever they appear.

### F6. First 60 minutes on release days (Q5 rule-shaped line, descriptive)

Shares: −$336 per session on release days vs −$150 on others, difference −$186
[−$577, +$215]. Options: −$144 vs −$56, difference −$88 [−$276, +$109]. Nothing.

### F7. Options vs shares (Q6)

No category shows divergence (both CIs excluding zero with opposite signs). Fed days are
−0.28 σ in both arms; the largest gaps (financial_stress, pandemic_health) are the shares
arm's 2020 losses with the options arm flat.

## Conclusions by information set (13.8)

**(i) Calendar awareness, known before the open.** Release days as a whole: no difference
from baseline. Fed days: the shares arm loses about $900 more, $830 of it in the morning,
with a CI that excludes zero on the pre-registered bootstrap and a correction it fails. This
is the one candidate for a pre-open forward policy experiment, to be written as its own
one-paragraph pre-registration if the owner wants it, after the holdout and gate 2.

**(ii) Release information, known at the minute.** After-release trades on Fed days lose
less than before-release trades (descriptive). No intraday rule follows from this study.

**(iii) Post-release market reaction.** Never a condition. The twin column says Fed-day
options trades sat on a −7.9 bp average underlying move against +2.1 bp on no-event days: the
arm was wrong-footed on direction *and* paid for convexity. Reported, not used.

## What would make this wrong, and what was checked

- **A session counted twice or a session lost.** `load_replay` raises on a duplicate; 49
  shares chunks and 36 options chunks, one hash each (`7c6414fc549315f3`, `ccd00ac71f8243e9`,
  the live hashes). **23 shares sessions WERE lost** (top of this file); the calendar check
  that found them (`analyze.exclusions`, NYSE closures by rule) now runs on every analysis
  and lists every open weekday that is not a replayed session.
- **$0 option days inflating the effects (14.11).** 270 sessions without a 0DTE are excluded
  before σ, B0 and every table; all are Tuesdays and Thursdays before 2022-11-14; the join
  refuses a chunk without `sessions.json` and the tests fail on the previous code.
- **The families changed after seeing P&L.** They did not: the tested cells in the analysis
  equal the cells committed in `counts_*_discovery.json` before any P&L was read (shares 66,
  options 20; H3 4 and 2), checked by set equality.
- **The placebo as pre-registered favours "effect".** Confirmed at gate 2 and shown above;
  the in-pool sensitivity is beside every pre-registered rank. The Fed verdict survives it;
  the release-days verdict does not.
- **Serial dependence.** Daily z has lag-1 autocorrelation of about −0.03, |z| about +0.31
  (volatility clustering). A 10-session block bootstrap (post hoc) is beside every cell; it
  widens the Fed CI to touch zero and leaves the shock-day CI excluding zero.
- **A number that moves when nothing moved.** Seeds fixed (20260828); two runs of the
  analysis produce identical JSON apart from the timestamp.
- **Price-derived conditioning.** `CONDITIONING` lists every column a condition is built
  from and a test asserts none is a price or P&L field (15.4); the end-to-end tests run the
  tables on rows that carry no price field at all, which is the stronger check. Regimes use
  the 20 prior sessions only (auditor: 0 mismatches in 2,676) with discovery terciles.
- **One year carrying a result.** By-year effects against each year's own baseline are in
  every selected cell; F3 is 2020, F2 is four of five years with 2022 as half.
- **Post hoc additions, all labelled:** the within-regime split, the block bootstrap, the
  in-pool placebo, the Fed pre-14:00 excess, the inclusive/exclusive shock-day variants.
- **The holdout lock.** The first version matched a substring of this file and was open
  while the file described it (gate 2). It now requires the whole line `GATE 2 VERDICT: PASS`
  in the auditor's `audit_results.md`; a test asserts it is shut on the committed files.

## Next (section 11, step 6)

1. Re-replay the 23 lost sessions on the laptop into a fill-in chunk
   (`replay_chunks --arm shares --dates ... --label 2022 --chunk fill-a`), rerun the analysis,
   re-render, and record what moved.
2. Second gate 2 pass; defects corrected in place.
3. Holdout read **once**: `analyze --set holdout` for both arms, which refuses until
   `audit_results.md` carries the pass line. C1 (shares 2023-01-03 → 2026-08-27) and C2
   (options 2023, pre-declared underpowered, MDE reported) as one Holm family of two; no Q7
   shortlist to score.
4. Results published to the Space as exploratory.
