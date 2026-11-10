# news_arms — discovery results (shares 2018–2022, options 2020–2022)

Status: **discovery read once, 2026-11-10. Exploratory. Holdout not read. Gate 2: pending.**
Every number here is in `analysis_shares_discovery.json`, `analysis_options_discovery.json`,
`analysis_q6_discovery.json`; the tables are rendered from those files by
`trade_analysis/news_arms/report.py` into `results_tables.md` and are not typed by hand.
Pre-registration: `preregistration.md` (gate 1 PASS, owner approved; records in §18–§21).

## The answer in plain words

**The frozen runners do not earn a detectably different return on calendar-release days.**
On the 231 shares sessions with a scheduled release (CPI, jobs, GDP, PCE or a Fed decision)
the shares arm made $143 a day less than its average day, and the 95% confidence interval runs
from −$510 to +$250 a day. The options arm: $94 a day less, interval −$270 to +$90. Neither
is distinguishable from zero. That is the discovery preview of the one confirmatory test (C1,
shares; C2, options) that the holdout will decide.

**One exception, and it is the Fed.** On the 38 shares sessions with a Fed decision the arm
lost **$902 a day more** than its average day (interval −$1,770 to −$30). The options arm on
its 22 Fed sessions: −$295 a day, same sign, interval includes zero. The shares loss happens
**before 14:00**, in the morning trades, not after the announcement. It is below the Fed-day
range finding already settled in `news_impact/results.md` (Fed days move 1.23× a normal
day's range): the runners' stops are hit more, and the day's direction is not what they lose
on. This is the only row in the study that a forward policy experiment could be built on, and
it is not strong: it fails the pre-registered multiple-testing correction.

**Shock days look terrible for the shares arm, but that is the volatility regime, not the
shock.** The 114 geopolitical-surprise sessions lost $1,211 a day more than average. Split by
the trailing-20-day range regime (known before the open, already a condition in this study),
the loss sits entirely in the high-range regime (−0.75 σ) and vanishes in the low-range regime
(+0.05 σ). 2020 alone carries it. A rule "do not trade on shock days" would mostly be a rule
"do not trade when the last month was wild", which the regime column already says.

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
| shares | 1,226 | 222,363 | **−$288** | −$683 | $3,197 | −0.16 bp of $10,000 notional |
| options (ATM, SPY, 0DTE days only) | 482 | 9,943 | **−$100** | −$214 | $1,032 | −$4.8 per contract |

Both arms lose on an ordinary day. "Worse on news" below means "loses more", never "stops
winning". 270 options sessions (every Tuesday and Thursday before 2022-11-14) are excluded for
having no 0DTE expiry, as pre-registered in 14.11.

## Findings, each with its four verdicts (13.9)

### F1. Calendar-release days as a whole (Q3 scheduled; discovery preview of C1/C2)

| arm | n | effect σ | 95% CI | $/day | placebo (matched) | status |
|---|---|---|---|---|---|---|
| shares | 231 | −0.045 | [−0.16, +0.08] | −143 | mean +0.042 ± 0.047, rank 0.965 | tested, BH fail |
| options | 110 | −0.091 | [−0.26, +0.09] | −94 | mean +0.080 ± 0.057, rank 1.000 | tested, BH fail |

- **Statistical:** CI includes zero in both arms. The matched placebo tells a different and
  smaller story: against sessions of the *same year and weekday*, release days sit at the 96th
  (shares) and 100th (options) percentile on the losing side. The placebo centre is above B0
  (+0.04 and +0.08 σ) because the runners lose less on Mondays and Fridays than midweek, and
  most releases fall on Fridays (NFP) and Wednesdays (Fed, CPI). The pre-registered
  confirmatory statistic is the CI against B0, not the matched rank; both are reported, the
  first decides.
- **Effect size:** −0.05 σ and −0.09 σ; −$143 and −$94 a day. Spread over the shares arm's
  181 trades a day that is −0.8 bp per $10,000 trade.
- **Replay fidelity:** both arms passed acceptance (§18: shares ρ 0.932, options 0.953).
- **Practical usefulness:** information set (i), calendar awareness, could feed a pre-open
  rule; 46 sessions a year; the native effect is below one round-trip spread per trade.
  **Not a candidate.**

### F2. Fed decision days (Q1 FOMC)

| arm | n | effect σ | 95% CI | $/day | single-category | placebo rank | BH (family 4) | by year (σ) |
|---|---|---|---|---|---|---|---|---|
| shares | 38 | **−0.282** | **[−0.55, −0.01]** | **−902** | −0.254 [−0.52, +0.03] (n 33) | 0.983 → effect | fail (p 0.043 > 0.025) | 2018 −0.34, 2019 −0.15, 2020 −0.05, 2021 −0.46, 2022 −0.89 |
| options | 22 | −0.286 | [−0.77, +0.22] | −295 | −0.147 [−0.63, +0.37] (n 19) | 0.932 → chance | reported only (n < 30) | 2020 −0.16, 2021 −0.19, 2022 −0.75 |

Metric block, shares, Fed days vs no-event days: win rate per day **29% vs 39%**, expectancy
−6.1 bp vs −0.5 bp per trade, 195 trades per session vs 180. Options: stops 41% of exits vs
35%, return on premium −5.4% vs −0.2%, underlying twin −7.9 bp vs +2.1 bp.

Timing (Q5, descriptive): shares trades entered **before 14:00** on Fed days lost $6.87 each
(6,113 trades); trades after the announcement lost $2.51 (1,293 trades). Options: −$20.49 before,
−$8.86 after. The loss is a morning loss on a day the calendar already named.

Post hoc (not pre-registered, labelled): within the low-range regime, Fed days are still
−0.21 σ [−0.41, −0.01] on shares, so the regime does not explain this one.

- **Statistical:** shares CI excludes zero; fails BH at q = 0.10 in a family of four; same
  sign in all five years and in both arms; overlap-stable. Options underpowered (22 days).
- **Effect size:** −0.28 σ, **−$902 a day on 38 days = −$34,300 over five years**, −6.1 bp
  per trade on Fed days against −0.16 bp per trade on an average day.
- **Replay fidelity:** arm acceptance PASS. The per-setup cells inside it (Q7) include
  VWAP_Reclaim at 79% one-sided replay rate; the arm-level number does not depend on it.
- **Practical usefulness:** information set **(i)**: "a Fed decision is due today" is known
  before the open. Eight sessions a year. A pre-open rule ("shares arm stands aside before
  14:00 on Fed days") would have avoided a loss worth about $7,000 a year on this arm at
  $10,000 per trade. That is an **association** (13.2): the runner never saw the calendar, so
  this study cannot say what the rule would do live. **Candidate for a forward policy
  experiment**, the only one in the study, and the owner may weigh it; it is not a lab change.

### F3. Geopolitical surprise days (Q3 unscheduled; descriptive + placebo, never tested)

| arm | n | effect σ | 95% CI | $/day | placebo rank | low regime | mid | high regime | 2020 |
|---|---|---|---|---|---|---|---|---|---|
| shares | 114 | −0.379 | [−0.64, −0.13] | −1,211 | 1.000 | +0.05 [−0.17, +0.28] | −0.28 | **−0.75 [−1.26, −0.28]** | 27 days, −1.54 σ |
| options | 51 | −0.106 | [−0.42, +0.23] | −110 | 0.818 | +0.14 | +0.36 | −0.24 [−0.65, +0.26] | 20 days, −0.66 σ |

financial_stress alone (never tested by rule): shares −1.30 σ on 17 days, five of them in
2020 at −3.63 σ (March 2020); options −0.34 σ on 12 days.

- **Statistical:** shares CI excludes zero and the placebo rank is 1.000, but the within-regime
  split (post hoc) moves the whole effect into the high-range regime, and one year carries
  it. Options: nothing.
- **Effect size:** large in dollars (−$1,211 a day) and concentrated: 2020 is −$4,900 a day.
- **Replay fidelity:** PASS at arm level; 2020 is outside the acceptance period and the
  shares arm's rules were written on 2026 microstructure (§9).
- **Practical usefulness:** the regime column (trailing-20 range, known pre-open) already
  captures this. A shock rule would add almost nothing to a regime rule. **Not a candidate.**
  The geopolitical list is salience-selected (§9) and this row is descriptive by design.

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
verdicts: no finding.

### F5. Setup × category (Q7)

Shares: 66 tested cells, one BH pass, **PDH_PDL_FailedBreak on Fed days, −0.056 σ [−0.09, −0.02],
−$179 a day**; it is 22% of the 0.25 σ bar. Options: 20 cells, no pass. **Shortlist: empty.**
By 15.7 a setup's cell is its net over the whole arm's σ, so one setup out of fourteen cannot
reach a quarter of the arm's daily variation unless it dominates the book; the bar was set
before the data and stands. The two low-fidelity setups (VWAP_Reclaim 79%, ORB_5min 72%
one-sided replay) appear near the top of the p-value list for Fed days and are labelled.

### F6. First 60 minutes on release days (Q5 rule-shaped line, descriptive)

Shares: −$336 per session on release days vs −$150 on others, difference −$186
[−$577, +$215]. Options: −$144 vs −$56, difference −$88 [−$276, +$109]. Nothing.

### F7. Options vs shares (Q6)

No category shows divergence (both CIs excluding zero with opposite signs). Fed days are
−0.28 σ in both arms; the largest gaps (financial_stress, pandemic_health) are the shares
arm's 2020 losses with the options arm flat.

## Conclusions by information set (13.8)

**(i) Calendar awareness, known before the open.** Release days as a whole: no difference
from baseline. Fed days: the shares arm loses about $900 more, in the morning, with a CI that
excludes zero and a correction it fails. This is the one candidate for a pre-open forward
policy experiment, to be written as its own one-paragraph pre-registration if the owner wants
it, after the holdout and gate 2.

**(ii) Release information, known at the minute.** After-release trades on Fed days lose
less than before-release trades (descriptive). No intraday rule follows from this study.

**(iii) Post-release market reaction.** Never a condition. The twin column says Fed-day
options trades sat on a −7.9 bp average underlying move against +2.1 bp on no-event days: the
arm was wrong-footed on direction *and* paid for convexity. Reported, not used.

## What would make this wrong, and what was checked

- **A session counted twice or a month missing.** `load_replay` raises on a duplicate; 17
  shares chunks and 36 options chunks, one hash each (`7c6414fc549315f3`, `ccd00ac71f8243e9`,
  the live hashes). Missing shares sessions are half days dropped by rule.
- **$0 option days inflating the effects (14.11).** 270 sessions without a 0DTE are excluded
  before σ, B0 and every table; all are Tuesdays and Thursdays before 2022-11-14; the join
  refuses a chunk without `sessions.json` and the tests fail on the previous code.
- **The families changed after seeing P&L.** They did not: the tested cells in the analysis
  equal the cells committed in `counts_*_discovery.json` before any P&L was read (shares 66,
  options 20; H3 4 and 2), checked by set equality.
- **The matched placebo disagreeing with the bootstrap on "scheduled".** Explained above: a
  weekday effect in the baseline. Both statistics are in the JSON; the pre-registered one
  decides.
- **A number that moves when nothing moved.** Seeds fixed (20260828); two runs of the
  analysis produce identical JSON apart from the timestamp.
- **Price-derived conditioning.** `CONDITIONING` lists every column a condition is built
  from; a test asserts none is a price or P&L field (15.4).
- **One year carrying a result.** By-year tables for every selected cell; F3 is 2020, F2 is
  every year.
- **Post hoc additions.** One: the within-regime split, added after the first read because
  F3 demanded it; labelled post hoc in the JSON key, the tables and here.

## Next (section 11, step 6)

1. Gate 2 audit of this report by the statistical auditor; defects corrected in place.
2. Holdout read **once**: `analyze --set holdout` for both arms, which refuses until this
   file carries `GATE 2: PASS`. C1 (shares 2023-01-03 → 2026-08-27) and C2 (options 2023,
   pre-declared underpowered, MDE reported) as one Holm family of two; no Q7 shortlist to
   score.
3. Results published to the Space as exploratory.
