# news_arms — discovery results (shares 2018–2022, options 2020–2022)

Status: **discovery read 2026-11-10; the 24 lost shares sessions re-replayed and folded in
2026-11-11. Holdout read once on 2026-11-11 after gate 2: C1 FAIL, C2 FAIL (underpowered),
Fed-day candidate did not replicate; holdout section awaits its own gate 2 audit. Gate 2: PASS on the fifth pass, 2026-11-11
(`audit_results.md`); passes one to four failed on 27 rows, three sentences, one clause and one
word, each corrected in place and recorded in §22–§23 of the pre-registration.**
Every number here is in `analysis_shares_discovery.json`, `analysis_options_discovery.json`,
`analysis_q6_discovery.json`; the tables are rendered from those files by
`trade_analysis/news_arms/report.py` into `results_tables.md` and are not typed by hand.
Pre-registration: `preregistration.md` (gate 1 PASS, owner approved; records in §18–§23).

**Defect found at gate 2, now closed.** 23 regular 2022 shares sessions had been missing
from the replay: during a vendor outage the feed returned zero bars and `backfill.run` logged
each as "not a session"; the chunks exited 0. They included the Fed day 2022-11-02 and the
jobs reports of 2022-08-05 and 2022-10-07; one more, 2024-10-01, was missing from the
holdout replay. All 24 were re-replayed on the laptop on 2026-11-10 into fill-in chunks
(`2022/shares/fill-a`, `2024/shares/fill-a`), at the same config hash, and the analysis was
rerun. **Shares discovery now rests on 1,249 of 1,249 regular sessions.** What moved: Fed
days went from 38 to 39 sessions and from −0.282 σ to −0.299 σ; release days from −0.045 to
−0.039 σ, and their placebo verdict under the pre-registered rule went from "effect" to
"chance"; one more Q7 cell passes BH; no Q1 or Q3 effect changed sign (three tested Q7 setup
cells within ±0.005 σ of zero did: Crabel_Stretch and ORB_5min on jobs days, PDH_PDL_Breakout
on Fed days, every p > 0.8; four untested Q7 cells within ±0.002 σ also flipped) and no
tested/reported status changed. The replay cannot lose
a session silently any more: a weekday with no bars that the NYSE calendar says was open
raises and fails the chunk (`backfill.SessionLost`, tests fail on the previous code), every
skip is recorded in `RUN.json` with its reason, and the analysis lists every open weekday
that is not a replayed session.

## The answer in plain words

**The frozen runners do not earn a detectably different return on calendar-release days.**
On the 234 shares sessions with a scheduled release (CPI, jobs, GDP, PCE or a Fed decision)
the shares arm made $123 a day less than its average day, and the 95% confidence interval runs
from −$495 to +$257 a day. The options arm: $94 a day less, interval −$270 to +$90. Neither
is distinguishable from zero. That is the discovery preview of the one confirmatory test (C1,
shares; C2, options) that the holdout will decide.

**One exception, and it is the Fed.** On the 39 shares sessions with a Fed decision the arm
lost **$958 a day more** than its average day (interval −$1,740 to −$100). The options arm on
its 22 Fed sessions: −$295 a day, same sign, interval includes zero. Of the shares excess,
$874 a day is incurred by trades entered **before 14:00**, so the loss precedes the
announcement. It sits beside the Fed-day range finding already settled in
`news_impact/results.md` (Fed days move 1.23× a normal day's range). It is the only row in
the study that a forward policy experiment could be built on, and it is not strong: it fails
the pre-registered multiple-testing correction (p 0.030 against a bar of 0.025), 2022 carries
half of it, and 2020 goes the other way.

**Shock days are bad for the shares arm, and the volatility regime does not fully explain
it.** The 137 sessions with a geopolitical surprise lost $1,069 a day more than average.
Within the high trailing-range regime alone they still lose 0.67 σ more than other
high-regime days (CI [−1.08, −0.26]); in the low regime there is nothing (+0.11 σ), so the
row is **unstable** by the §2 rule. 2020 carries most of it: within the high regime without
2020 the effect is −0.35 σ with a CI that crosses zero. Geopolitical rows are descriptive by
design and this one stays descriptive.

**Nothing per setup survives.** 66 setup × category cells on shares, 20 on options; two pass
Benjamini–Hochberg on shares (PDH_PDL_FailedBreak on Fed days, −$189 a day; VWAP_Reclaim on
Fed days, +$157 a day, a low-fidelity setup) and both are a fifth of the pre-registered
quarter-sigma bar. The shortlist for the holdout is **empty**. The holdout read will consist
of C1 and C2 and their sensitivity lines, nothing else.

**No arm diverges from the other** on any category (Q6).

Terms, once: *σ* is one standard deviation of the arm's daily net over discovery ($3,199 a
day for shares, $1,032 for options); *effect* is the condition's mean daily net minus the
mean over all sessions, in σ; *B0* is that all-sessions mean; *CI* is a 95% interval from
resampling whole sessions (day-clustered); *placebo rank* is where the real effect sits among
5,000 random conditions of the same size matched by year and weekday.

## Baseline first (section 5)

| arm | sessions | trades | B0 mean/day | B0 median/day | σ/day | per trade |
|---|---|---|---|---|---|---|
| shares | 1,249 | 226,678 | **−$286** | −$682 | $3,199 | −$1.57 = −1.6 bp of $10,000 notional |
| options (ATM, SPY, 0DTE days only) | 482 | 9,943 | **−$100** | −$214 | $1,032 | −$4.9 per contract |

Both arms lose on an ordinary day. "Worse on news" below means "loses more", never "stops
winning". 270 options sessions, Tuesdays and Thursdays without a 0DTE expiry before
2022-11-14, are excluded as pre-registered in 14.11; 39 Tuesdays and Thursdays that did have
an expiry are kept: 13 from 2022-11-15 on, 21 next to a weekday market holiday (the
Mon/Wed/Fri expiry moved), and 5 quarter ends. Shares: 10 early-close half
days are dropped by rule (listed in the JSON), no other exclusion.

## Findings, each with its four verdicts (13.9)

### F1. Calendar-release days as a whole (Q3 scheduled; discovery preview of C1/C2)

| arm | n | effect σ | 95% CI | block-10 CI (post hoc) | $/day | placebo rank, pre-reg. | placebo rank, condition in pool (sens.) | §2 |
|---|---|---|---|---|---|---|---|---|
| shares | 234 | −0.039 | [−0.15, +0.08] | [−0.14, +0.09] | −123 | 0.942 → chance | 0.815 → chance | unstable |
| options | 110 | −0.091 | [−0.26, +0.09] | [−0.24, +0.07] | −94 | 1.000 → "effect" | 0.925 → chance | unstable |

- **Statistical:** CI includes zero in both arms. The pre-registered placebo (13.3: the
  condition's own sessions left out of the pool) is centred above B0 (+0.035 ± 0.046 σ on
  shares) and favours "effect": leaving the release days out of each year × weekday pool
  pushes the null away from the real effect. The auditor measured the weekday mix as worth
  about a quarter of that shift; the rest is the exclusion. With the condition left in the
  pool (a sensitivity, not pre-registered) both arms read chance; after the fill-in the
  shares arm reads chance under the pre-registered rule as well. The confirmatory statistic
  is the CI against B0; it decides, and it says nothing. The sign flips across regimes
  (mid +0.11 σ), so the row is unstable by §2.
- **Effect size:** −0.04 σ and −0.09 σ; −$123 and −$94 a day; −0.7 bp per $10,000 shares
  trade against a −1.6 bp average trade.
- **Replay fidelity:** both arms passed acceptance (§18: shares ρ 0.932, options 0.953).
- **Practical usefulness:** information set (i) could feed a pre-open rule; 47 sessions a
  year; the native effect is below one round-trip spread. **Not a candidate.**

### F2. Fed decision days (Q1 FOMC)

| arm | n | effect σ | 95% CI | block-10 CI | $/day | single-category | placebo (pre-reg. / in-pool) | BH (family 4) | §2 |
|---|---|---|---|---|---|---|---|---|---|
| shares | 39 | **−0.299** | **[−0.54, −0.03]** | [−0.56, −0.03] | **−958** | −0.273 [−0.54, +0.01] (n 34) | 0.985 / 0.961 → effect | fail (p 0.030 > 0.025) | stable |
| options | 22 | −0.286 | [−0.77, +0.22] | [−0.75, +0.18] | −295 | −0.147 [−0.63, +0.37] (n 19) | 0.932 / 0.866 → chance | reported only (n < 30) | stable |

By year, effect against that year's own baseline (σ): shares 2018 −0.29, 2019 −0.09,
**2020 +0.18**, 2021 −0.40, 2022 −0.86 — four of five years negative, not five. Without 2022
the effect halves to −0.16 σ [−0.30, −0.00]. The recovered Fed session 2022-11-02 (−$3,248)
is in these numbers. Options: 2020 −0.02, 2021 −0.14, 2022 −0.64.

Metric block, shares, Fed days vs no-event days: win rate per day **28% vs 40%**, expectancy
−6.4 bp vs −0.5 bp per trade (4× the −1.6 bp average trade), 195 trades per session vs
180. Options: stops 41% of exits vs 35%, return on premium −5.4% vs −0.2%, underlying twin
−7.9 bp vs +2.1 bp.

Timing (Q5, descriptive, 14.4): on Fed days the shares arm's trades entered before 14:00
earned −$1,145 per session against −$272 on all sessions, an excess of **−$874 per session**;
trades entered after 14:00 lost $2.87 each (1,331 trades) against $7.11 for the morning
trades (6,283). The money is lost in the morning of a day the calendar had already named.

Post hoc (not pre-registered, labelled): within the low-range regime, Fed days are still
−0.21 σ [−0.41, −0.01] on shares, so the regime does not explain this one.

- **Statistical:** shares CI excludes zero on the pre-registered bootstrap and on the block
  bootstrap; fails BH at q = 0.10 in a family of four; same sign in four of five years and
  in both arms; overlap-stable; stable by §2. Options underpowered (22 days).
- **Effect size:** −0.30 σ, **−$958 a day on 39 days = −$37,400 over five years**, −6.4 bp
  per trade on Fed days.
- **Replay fidelity:** arm acceptance PASS. Among the per-setup cells inside it (Q7),
  VWAP_Reclaim (79% one-sided replay) is second by p-value and ORB_5min (72%) sixth of the
  14 tested Fed-day cells; the arm-level number does not depend on either.
- **Practical usefulness:** information set **(i)**: "a Fed decision is due today" is known
  before the open. Eight sessions a year. A rule "shares arm stands aside before 14:00 on
  Fed days" would have forgone about $8,900 a year of raw morning loss, of which about
  **$6,800 a year is the Fed-specific excess** (−$874 × 39 ÷ 5); the whole-day excess is
  about $7,500 a year. That is an **association** (13.2): the runner never saw the calendar,
  so this study cannot say what the rule would do live. **Candidate for a forward policy
  experiment**, the only one in the study, and the owner may weigh it; it is not a lab change.

### F3. Geopolitical surprise days (Q3 unscheduled; descriptive + placebo, never tested)

Definition per the pre-registration text: every session carrying a `surprise = yes`
geopolitical row (137 shares, 62 options). The first version of this report used only the
sessions without a scheduled release (now 115 / 51); that variant is kept as a sensitivity
(−0.381 σ shares, −0.106 options).

| arm | n | effect σ | 95% CI | block-10 CI | $/day | placebo (pre-reg. / in-pool) | low regime | mid | high regime | §2 |
|---|---|---|---|---|---|---|---|---|---|---|
| shares | 137 | −0.334 | [−0.57, −0.12] | [−0.66, −0.07] | −1,069 | 1.000 / 1.000 | +0.11 | −0.26 | **−0.67 [−1.08, −0.26]** | unstable |
| options | 62 | −0.104 | [−0.39, +0.20] | [−0.39, +0.21] | −107 | 0.862 / 0.770 → chance | +0.36 | +0.17 | −0.26 | unstable |

By year, shares, effect vs that year's baseline: 2018 −0.05, 2019 +0.06, **2020 −0.99** (35
sessions), 2021 −0.02, 2022 −0.34. Within the high regime without 2020: −0.35 σ
[−0.78, +0.10], 35 sessions. financial_stress alone (never tested by rule): shares −1.30 σ on
17 days, five of them in 2020 at −3.40 σ against that year's baseline (March 2020); options
−0.34 σ on 12 days.

- **Statistical:** shares CI excludes zero under both bootstraps and both placebo rules.
  The effect is real within the high regime, so "the regime, not the shock" (the first
  version's claim) was wrong. It is, however, concentrated in 2020 and absent in the low
  regime, which makes it unstable by §2.
- **Effect size:** −0.33 σ, −$1,069 a day; 2020 alone −$3,170 a day.
- **Replay fidelity:** PASS at arm level; 2020 is outside the acceptance period and the
  shares arm's rules were written on 2026 microstructure (§9).
- **Practical usefulness:** a shock is known only when it breaks; 56 of these sessions
  carry a break time inside the session (37) or an unknown time (19), so the information set
  is mixed between (i) overnight and (ii) intraday. The geopolitical list is
  salience-selected (§9) and the category is descriptive by design. **Not a candidate.**

### F4. CPI, jobs, GDP, PCE days (Q1)

| category | shares n | shares σ | shares CI | options n | options σ | options CI |
|---|---|---|---|---|---|---|
| CPI | 60 | +0.091 | [−0.18, +0.39] | 19 | −0.327 | [−0.70, +0.06] |
| NFP | 60 | +0.100 | [−0.11, +0.32] | 36 | −0.100 | [−0.39, +0.22] |
| GDP_ADV | 20 | −0.204 | [−0.58, +0.16] | 1 | — | — |
| PCE | 59 | −0.077 | [−0.34, +0.23] | 34 | +0.163 | [−0.21, +0.56] |

Every CI includes zero; every placebo verdict is "selection / chance". The one line worth a
glance is options on CPI days: 19 days, return on premium −13.9% against −0.2% on no-event
days, win rate per day 16%; it is under the 30-session minimum and reported only. All four
verdicts: no finding. The two recovered jobs days (2022-08-05, 10-07) moved NFP from +0.062
to +0.100 σ and did not change its status.

### F5. Setup × category (Q7)

Shares: 66 tested cells, two BH passes, both on Fed days: **PDH_PDL_FailedBreak −0.059 σ
[−0.10, −0.03], −$189 a day**, and **VWAP_Reclaim +0.049 σ [+0.02, +0.08], +$157 a day**,
the latter a low-fidelity setup (79% of its replayed trades have no live twin). Both are
about a fifth of the 0.25 σ bar. Options: 20 cells, no pass. **Shortlist: empty.** By 15.7 a
setup's cell is its net over the whole arm's σ, so one setup out of fifteen cannot reach a
quarter of the arm's daily variation unless it dominates the book; the bar was set before
the data and stands. The tested families are identical to the ones committed before any
P&L was read (set equality, 66 and 20).

### F6. First 60 minutes on release days (Q5 rule-shaped line, descriptive)

Shares: −$319 per session on release days vs −$155 on others, difference −$164
[−$554, +$234]. Options: −$144 vs −$56, difference −$88 [−$276, +$109]. Nothing.

### F7. Options vs shares (Q6)

No category shows divergence (both CIs excluding zero with opposite signs). Fed days are
−0.30 σ on shares and −0.29 σ on options; the largest gaps (financial_stress,
pandemic_health) are the shares arm's 2020 losses with the options arm flat.

## Conclusions by information set (13.8)

**(i) Calendar awareness, known before the open.** Release days as a whole: no difference
from baseline. Fed days: the shares arm loses about $960 more, $870 of it in the morning,
with a CI that excludes zero and a correction it fails. This is the one candidate for a
pre-open forward policy experiment, to be written as its own one-paragraph pre-registration
if the owner wants it, after the holdout and gate 2.

**(ii) Release information, known at the minute.** After-release trades on Fed days lose
less than before-release trades (descriptive). No intraday rule follows from this study.

**(iii) Post-release market reaction.** Never a condition. The twin column says Fed-day
options trades sat on a −7.9 bp average underlying move against +2.1 bp on no-event days: the
arm was wrong-footed on direction *and* paid for convexity. Reported, not used.

## What would make this wrong, and what was checked

- **A session counted twice or a session lost.** `load_replay` raises on a duplicate; 50
  shares chunks (49 plus the fill-in) and 36 options chunks, one hash each
  (`7c6414fc549315f3`, `ccd00ac71f8243e9`, the live hashes). 23 shares sessions WERE lost and
  are now back (top of this file); `analyze.exclusions` compares every open weekday by the
  NYSE calendar with the replayed sessions on every run: 0 lost.
- **$0 option days inflating the effects (14.11).** 270 sessions without a 0DTE are excluded
  before σ, B0 and every table; all are Tuesdays and Thursdays before 2022-11-14; the join
  refuses a chunk without `sessions.json` and the tests fail on the previous code.
- **The families changed after seeing P&L.** They did not: the tested cells in the analysis
  equal the cells committed in `counts_*_discovery.json` before any P&L was read (shares 66,
  options 20; H3 4 and 2), checked by set equality, before and after the fill-in.
- **The placebo as pre-registered favours "effect".** Confirmed at gate 2 and shown above;
  the in-pool sensitivity is beside every pre-registered rank. The Fed verdict survives it;
  the release-days verdict does not.
- **Serial dependence.** Daily z has lag-1 autocorrelation of about −0.03, |z| about +0.31
  (volatility clustering). A 10-session block bootstrap (post hoc) is beside every cell; on
  Fed days it now reads [−0.56, −0.03], and the shock-day CI still excludes zero.
- **A number that moves when nothing moved.** Seeds fixed (20260828); two runs of the
  analysis produce identical JSON apart from the timestamp. The fill-in moved numbers, and
  every move is listed at the top of this file.
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
- **The fill-in itself.** The heartbeat printed the 24 recovered sessions' nets to the
  owner's console, including the one holdout day (2024-10-01); that single number was seen
  by the assistant and is logged in §23, unaggregated.

## Holdout, read once on 2026-11-11 (section 7 confirmatory tests)

Shares 2023-01-03 → 2026-08-27: 908 sessions, 162,374 trades, 38 chunks, 0 lost, 8 early
closes dropped. Options 2023: 248 sessions, 5,993 trades, 0 lost. Both at the live hashes. The
read happened after gate 2 passed on the discovery report (`audit_results.md`); the lock
opened on the auditor's line and nothing read these files before. σ is the discovery σ
(15.5). The holdout's own daily σ, descriptive: shares $2,302, options $862; both arms were
calmer than in discovery and lost less on an average day (shares −$109, options −$61).

**C1 and C2: no difference on calendar-release days, in either arm.**

| test | arm | scheduled sessions | effect σ | 95% CI | $/day | p (boot) | MDE (80%) | pass rule | Holm (family of 2) |
|---|---|---|---|---|---|---|---|---|---|
| C1 | shares | 166 of 908 | **+0.024** | [−0.07, +0.11] | +75 | 0.597 | 0.24 σ | **FAIL** | no |
| C2 | options | 48 of 248 | +0.057 | [−0.17, +0.31] | +59 | 0.654 | 0.45 σ | FAIL (pre-declared underpowered) | no |

In plain words: on the 166 shares sessions with a scheduled release the arm made $75 a day
*more* than its average day, with an interval from −$230 to +$360. The study was powered to
see a 0.24 σ difference on shares, about $770 a day; the observed difference is a tenth of
that. C2 could only have seen 0.45 σ and saw 0.06. Single-category sensitivity: +0.04 σ and
+0.08 σ. By year on shares: 2023 +0.07, 2024 +0.04, 2025 −0.01, 2026 −0.02. Nothing to Holm.

**The Fed-day candidate did not replicate.** On the 29 holdout Fed sessions the shares arm
made −$176 a day against its average, interval [−0.22, +0.10] σ, i.e. −$690 to +$310. The
discovery figure was −$958 a day with an interval that excluded zero. By year: 2023 −0.01,
2024 −0.08, 2025 +0.02, 2026 −0.21 σ. The pre-14:00 excess fell from −$874 to −$150 a
session; win rate per day 38% vs 41% on no-event days (discovery: 28% vs 40%). Options: 8 Fed
sessions, +0.003 σ. The discovery row was exploratory, failed its correction, and was half
2022; the holdout says it was selection. **No forward policy experiment is proposed.** The
"stand aside on Fed mornings" rule would have saved about $250 a session of raw morning
loss on the 29 holdout Fed days, of which about $150 is Fed-specific excess, against $1,145
and $874 in discovery; a $2,300-σ arm does that by chance.

**Everything else, exploratory and consistent with chance:** every Q1 category on both arms
has a CI including zero except financial_stress (shares −0.53 σ on 13 days, options −0.55 σ
on 6 days; never tested by rule, 2023 banking days). Shock days: shares −0.04 σ [−0.28,
+0.31] on 77 sessions, nothing left of the discovery −0.33 σ; the within-regime split is
flat. Q7: 39 tested cells on shares, none pass BH; options 0 tested cells. First-60 line:
nothing. CPI on options, the one discovery line "worth a glance", is −0.28 σ on 12 days with
a CI across zero.

**Information sets (13.8), final.** (i) Calendar awareness: no difference from baseline on
release days (C1 FAIL, C2 FAIL/underpowered); the Fed-day loss seen in discovery did not
hold. (ii) Release information: nothing. (iii) Post-release reaction: never a condition. The
frozen runners' outcomes are not detectably associated with the news calendar.

## Next (section 11, step 6)

1. Gate 2 audit of the holdout section above; defects corrected in place.
2. Results published to the Space as exploratory, with the holdout verdicts.
3. The forward period (live sessions from 2026-11-02) is never read by this study; no
   forward policy experiment is registered.
