---
task_id: R-2026-10-30-03
trace_id: https://claude.ai/code/session_01TqhCAiwZiZ56SHrYyFXGVT
parent: R-2026-10-30-01
role: geopolitical-agent
repo_sha: a367245
inputs: [main sources]
decision_rule: "n/a: data collection per preregistration section 3.2"
outputs: [research/news_impact/events_geopolitical.csv]
evidence: [row counts per category and per year]
assumptions: [where you judged a date shift to next trading day]
open_questions: []
errors: []
status: done
verified_by: null
budget: {cap: "one pass"}
---

# events_geopolitical.csv — how it was built

**One line:** 321 dated, sourced shocks, 2016-01-06 to 2026-10-28, built in two passes
(247 rows in pass one; 74 rows added in a scoped second pass, before any feature file
existed, from the list of events pass one had itemised as left out for lack of a source)
under preregistration §3.2 as amended by §9 A1/A2 (committed ecccb5a). Frozen at first
commit; date errors may be corrected in place with a note, rows never added.

## Columns

`date,break_time_et,category,headline,source_url,surprise`

- `date` — first US regular session on which the market could react. News at or after
  16:00 ET, on a weekend or on a market holiday is assigned to the next session (A2).
- `break_time_et` — HH:MM Eastern when the news broke, or `unknown`. Times are to the
  nearest hour or half-hour from my own knowledge of the event, not read off the source;
  63 rows are `unknown`, nearly all weekend/holiday events where the hour does not change
  the assigned session. A time earlier than 16:00 on a date *before* `date` means the news
  broke after the close on the previous day or on a closed day (e.g. `20:00` on a row dated
  2016-01-06 = 20:00 ET on 2016-01-05).
- `category` — one of the eight fixed values in §3.2, nothing else.
- `headline` — ≤12 words, factual; a parenthetical such as "(Sunday)" marks a date shift.
- `source_url` — one URL per row, each either returned by a web search I ran or fetched
  directly and read in this session. No URL was typed from memory without a successful
  fetch confirming it. 265 Bloomberg, 56 primary sources (White House incl. Biden archive,
  Federal Reserve, Treasury, FDIC, USTR, DOJ, DoD, House Clerk, CISA, National Archives,
  Court of International Trade, Bank of England, SNB, Bank of Japan, gov.uk, Moody's).
- `surprise` — `yes` if unscheduled, `no` for a scheduled vote, deadline, ruling or
  pre-announced action.

## Counts

| year | rows | | category | rows |
|---|---|---|---|---|
| 2016 | 19 | | trade_tariff | 70 |
| 2017 | 26 | | political_shock | 62 |
| 2018 | 34 | | war_military | 56 |
| 2019 | 37 | | financial_stress | 36 |
| 2020 | 43 | | energy_shock | 35 |
| 2021 | 24 | | us_fiscal | 35 |
| 2022 | 33 | | pandemic_health | 17 |
| 2023 | 23 | | cyber_infrastructure | 10 |
| 2024 | 28 | | | |
| 2025 | 36 | | surprise=yes | 262 |
| 2026 | 18 | | surprise=no | 59 |

Every year has at least 8 rows. After the second pass the decade is covered at 23–43
rows/year except 2016 (19), 2021 (24) and 2026 (18); the 2026 thinness is the year being
ten months long and sourced from primary pages only (see caveat below).

## Method

1. Candidate list drafted year by year from memory against the §3.2 bar (led at least
   two of Reuters/AP/WSJ/FT/Bloomberg; plausible channel to US equities, rates or dollar),
   then each candidate confirmed by a web search restricted to `bloomberg.com` (Reuters,
   AP, WSJ and FT block the search tool's crawler, so Bloomberg is the only one of the five
   outlets that returns URLs) or to a primary-source domain. Query form:
   `"<event> <Month D YYYY>"`, e.g. "Trump withdraws from Iran nuclear deal May 8 2018
   sanctions oil", "Fed emergency rate cut March 3 2020 statement", "Abqaiq drone attack
   Saudi Aramco September 14 2019". Snippets were used to confirm the date; Bloomberg URLs
   carry the publication date in the path and were checked against the assigned session.
2. The web-search budget (200 calls per turn, shared) ran out at the 2022-03-08 query.
   Everything from mid-2022 onward was then sourced by fetching primary-source pages
   directly: Federal Reserve and Treasury press-release indexes, FDIC releases, the
   whitehouse.gov `presidential-actions` index (pages 1–62 sampled), the Biden White House
   archive, gov.uk, BoE, SNB, BoJ, CISA, National Archives, the CIT slip opinion, Moody's.
   Bloomberg pages cannot be fetched (403), so no new Bloomberg rows could be added after
   the budget ran out; a handful of 2024–2026 Bloomberg URLs that had appeared incidentally
   in earlier search results were used where they document the event.
   **Second pass (same day, before any feature file existed):** the coordinator reset the
   search budget and scoped the pass to the itemised left-out list only. Each of those
   events was searched once on `bloomberg.com` with the same query form, and all 74 rows
   (the list had ~70 entries; US strikes on Iran / Al Udeid / ceasefire became three rows,
   the July 2025 tariff letters two, the Trump–Powell episode two) were found and added.
   No event outside that list was added and no existing row was changed or dropped.
3. Tightened front-page rule (A1): a row was kept only if the day's lead headline was
   about the event itself. **19 candidate rows were dropped** because their lead headline
   was the market's reaction, not an event: 2016-01-04 China circuit-breaker halt,
   2016-01-20 oil below $27, 2016-02-11 global bank rout, 2016-10-07 sterling flash crash,
   2018-02-05 "Volmageddon", 2019-08-14 2s10s inversion, 2020-04-20 WTI settles negative,
   2020-06-11 virus-fear selloff, 2021-01-27 GameStop squeeze day, 2021-02-25 Treasury
   selloff, 2021-05-19 crypto crash, 2021-07-19 Delta-variant selloff, 2021-09-20 Evergrande
   contagion day, 2021-10-06 European gas-price spike, 2022-01-24 Nasdaq intraday reversal,
   2023-10-23 10-year at 5%, 2024-08-05 Japan carry-trade unwind, 2025-03-10
   recession-comment selloff, 2025-10-16 regional-bank credit scare. The `financial_stress`
   rows that remain are discrete actions or failures (a bank closed, a central-bank
   facility launched, a fund's banks warning of losses) and are reported, never tested.
4. Excluded by design: scheduled macro releases (another file), company earnings, Fed
   speeches and scheduled FOMC meetings, scheduled foreign central-bank meetings except
   where the decision itself was the shock (BoJ 2016-01-29, 2022-12-20; the BoJ
   2024-03-19 and 2024-07-31 rows are `surprise=no`).

## Left out for lack of a fetchable source

After the second pass: **none of the itemised events remain missing.** Every event listed
here at the end of pass one now has a row with a Bloomberg URL returned by search. For
the record, the pass-one list was: 2016 Comey letter; 2022 Shanghai lockdown, Gazprom
Poland/Bulgaria, TerraUSD, Freeport LNG, Celsius, Abe, Nord Stream 20%, IRA deal, Nord
Stream halt, mobilization, Nord Stream sabotage, OPEC+ cut, chip controls, Truss, Xi third
term, midterms, FTX, Poland missile, China protests, zero-Covid exit; 2023 balloon, OPEC+
surprise cut, PacWest, debt deal and passage, Wagner, Moody's banks, Country Garden, iPhone
ban, shutdown averted, chip rules, Gaza ground invasion, Moody's outlook, Red Sea; 2024
Taiwan election, Key Bridge, Israel–Iran (04-19, 10-28), Raisi, Trump conviction, Biden
withdrawal, Haniyeh, second attempt, China stimulus, Iran missiles (10-01), port strike,
ATACMS, tariff threat, Korea martial law, French government, December shutdown threat;
2025 inauguration, Zelenskiy clash, tariffs effective, USMCA carve-out, Powell episode,
India–Pakistan, EU 50%, Israel strikes Iran, US strikes Iran and ceasefire, OBBBA, tariff
letters, US–EU deal, BLS firing, Cook, appeals court, Kirk, shutdown, 100% China threat.

Primary-source URLs for these were tried first where a page was guessable (Kremlin 503,
opec.org 402, bis.gov 403, federalregister.gov blocked, Biden-archive slugs not found,
gov.uk Truss page not found); where none fetched, the Bloomberg URL from search was used.
Reuters/AP/WSJ/FT still block both search and fetch, so the "two outlets" test remains
asserted from the nature of the event for every row, not verified against two front pages.

Second-pass date judgements worth flagging: 2022-09-02 Gazprom halt (announced ~13:00 ET
Friday, dated Friday not Tuesday); 2024-12-03 Korea martial law (declared 08:30 ET, lifted
overnight, dated 12-03); 2025-05-06 India strikes (01:05 IST 05-07 = 15:35 ET 05-06, so
dated 05-06 by the 16:00 rule); 2025-06-23 Al Udeid (11:30 ET) and 2025-06-24 ceasefire
(18:00 ET 06-23) are separate rows; 2025-07-07 OBBBA (House vote 14:31 ET on the 3 July
half-day, after the 13:00 close → 07-07); 2022-10-24 Xi reshuffle is kept with an
event headline although Monday's lead headlines were as much about the market rout.

## 2026 rows — caveat

The 18 rows for 2026 are built from (a) whitehouse.gov presidential actions and Federal
Reserve releases, which document the action and its signing date but not the hour, and
(b) Bloomberg headlines that surfaced in earlier searches. For these rows `break_time_et`
is `unknown`, the date is the publication/signing date (an action signed after 16:00 ET
would belong to the next session; I could not check), and the two-outlet front-page test is
asserted from the nature of the event, not verified. Gate 1 should weigh them accordingly.

## Assumptions on date shifts (judgement calls)

- 2016-07-18 Turkey coup: first wires ~15:30–16:00 ET Friday 07-15, PM's confirmation
  ~16:00; assigned to Monday 07-18 because the market could not meaningfully react Friday.
- 2016-12-19 Ankara/Berlin: both attacks broke during US hours (11:00 and 14:00 ET), so
  dated 12-19, not 12-20.
- 2017-07-28 and 2017-11-28 North Korea ICBMs: launches at 10:45 and 13:17 ET, so dated
  the same US day (not the Korean calendar day).
- 2017-05-18 Mueller appointed ~18:00 ET 05-17 → 05-18. The 05-17 selloff followed the
  05-16 evening Comey-memo report, which has no fetched source and is not a row.
- 2018-05-29 Italy: Mattarella's veto broke Sunday 05-27, 05-28 was Memorial Day → 05-29.
- 2018-12-06 Huawei CFO: broke ~21:00 ET 12-05, a day US markets were closed (Bush
  funeral) → 12-06.
- 2019-08-05/06 yuan breaks 7 (21:30 ET 08-04 → 08-05) and the manipulator designation
  (~18:00 ET 08-05 → 08-06) are two rows.
- 2019-09-24 impeachment inquiry: afternoon reports (~14:00 ET) that Pelosi would back
  it moved markets before her 17:00 announcement; dated 09-24.
- 2020-01-22 Wuhan lockdown: announced 02:00 CST 01-23 = 13:00 ET 01-22 → 01-22.
- 2020-12-24 Brexit deal: broke ~10:00 ET on a half-day session; dated 12-24 (the
  feature script's `thin` flag will exclude it).
- 2021-05-10 Gaza: rockets on Jerusalem ~11:00 ET 05-10 → 05-10, not 05-11.
- 2022-08-02 Pelosi Taiwan: landing ~10:45 ET 08-02; the cited statement is dated 08-03
  Taipei time.
- 2023-05-02 Yellen June-1 X-date letter released ~16:00 ET 05-01 → 05-02.
- 2023-10-04 McCarthy ousted: vote at 16:42 ET 10-03 → 10-04.
- 2025-02-10 steel/aluminum: announced Sunday 02-09, proclamation signed Monday evening;
  dated 02-10 on the Sunday announcement.
- 2025-03-27 auto tariffs: signed ~16:00 ET 03-26 → 03-27. 2025-04-03 reciprocal
  tariffs: 16:00 ET 04-02 → 04-03.
- 2026-04-06 DHS shutdown order: signed Good Friday 04-03 → 04-06.
- Category judgements: terror attacks and strikes on infrastructure by state actors are
  `war_military`; Suez blockage, Colonial Pipeline, Facebook outage, CrowdStrike, Equifax,
  SolarWinds, bulk-power emergency are `cyber_infrastructure`; China regulatory
  crackdowns (Hong Kong law, tutoring ban) are `political_shock`; the UK mini-budget is
  `financial_stress`; hurricanes that shut Gulf refining are `energy_shock`.
