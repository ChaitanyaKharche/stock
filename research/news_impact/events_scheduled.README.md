---
task_id: R-2026-10-30-02
trace_id: https://claude.ai/code/session_01TqhCAiwZiZ56SHrYyFXGVT
parent: R-2026-10-30-01
role: calendar-agent
repo_sha: a367245
inputs:
  - https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm
  - https://www.federalreserve.gov/monetarypolicy/fomchistorical2016.htm (and 2017, 2018, 2019, 2020)
  - https://www.bls.gov/bls/news-release/cpi.htm
  - https://www.bls.gov/bls/news-release/empsit.htm
  - https://www.bea.gov/news/archive?field_related_product_target_id=451&created_1=All (GDP, pages 0-5)
  - https://www.bea.gov/news/archive?field_related_product_target_id=476&created_1=All (Personal Income, pages 0-5)
  - https://www.bea.gov/news/2026/... and /news/20YY/... (each BEA release page, 172 pages, used as the per-row source_url)
  - https://www.bea.gov/news/schedule/full and https://www.bea.gov/news/schedule/full-2025 (cross-check only)
decision_rule: "n/a: data collection per preregistration section 3.1"
outputs: [research/news_impact/events_scheduled.csv]
evidence:
  - "FOMC per year 2016-2026: 8,8,8,8,7,8,8,8,8,8,7 = 86 (2020 March 17-18 meeting cancelled; 2026 counted through Oct 28)"
  - "CPI per year: 12 x9, 2025=11 (Oct-2025 CPI never published), 2026=10 through Oct 14 = 129"
  - "NFP per year: 12 x9, 2025=11 (Oct-2025 report never published), 2026=10 through Oct 2 = 129"
  - "GDP_ADV per year: 4 every year = 44 (2019-02-28 and 2025-12-23 are post-shutdown 'Initial Estimates', noted)"
  - "PCE per year: 12, except 2019=11, 2025=10, 2026=11 = 128 (shutdown-delayed releases folded together, noted)"
  - "516 rows, no duplicates on (date,type), no weekend dates, all dates parse, every row has a source_url"
  - "6 rows have session != date (Good Friday releases): 2017-04-14 CPI, 2020-04-10 CPI, 2021-04-02 NFP, 2023-04-07 NFP, 2024-03-29 PCE, 2026-04-03 NFP"
assumptions: []
open_questions:
  - "NFP 2026-02-11 is not the first-Friday pattern; the BLS archive gives the date but not the reason"
  - "BLS dates come from the archive filenames (cpi_MMDDYYYY, empsit_MMDDYYYY) read through WebFetch twice with agreeing output; bls.gov blocks plain HTTP clients so the raw HTML was not inspected"
errors: []
status: done
verified_by: null
budget: {cap: "one pass"}
---
One row per (release date, type) for FOMC statements, CPI, NFP, advance GDP and Personal Income
and Outlays, 2016-01-01 to 2026-10-29. `date` is the actual release date; `session` is the same
day when US equities traded, otherwise the next NYSE session (6 rows differ, all Good Fridays).
BEA archive "Published" dates were checked against each release page's printed date: two
disagreed (2018-10-26 GDP advance, 2019-03-01 PCE) and the page date was used. The Dec 23, 2025
"Data Update, September 2025" is a revision, not a release, and is excluded.
