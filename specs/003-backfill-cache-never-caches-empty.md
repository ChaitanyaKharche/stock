Status:      approved (news_arms pre-registration 14.11, owner 2026-11-01); implemented
Why:         backfill._cached pickles whatever the fetch returns, including [] and 0. A dead
             upstream link (seen 2026-10-26 and 10-30) would cache "no data" for a session and
             every later replay would read it as fact. Same defect class as specs/002.
Requirements:
  R1  When a cached fetch returns an empty or falsy value, _cached shall return it and shall
      NOT write a cache file.
  R2  When a fetch raises, _cached shall propagate the exception and write nothing (unchanged
      behaviour, now covered by a test).
  R3  When a fetch returns non-empty data, _cached shall cache it as before.
Touches the frozen record?  none (.cache/backfill is a research cache)
Acceptance tests: backfill_barrier_test.py::test_cache_never_stores_an_empty_response (FAILS on
             the code before this change), ::test_cache_propagates_errors_and_stores_nothing,
             ::test_cache_stores_real_data.
Rollout:     any time; the news_arms replay uses a fresh cache directory.
Out of scope: expiry; distinguishing 472 from empty text (both are "no data" at the feed).
