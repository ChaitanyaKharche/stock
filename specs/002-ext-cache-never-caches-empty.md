Status:      approved (orchestrator, defect found at gate 2 of news_impact 2026-10-30); implemented
Why:         levels_test.get_ext pickles whatever the feed returns. On 2026-10-30 the Theta
             Terminal upstream link was dead; 25 QQQ sessions (2026-09-25 -> 10-29) were cached as
             empty and read as "no bars" on every re-run (research/news_impact/results.md section 9,
             audit_results.md item 11). A proven helper was caching an outage as a fact.
Requirements:
  R1  When the feed returns no RTH bars for a session, get_ext shall return what it got and
      shall NOT write a cache file for that session.
  R2  When the feed returns RTH bars (a half day included), get_ext shall cache as before.
Touches the frozen record?  none (ext_cache is a research cache; the lab's runners do not read it)
Acceptance tests: levels_test.py::test_get_ext_does_not_cache_an_empty_response and
             ::test_get_ext_caches_a_real_session; the first shown to FAIL on the code before
             this change (run 2026-10-30, see commit message).
Rollout:     any time; no session impact. Owner deletes the 25 stale QQQ cache files once
             (Remove-Item live_lab_data\ext_cache\QQQ_2026-09-2*.pkl, QQQ_2026-10-*.pkl) and
             re-runs day_features for QQQ.
Out of scope: cache expiry, cache format.
