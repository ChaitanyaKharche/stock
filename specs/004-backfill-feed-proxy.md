Status:      approved (news_arms pre-registration 15.1, owner 2026-11-01); implemented
Why:         During a replay the runner held a HistoryFeed that exposes the real feed's methods
             by delegation of whatever the runner asks for. Nothing stopped a runner path from
             calling an unguarded method and reading data past the simulated clock. The
             pre-registration's test (i) requires that this be impossible, not just unobserved.
Requirements:
  R1  When backfill.run attaches a feed to the runner, it shall attach a proxy exposing ONLY:
      minute_bars, extended_bars, stock_quote, chain_quotes, zero_dte, open_interest,
      minute_bars_many, stock_quote_many, calls, stats, wait_for_upstream, close.
  R2  When any other attribute is accessed on the proxy, it shall raise AttributeError naming
      the attribute and the spec.
  R3  The runner shall complete a replayed day through the proxy (the live-period acceptance
      run is the proof; backfill_barrier_test covers R1 and R2 offline).
Touches the frozen record?  none (replay code path only; live runners never see the proxy)
Acceptance tests: backfill_barrier_test.py::test_proxy_allows_only_the_guarded_surface,
             ::test_run_attaches_the_proxy (FAILS before this change: _attach_feed did not exist).
Rollout:     any time; no session impact.
Out of scope: changing HistoryFeed's own guards (already tested as (a)-(g)).
