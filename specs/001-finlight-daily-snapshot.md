Status:      approved (owner, 2026-10-30: "yes put in lab for testing"); implemented; awaiting verified-live
Why:         The news_impact study (research/news_impact/results.md) could only use scheduled
             releases because the Finlight free plan holds about one month of history. A daily
             snapshot taken by the lab builds the forward archive that a later study can use.
             Collection only; no claim, no trading rule.
Requirements:
  R1  When autostart runs its pre-open idle window, the lab shall run news_snapshot once, with
      a 60-second timeout, and shall continue the session whatever its outcome.
  R2  When news_snapshot runs, it shall write live_lab_data/news/<session>.json with the
      articles Finlight returns for the frozen query over the window [previous session 16:00 ET,
      now], never overwriting an existing file (a second run the same day writes
      <session>_<HHMMSS>.json).
  R3  When Finlight is unreachable, the key is missing, or the client is not installed, the
      lab shall write live_lab_data/news/<session>.error.json with the reason and exit 0.
  R4  The API key shall be read from the FINLIGHT_API_KEY environment variable only; it shall
      never appear in a file, a log line or an argument.
  R5  The query string, page size and window are constants in news_snapshot.py and are not
      changed without a dated note here.
Touches the frozen record?  none (new directory live_lab_data/news/; no config hash; no trading code)
Acceptance tests: trade_analysis/live_lab/news_snapshot_test.py (fake client: writes file; never
             overwrites; error file on failure; key absent -> error file, exit 0) and
             autostart_guard_test.py::test_news_snapshot_hook_never_raises (subprocess error and
             timeout both swallowed and logged).
Rollout:     pushed 2026-10-30; the laptop pulls after 16:05 ET; first run Monday 2026-11-02 at
             ~08:30 ET. Proof: live_lab_data/news/2026-11-02.json exists in that day's commit,
             or a .error.json naming why. Requires once on the laptop, in the pyenv Python:
             `pip install finlight-client` and the env var FINLIGHT_API_KEY.
Out of scope: any reading of the snapshots; sentiment; any join to the trading record.
