"""Unattended daily launcher for the live lab.

Designed to be fired by Windows Task Scheduler well before the open and left alone.

What it handles that the bare runner does not:

  * **Timezone drift.** It sleeps until 09:20 *exchange* time, computed live. The machine
    is on MST (currently -3h from ET) and Arizona does not observe DST, so the local-time
    offset changes twice a year. Scheduling a fixed local time would silently drift; this
    does not.
  * **Weekends and market holidays.** Exits immediately rather than idling all day. A
    holiday is detected by the absence of bars shortly after the open, not by a hardcoded
    calendar that would go stale.
  * **Terminal not running.** Detects it, and with --start-terminal will launch it and
    wait for it to answer.
  * **Preflight gating.** Aborts on data-integrity failures (delayed quotes, missing
    warmup bars) because those silently corrupt the record. Does NOT abort on the LAN
    exposure finding -- that is a real security issue but it does not corrupt data, and
    blocking collection on it would just lose sessions.
  * **Supervision.** The runner exiting is not the same as the session being over. If it
    dies before 15:55 ET -- unhandled error, terminal restart, network reset, wake from
    sleep -- this restarts it with backoff, up to MAX_RESTARTS. Restarting is lossless:
    the runner checkpoints open positions after every tick and reloads them on start, so
    it resumes holding exactly what it was holding.
  * **Single instance.** Both the supervisor and the runner take an OS-level lock, so the
    scheduled task can repeat through the day (covering a late boot or a whole dead
    process tree) without ever producing two runners appending to the same files.
  * **Stale recovery refused.** positions_open.json is stamped with its session date. A
    process killed with positions open leaves that file behind; recovery on a LATER day
    archives it and starts flat instead of marking expired contracts against a new day.
  * **Logging.** Everything tee'd to live_lab_data/logs/<date>.log.

    python -m trade_analysis.live_lab.autostart
    python -m trade_analysis.live_lab.autostart --start-terminal --symbols QQQ SPY
    python -m trade_analysis.live_lab.autostart --now        (skip the wait; run at once)
"""
from __future__ import annotations

import argparse
import datetime as dt
import subprocess
import sys
import threading
import time
from pathlib import Path

from .clock import now_et
from .feed import FeedOutage, ThetaLiveFeed
from . import ledger
from .lock import SingleInstance

try:
    from ..bulk_download.trading_days import is_trading_day as _is_session
except Exception:                                        # pragma: no cover
    def _is_session(d):                                  # noqa: D103
        return d.weekday() < 5
from .preflight import EXPOSURE_TAG, OPTIONS_TAG

FAIL_MARK = "[FAIL]"
from .runner import EXIT_ALREADY_RUNNING
from .shares_runner import BROAD_UNIVERSE
from .store import DEFAULT_LAB_DIR

START_AT = dt.time(9, 20)          # exchange time; structural preflight runs here
FRESHNESS_AT = dt.time(9, 33)      # RTH freshness re-check -- MUST be after the open,
                                   # otherwise the delayed-feed gate can never fire.
                                   # Costs nothing: no setup can signal before 09:36
                                   # because the 09:30-09:35 opening range is not formed.
GIVE_UP_AFTER = dt.time(15, 30)    # too late in the session to bother starting
HOLIDAY_CHECK_AT = dt.time(9, 45)
RUNNER_DONE_AFTER = dt.time(15, 55)  # an exit at/after this is a normal end of session,
                                     # not a crash to restart. NOT a kill time -- the
                                     # runners flatten at 15:55 and exit on their own at
                                     # 16:00, and terminating them at 15:55 destroyed the
                                     # EOD flatten and the daily summary for three straight
                                     # sessions (2026-09-01..03).
HARD_STOP = dt.time(16, 10)          # backstop only: by now a healthy runner has long since
                                     # exited, so anything still alive is wedged
MAX_RESTARTS = 20
RESTART_BACKOFF = [5, 15, 30, 60, 120]   # seconds; holds at the last value
TERMINAL_PORT = 25503
TERMINAL_DIR = Path(r"C:\Users\chaitanyakharche\Documents\research_data\thetaterminal")
TERMINAL_CMD = [str(TERMINAL_DIR / "jdk21" / "jdk-21.0.12+8" / "bin" / "java.exe"),
                "-jar", str(TERMINAL_DIR / "ThetaTerminalv3.jar")]


class Tee:
    """stdout -> console AND the day's log file."""

    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.fh = open(path, "a", encoding="utf-8", buffering=1)
        self.stdout = sys.stdout
        self._lock = threading.Lock()      # two relay threads write here concurrently

    def write(self, s):
        with self._lock:
            self.stdout.write(s)
            self.fh.write(s)

    def flush(self):
        with self._lock:
            self.stdout.flush()
            self.fh.flush()


def log(msg: str) -> None:
    print(f"[autostart {now_et():%Y-%m-%d %H:%M:%S ET}] {msg}", flush=True)


def terminal_up(timeout=4.0) -> bool:
    try:
        import httpx
        r = httpx.get("http://127.0.0.1:25503/v3/stock/snapshot/quote",
                      params={"symbol": "QQQ"}, timeout=timeout)
        return r.status_code == 200
    except Exception:                                        # noqa: BLE001
        return False


def _evict_stale_terminal() -> bool:
    """Kill whatever holds 25503 when it is NOT serving. Returns True if it killed one.

    start_terminal() is only ever called after terminal_up() said no. If the port is
    nevertheless held, the holder is a terminal that is listening but not answering --
    and launching a second one on top of it cannot work: the new instance logs
    "Address 127.0.0.1:25503 already in use ... Shutting down terminal" and exits, so
    the 90s wait always times out and autostart aborts with 'no feed'.

    That is exactly what happened on 2026-09-08. A terminal started 2026-09-03 survived
    the weekend, lost its upstream, kept the socket, and every scheduled retry launched
    a doomed second copy. The session was lost from 09:20 until it was killed by hand.

    An open socket is not a working service. The repo already learned this once, in the
    LAN-exposure check, which was rewritten to ask Windows rather than self-connect.
    """
    try:
        import psutil
    except ImportError:
        log("cannot check for a stale terminal: psutil not installed")
        return False
    victims = set()
    try:
        for c in psutil.net_connections(kind='tcp'):
            if c.laddr and c.laddr.port == TERMINAL_PORT and c.status == 'LISTEN' and c.pid:
                victims.add(c.pid)
    except (psutil.AccessDenied, OSError) as exc:
        log(f'cannot enumerate listeners on {TERMINAL_PORT}: {exc!r}')
        return False
    if not victims:
        return False
    killed = False
    for pid in victims:
        try:
            p = psutil.Process(pid)
            # Only ever kill a terminal. Never something else that happens to hold it.
            if 'java' not in p.name().lower():
                log(f'{TERMINAL_PORT} held by {p.name()} (pid {pid}); refusing to kill it')
                continue
            age = time.time() - p.create_time()
            log(f'evicting unresponsive terminal pid {pid} (up {age/3600:.1f}h)')
            for child in p.children(recursive=True):
                child.kill()
            p.kill()
            p.wait(timeout=10)
            killed = True
        except psutil.NoSuchProcess:
            killed = True
        except (psutil.AccessDenied, psutil.TimeoutExpired) as exc:
            log(f'could not kill pid {pid}: {exc!r}')
    if killed:
        time.sleep(3)          # let the socket clear before the new one binds
    return killed


def _wait_for_history(wait_sec=45) -> bool:
    """The quote endpoint answers BEFORE the historical upstream is connected.

    Theta Terminal binds its HTTP server and serves live snapshot quotes as soon as it
    starts, but /stock/history/ohlc goes through MDDS, which connects a few seconds
    later. On 2026-09-08 the terminal answered at 10:30:29 and preflight ran at
    10:30:30 -- one second on -- so the first two symbols it asked for warmup bars,
    QQQ and SPY, both came back

        HTTP 503: Unable to resolve host mdds-01.thetadata.us

    while the other thirteen, queried seconds later, returned a full [390, 390, 390].
    Two FAILs, whole session aborted, purely because the readiness check asked the one
    endpoint that is ready first. It looked like a DNS fault and was a race.

    "The socket is open" was never the question. Neither is "a quote came back". The
    question is whether the data the session actually needs can be served yet.
    """
    day = now_et().date() - dt.timedelta(days=1)
    for _ in range(30):
        if _is_session(day):
            break
        day -= dt.timedelta(days=1)
    feed = ThetaLiveFeed()
    try:
        deadline = time.time() + wait_sec
        attempt = 0
        while time.time() < deadline:
            attempt += 1
            try:
                if feed.minute_bars('QQQ', day):
                    log(f'historical upstream ready (after {attempt} probe(s))')
                    return True
            except FeedOutage as exc:
                if attempt == 1:
                    log(f'waiting for the historical upstream: {str(exc)[:90]}')
            time.sleep(3)
    finally:
        feed.close()
    log(f'historical upstream still not serving after {wait_sec}s; preflight would fail on warmup')
    return False


def start_terminal(wait_sec=90) -> bool:
    if not Path(TERMINAL_CMD[0]).exists():
        log(f"cannot start terminal: {TERMINAL_CMD[0]} not found")
        return False
    # The port must be free before launching, or the new instance shuts itself down.
    _evict_stale_terminal()
    log("starting Theta Terminal ...")
    try:
        subprocess.Popen(TERMINAL_CMD, cwd=str(TERMINAL_DIR),
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                         creationflags=getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0))
    except OSError as exc:
        log(f"failed to launch terminal: {exc!r}")
        return False
    for _ in range(wait_sec // 3):
        time.sleep(3)
        if terminal_up():
            log("terminal is answering")
            return _wait_for_history(wait_sec=45)
    log(f"terminal did not answer within {wait_sec}s")
    return False


def wait_for_open(target: dt.time) -> bool:
    """Sleep until `target` exchange time today. False if that moment has passed."""
    while True:
        now = now_et()
        if now.time() >= GIVE_UP_AFTER:
            log(f"it is already {now:%H:%M} ET, past the {GIVE_UP_AFTER:%H:%M} cutoff")
            return False
        if now.time() >= target:
            return True
        remaining = (dt.datetime.combine(now.date(), target) - now).total_seconds()
        # log sparsely: hourly when far out, every 5 min in the last quarter hour
        if remaining > 900:
            if int(remaining // 60) % 60 < 6:
                log(f"waiting {remaining/60:.0f} min until {target:%H:%M} ET")
            time.sleep(min(remaining - 900, 1800))
        else:
            log(f"waiting {remaining/60:.0f} min until {target:%H:%M} ET")
            time.sleep(min(remaining, 300))


def is_holiday(feed, symbols, day) -> bool:
    """Detected from data, not a hardcoded calendar that would go stale."""
    if now_et().time() < HOLIDAY_CHECK_AT:
        return False
    for sym in symbols:
        try:
            if len(feed.minute_bars(sym, day)) >= 5:
                return False
        except FeedOutage:
            return False
    return True


def _preflight(symbols) -> str:
    pf = subprocess.run(
        [sys.executable, "-m", "trade_analysis.live_lab.preflight",
         "--ignore-exposure", "--symbols", *symbols],
        capture_output=True, text=True,
        cwd=str(Path(__file__).resolve().parents[2]))
    return pf.stdout or ""


def _all_symbols(args) -> list[str]:
    """Every symbol either arm will touch, in a stable order.

    Preflight checks warmup bars and quote freshness per symbol, so it has to see the
    UNION. Checking only the options universe would let the shares arm start on 13 names
    nothing had verified -- and a missing warmup session there does not fail loudly, it
    quietly produces a session whose prior-day levels are wrong.
    """
    seen, out = set(), []
    for s in list(args.symbols) + list(getattr(args, "share_symbols", []) or []):
        if s not in seen:
            seen.add(s)
            out.append(s)
    return out


def _fatal_lines(out: str) -> list[str]:
    """Preflight FAILs that must abort EVERY arm.

    LAN exposure is deliberately not one: it is a real security issue but it does not
    corrupt data, and blocking collection on it just loses sessions.

    Neither are OPTIONS_TAG failures. The shares arm never calls an option endpoint, so
    an options entitlement lapse says nothing about whether its data is sound. On
    2026-09-08 the ThetaData subscription dropped to Stock: STANDARD / Options: FREE;
    four 403s from the options chain aborted BOTH arms, even though all fifteen
    underlyings were REAL-TIME and thirteen had full warmup. A frozen forward test lost
    a whole session of shares data to a failure in a different arm.

    Keyed on tokens, never on prose -- the exposure guard was once written as a substring
    match against the check's wording and silently stopped matching when that was
    reworded, turning a warning into an abort.
    """
    return [l for l in out.splitlines()
            if FAIL_MARK in l and EXPOSURE_TAG not in l and OPTIONS_TAG not in l]


def _options_blocked(out: str) -> list[str]:
    """Failures that disable the OPTIONS arm but leave the shares arm sound."""
    return [l for l in out.splitlines() if FAIL_MARK in l and OPTIONS_TAG in l]


class _Child:
    """One supervised runner: its command, its process, its restart budget."""

    def __init__(self, name, cmd, root):
        self.name, self.cmd, self.root = name, cmd, root
        self.proc = None
        self.restarts = 0
        self.rc = 0
        self.done = False           # finished normally; do not restart
        self._thread = None

    def start(self):
        self.proc = subprocess.Popen(
            self.cmd, cwd=self.root, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1, encoding="utf-8", errors="replace")
        # Relay the child's output through THIS process's stdout, which is the Tee. Without
        # this the child inherits the console handle directly and every line exists only in
        # the terminal window -- lost the moment it is closed.
        self._thread = threading.Thread(target=self._relay, daemon=True)
        self._thread.start()

    def _relay(self):
        try:
            for line in self.proc.stdout:
                print(line.rstrip(), flush=True)
        except (ValueError, OSError):
            pass

    def poll(self):
        return None if self.proc is None else self.proc.poll()

    def terminate(self):
        if self.proc and self.proc.poll() is None:
            try:
                self.proc.terminate()
            except OSError:
                pass


def build_specs(args) -> list[tuple[str, list[str]]]:
    """The arms to supervise. Separated out so tests can inject fakes.

    The two arms take DIFFERENT universes, and must. `--symbols` used to be forwarded to
    both, which would now be actively wrong:

      * the OPTIONS arm cannot broaden at all. There is no single-name option entitlement
        on this subscription tier and no 0DTE listed on these names anyway, and `symbols`
        is inside its config hash -- widening it would fork 1f7247d7839d9950 and throw
        away the trades collected since 2026-08-28 in exchange for nothing.
      * the SHARES arm broadens to the 15 names the spread survey cleared, because its
        edge is 3.34 bp and spread comes straight off that.
    """
    specs = []
    if not getattr(args, "no_options", False):
        specs.append(("options", [sys.executable, "-m",
                                  "trade_analysis.live_lab.runner",
                                  "--symbols", *args.symbols,
                                  "--lab-dir", args.lab_dir,
                                  "--contracts", str(args.contracts)]))
    if not getattr(args, "no_shares", False):
        specs.append(("shares", [sys.executable, "-m",
                                 "trade_analysis.live_lab.shares_runner",
                                 "--symbols", *args.share_symbols,
                                 "--lab-dir",
                                 str(Path(args.lab_dir) / "shares")]))
    return specs


def supervise(args, day: dt.date, specs=None, child_cls=None) -> int:
    """Keep every arm alive until the close, restarting any that dies early.

    A runner exiting is NOT the same as the session being over. It can die on an unhandled
    error, a Theta Terminal restart, a network stack reset, or the machine waking from
    sleep -- and before this loop existed, any of those silently ended collection for the
    day while the log still said "runner exited rc=0".

    Restarting is lossless: each arm checkpoints its open positions after every tick and
    reloads them on start, so it resumes holding exactly what it was holding. The recovery
    file is stamped with its session date, so a restart on a LATER day refuses stale
    positions rather than marking expired contracts against a new day.

    Two arms run side by side and are supervised independently:
      options  the frozen 0DTE test          -> live_lab_data/
      shares   IntradayMomentumBoundary etc. -> live_lab_data/shares/
    They share nothing but the feed; neither can affect the other's counts.
    """
    root = str(Path(__file__).resolve().parents[2])
    specs = build_specs(args) if specs is None else specs
    cls = child_cls or _Child
    children = [cls(n, c, root) for n, c in specs]
    for ch in children:
        log(f"starting {ch.name} arm")
        ch.start()

    pending = {}                    # name -> monotonic time at which to restart
    while True:
        now = now_et()
        alive = [ch for ch in children if not ch.done and ch.poll() is None]
        if all(ch.done for ch in children):
            log("every arm has finished; nothing left to supervise")
            break
        # Past RUNNER_DONE_AFTER we stop RESTARTING, but we do NOT kill: each runner still
        # has to flatten at 15:55, write its daily summary and exit at 16:00. Killing it
        # here loses every position that was open at the close, and the summary with it.
        if now.date() != day or now.time() >= HARD_STOP:
            if alive:
                log(f"{now:%H:%M:%S} ET -- past the {HARD_STOP:%H:%M} backstop and "
                    f"{len(alive)} arm(s) are still alive; terminating")
            for ch in children:
                ch.terminate()
            break
        if now.time() >= RUNNER_DONE_AFTER:
            if alive:
                time.sleep(5)          # let them flatten and write their summary
                continue
            break

        for ch in children:
            if ch.done:
                continue
            rc = ch.poll()
            if rc is None:
                continue
            ch.rc = rc
            if rc == EXIT_ALREADY_RUNNING:
                log(f"{ch.name}: another instance holds the lock -- not restarting")
                ch.done = True
                continue
            if ch.name not in pending:
                ch.restarts += 1
                if ch.restarts > MAX_RESTARTS:
                    log(f"{ch.name}: exited rc={rc} but has already restarted "
                        f"{MAX_RESTARTS} times; giving up rather than thrashing")
                    ch.done = True
                    continue
                wait = RESTART_BACKOFF[min(ch.restarts - 1, len(RESTART_BACKOFF) - 1)]
                log(f"{ch.name}: exited rc={rc} at {now:%H:%M:%S} ET, before the close -- "
                    f"restart {ch.restarts}/{MAX_RESTARTS} in {wait}s")
                pending[ch.name] = time.monotonic() + wait
            elif time.monotonic() >= pending[ch.name]:
                del pending[ch.name]
                if not terminal_up():
                    log("Theta Terminal is down as well")
                    if args.start_terminal:
                        start_terminal()
                ch.start()
        time.sleep(5)

    for ch in children:
        log(f"{ch.name} arm finished rc={ch.rc} after {ch.restarts} restart(s)")
    return max((ch.rc for ch in children), default=0)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Unattended launcher for the live lab.")
    ap.add_argument("--symbols", nargs="+", default=["QQQ", "SPY"],
                    help="OPTIONS arm universe. Frozen: widening it forks the "
                         "config hash and discards the trades collected so far.")
    ap.add_argument("--share-symbols", nargs="+", default=list(BROAD_UNIVERSE),
                    help="SHARES arm universe; defaults to the 15 names the "
                         "spread survey cleared.")
    ap.add_argument("--lab-dir", default=str(DEFAULT_LAB_DIR))
    ap.add_argument("--start-terminal", action="store_true")
    ap.add_argument("--now", action="store_true", help="skip the wait and start immediately")
    ap.add_argument("--contracts", type=float, default=1.0)
    ap.add_argument("--no-options", action="store_true",
                    help="run only the shares arm (options entitlement lapsed, etc.)")
    ap.add_argument("--no-shares", action="store_true",
                    help="run only the frozen options arm")
    args = ap.parse_args(argv)

    day = now_et().date()
    sys.stdout = Tee(Path(args.lab_dir) / "logs" / f"{day.isoformat()}.log")

    log("=" * 66)
    log(f"autostart invoked | local {dt.datetime.now():%H:%M:%S} | symbols {args.symbols}")

    if day.weekday() >= 5:
        log(f"{day} is a weekend; nothing to do")
        ledger.record(day, "NOT_A_SESSION", "weekend", args.lab_dir)
        return 0

    # The scheduled task repeats through the day so a machine that boots late, or a
    # supervisor that dies with its whole process tree, still gets the lab running. That
    # only works if a second invocation is a no-op while one is already alive.
    # Fill in the records the lab could not write for itself -- days the machine was
    # off, or a session that started and never recorded an end.
    for _w in ledger.reconcile(args.lab_dir):
        log("ledger backfill: " + _w["date"] + " -> " + _w["outcome"])

    guard = SingleInstance("autostart", args.lab_dir)
    if not guard.acquire():
        log(f"another autostart is already supervising ({guard.holder()}); exiting")
        return 0

    if not args.now and not wait_for_open(START_AT):
        return 0

    if not terminal_up():
        log("Theta Terminal is not answering on 127.0.0.1:25503")
        if not args.start_terminal or not start_terminal():
            log("ABORT: no feed")
            ledger.record(day, "ABORTED", "no feed: Theta Terminal would not start", args.lab_dir)
            return 2

    feed = ThetaLiveFeed()
    if is_holiday(feed, args.symbols, day):
        log(f"no bars for any symbol well after the open -- {day} looks like a market "
            f"holiday; exiting rather than idling")
        ledger.record(day, "NOT_A_SESSION", "no bars well after the open; market holiday", args.lab_dir)
        feed.close()
        return 0
    feed.close()

    # ---- structural preflight, pre-open ------------------------------------
    log("running preflight (structural, pre-open) ...")
    out = _preflight(_all_symbols(args))
    for line in out.splitlines():
        log("  " + line)
    if _fatal_lines(out):
        log("ABORT: structural preflight failed")
        ledger.record(day, "ABORTED", "structural preflight failed", args.lab_dir)
        return 3

    # ---- freshness re-check, AFTER the open --------------------------------
    # This is the gate that matters. At 09:20 the market is shut, so freshness is
    # always "UNVERIFIABLE" and a delayed feed would sail straight through. Day one
    # ran with the check effectively disabled; this is the fix.
    if not args.now:
        if not wait_for_open(FRESHNESS_AT):
            return 0
        log("re-running preflight for FEED FRESHNESS (market now open) ...")
        out = _preflight(_all_symbols(args))
    for line in out.splitlines():
        log("  " + line)

    fatal = _fatal_lines(out)
    blocked = _options_blocked(out)
    if blocked and not getattr(args, "no_options", False):
        log(f"{len(blocked)} OPTIONS-only failure(s); the options arm cannot run:")
        for l in blocked:
            log("   " + l.strip())
        if getattr(args, "no_shares", False):
            log("ABORT: options is the only requested arm and it is blocked")
            ledger.record(day, "ABORTED", "options is the only requested arm and it is blocked", args.lab_dir)
            return 3
        log("continuing with the SHARES arm alone -- it touches no option endpoint, and")
        log("losing its session to a failure in the other arm is a worse outcome.")
        args.no_options = True

    if any("DELAYED" in l for l in out.splitlines()):
        log("ABORT: the feed is DELAYED. Entries and exits would be priced off stale")
        ledger.record(day, "ABORTED", "feed is DELAYED; prices would be stale", args.lab_dir)
        log("       quotes, which corrupts the record rather than merely degrading it.")
        return 3
    if fatal:
        log(f"ABORT: {len(fatal)} preflight data failure(s)")
        ledger.record(day, "ABORTED", "preflight data failure(s)", args.lab_dir)
        return 3
    if EXPOSURE_TAG in out:
        log("WARNING: the paid feed is exposed to the LAN. Not blocking the session --")
        log("         it is a security issue, not a data-integrity one -- but fix it.")

    # ---- run, supervised --------------------------------------------------
    log("preflight clean; starting runner")
    # The ledger is what makes a gap in the record readable. Written locally, needing
    # no feed and no entitlement, so it survives exactly the failures it documents.
    ledger.note_opened(day, args.lab_dir, arms=[n for n, _ in _arm_specs(args)])
    rc = supervise(args, day)
    ledger.note_finished(day, args.lab_dir, supervisor_rc=rc)
    return rc


if __name__ == "__main__":
    sys.exit(main())
