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
from .lock import SingleInstance
from .runner import EXIT_ALREADY_RUNNING
from .store import DEFAULT_LAB_DIR

START_AT = dt.time(9, 20)          # exchange time; structural preflight runs here
FRESHNESS_AT = dt.time(9, 33)      # RTH freshness re-check -- MUST be after the open,
                                   # otherwise the delayed-feed gate can never fire.
                                   # Costs nothing: no setup can signal before 09:36
                                   # because the 09:30-09:35 opening range is not formed.
GIVE_UP_AFTER = dt.time(15, 30)    # too late in the session to bother starting
HOLIDAY_CHECK_AT = dt.time(9, 45)
RUNNER_DONE_AFTER = dt.time(15, 55)  # runner flattens at 15:55; an exit at/after this is
                                     # a normal end of session, not a crash to restart
MAX_RESTARTS = 20
RESTART_BACKOFF = [5, 15, 30, 60, 120]   # seconds; holds at the last value
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


def start_terminal(wait_sec=90) -> bool:
    if not Path(TERMINAL_CMD[0]).exists():
        log(f"cannot start terminal: {TERMINAL_CMD[0]} not found")
        return False
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
            return True
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
        [sys.executable, "-m", "trade_analysis.live_lab.preflight", "--symbols", *symbols],
        capture_output=True, text=True,
        cwd=str(Path(__file__).resolve().parents[2]))
    return pf.stdout or ""


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
    """The arms to supervise. Separated out so tests can inject fakes."""
    root_args = ["--symbols", *args.symbols]
    specs = [("options", [sys.executable, "-m",
                          "trade_analysis.live_lab.runner", *root_args,
                          "--lab-dir", args.lab_dir,
                          "--contracts", str(args.contracts)])]
    if not getattr(args, "no_shares", False):
        specs.append(("shares", [sys.executable, "-m",
                                 "trade_analysis.live_lab.shares_runner",
                                 *root_args, "--lab-dir",
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
        if now.date() != day or now.time() >= RUNNER_DONE_AFTER:
            if any(not ch.done for ch in children):
                log(f"{now:%H:%M:%S} ET -- session over; stopping any arm still running")
            for ch in children:
                ch.terminate()
            break
        if all(ch.done for ch in children):
            log("every arm has finished; nothing left to supervise")
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
    ap.add_argument("--symbols", nargs="+", default=["QQQ", "SPY"])
    ap.add_argument("--lab-dir", default=str(DEFAULT_LAB_DIR))
    ap.add_argument("--start-terminal", action="store_true")
    ap.add_argument("--now", action="store_true", help="skip the wait and start immediately")
    ap.add_argument("--contracts", type=float, default=1.0)
    ap.add_argument("--no-shares", action="store_true",
                    help="run only the frozen options arm")
    args = ap.parse_args(argv)

    day = now_et().date()
    sys.stdout = Tee(Path(args.lab_dir) / "logs" / f"{day.isoformat()}.log")

    log("=" * 66)
    log(f"autostart invoked | local {dt.datetime.now():%H:%M:%S} | symbols {args.symbols}")

    if day.weekday() >= 5:
        log(f"{day} is a weekend; nothing to do")
        return 0

    # The scheduled task repeats through the day so a machine that boots late, or a
    # supervisor that dies with its whole process tree, still gets the lab running. That
    # only works if a second invocation is a no-op while one is already alive.
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
            return 2

    feed = ThetaLiveFeed()
    if is_holiday(feed, args.symbols, day):
        log(f"no bars for any symbol well after the open -- {day} looks like a market "
            f"holiday; exiting rather than idling")
        feed.close()
        return 0
    feed.close()

    # ---- structural preflight, pre-open ------------------------------------
    log("running preflight (structural, pre-open) ...")
    out = _preflight(args.symbols)
    for line in out.splitlines():
        log("  " + line)
    if [l for l in out.splitlines() if "[FAIL]" in l and "REACHABLE from" not in l]:
        log("ABORT: structural preflight failed")
        return 3

    # ---- freshness re-check, AFTER the open --------------------------------
    # This is the gate that matters. At 09:20 the market is shut, so freshness is
    # always "UNVERIFIABLE" and a delayed feed would sail straight through. Day one
    # ran with the check effectively disabled; this is the fix.
    if not args.now:
        if not wait_for_open(FRESHNESS_AT):
            return 0
        log("re-running preflight for FEED FRESHNESS (market now open) ...")
        out = _preflight(args.symbols)
    for line in out.splitlines():
        log("  " + line)

    fatal = [l for l in out.splitlines()
             if "[FAIL]" in l and "REACHABLE from" not in l]
    if any("DELAYED" in l for l in out.splitlines()):
        log("ABORT: the feed is DELAYED. Entries and exits would be priced off stale")
        log("       quotes, which corrupts the record rather than merely degrading it.")
        return 3
    if fatal:
        log(f"ABORT: {len(fatal)} preflight data failure(s)")
        return 3
    if "REACHABLE from" in out:
        log("WARNING: the paid feed is exposed to the LAN. Not blocking the session --")
        log("         it is a security issue, not a data-integrity one -- but fix it.")

    # ---- run, supervised --------------------------------------------------
    log("preflight clean; starting runner")
    return supervise(args, day)


if __name__ == "__main__":
    sys.exit(main())
