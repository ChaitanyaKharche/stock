"""Hypothetical position management.

No orders. Ever. A Position is a record of what a trade WOULD have been, priced from
observed NBBO only: entered at the ask, exited at the bid, with the real broker fee
applied to both sides.

Each signal produces up to three positions -- one per strike arm (ATM, ATM-1, ATM+1) --
sharing a signal_id, so the arms are directly comparable on identical entry/exit instants.

The UNDERLYING TWIN travels with every position: the same signal's forward move in the
underlying over the identical holding period. That is what separates
    "this setup has no directional edge"          (twin flat/negative)
from
    "direction is right, the option structure eats it"   (twin positive, option negative)
Those two demand opposite responses and no prior study in this project could tell them
apart.
"""
from __future__ import annotations

import datetime as dt
import uuid
from dataclasses import dataclass, field, asdict

FEE_PER_CONTRACT_SIDE = 0.0404      # the trader's real broker rate, measured from his tape
CONTRACT_MULTIPLIER = 100.0
EOD_FLAT = dt.time(15, 55)


@dataclass
class Position:
    position_id: str
    signal_id: str
    config_hash: str
    setup_id: str
    setup_version: str
    symbol: str
    arm: str                              # "ATM" | "ATM-1" | "ATM+1"
    direction: str                        # "long" | "short" (of the UNDERLYING view)
    right: str                            # "call" | "put"
    strike: float
    expiration: str
    contracts: float

    entry_ts: str
    entry_bar_ts: str
    entry_ask: float
    entry_bid: float
    entry_spread_pct: float
    entry_underlying: float
    iv_derived: float | None
    delta_derived: float | None
    open_interest: float | None

    stop: float | None
    target: float | None
    time_exit_min: int | None
    bar_exit: int | None
    trailing: str | None
    state: dict

    # --- mutable while open ---
    bars_held: int = 0
    mfe_opt: float = 0.0                  # max favourable excursion, option return
    mae_opt: float = 0.0
    mfe_opt_ts: str | None = None
    mae_opt_ts: str | None = None
    mfe_und: float = 0.0                  # underlying twin, signed to the trade direction
    mae_und: float = 0.0
    last_bid: float | None = None
    last_underlying: float | None = None

    exit_ts: str | None = None
    exit_bid: float | None = None
    exit_underlying: float | None = None
    exit_reason: str | None = None
    trigger_level: float | None = None    # the theoretical stop/target that fired

    # ------------------------------------------------------------------ marks

    def mark(self, opt_bid: float | None, underlying: float, ts: dt.datetime) -> None:
        """Update excursions. Called on every poll while open. Never changes the decision."""
        if opt_bid is not None and self.entry_ask > 0:
            r = opt_bid / self.entry_ask - 1.0
            if r > self.mfe_opt:
                self.mfe_opt, self.mfe_opt_ts = r, ts.isoformat()
            if r < self.mae_opt:
                self.mae_opt, self.mae_opt_ts = r, ts.isoformat()
            self.last_bid = opt_bid
        sign = 1.0 if self.direction == "long" else -1.0
        u = sign * (underlying / self.entry_underlying - 1.0)
        self.mfe_und = max(self.mfe_und, u)
        self.mae_und = min(self.mae_und, u)
        self.last_underlying = underlying

    # ------------------------------------------------------------------ exit tests

    def check_underlying_exit(self, bar: dict, now: dt.datetime) -> tuple[str, float] | None:
        """Stop/target on the underlying, detected on the bar's HIGH/LOW.

        The trigger is a level touch (that is how the sources define it); the FILL is
        always taken from the next observed option quote, never from the level. The gap
        between the two is recorded, not assumed away.
        """
        if self.direction == "long":
            if self.stop is not None and bar["low"] <= self.stop:
                return ("stop", self.stop)
            if self.target is not None and bar["high"] >= self.target:
                return ("target", self.target)
        else:
            if self.stop is not None and bar["high"] >= self.stop:
                return ("stop", self.stop)
            if self.target is not None and bar["low"] <= self.target:
                return ("target", self.target)
        return None

    def check_clock_exit(self, now: dt.datetime) -> str | None:
        if now.time() >= EOD_FLAT:
            return "eod"
        if self.time_exit_min is not None:
            entered = dt.datetime.fromisoformat(self.entry_ts)
            if (now - entered).total_seconds() / 60.0 >= self.time_exit_min:
                return "time"
        if self.bar_exit is not None and self.bars_held >= self.bar_exit:
            return "bars"
        return None

    # ------------------------------------------------------------------ close

    def close(self, *, bid: float | None, underlying: float, ts: dt.datetime,
              reason: str, trigger_level: float | None = None) -> dict:
        self.exit_ts = ts.isoformat()
        self.exit_bid = bid
        self.exit_underlying = underlying
        self.exit_reason = reason
        self.trigger_level = trigger_level
        return self.to_trade()

    # ------------------------------------------------------------------ output

    def to_trade(self) -> dict:
        gross = net = ret = None
        if self.exit_bid is not None:
            gross = (self.exit_bid - self.entry_ask) * self.contracts * CONTRACT_MULTIPLIER
            net = gross - 2.0 * FEE_PER_CONTRACT_SIDE * self.contracts
            ret = (self.exit_bid / self.entry_ask - 1.0) if self.entry_ask > 0 else None
        hold = None
        if self.exit_ts:
            hold = (dt.datetime.fromisoformat(self.exit_ts)
                    - dt.datetime.fromisoformat(self.entry_ts)).total_seconds() / 60.0
        sign = 1.0 if self.direction == "long" else -1.0
        und_ret = None
        if self.exit_underlying and self.entry_underlying:
            und_ret = sign * (self.exit_underlying / self.entry_underlying - 1.0)
        d = asdict(self)
        d.update({
            "pnl_gross": gross, "pnl_net": net, "return_pct": ret,
            "hold_minutes": hold, "underlying_return": und_ret,
            "fee_per_contract_side": FEE_PER_CONTRACT_SIDE,
        })
        return d


def open_positions_from_signal(*, signal_id, config_hash, setup, sig, symbol,
                               arms: dict, expiration, spot_mid, now, bar_ts,
                               oi: dict, contracts: float = 1.0) -> list[Position]:
    """Build one Position per available strike arm. ATM is guaranteed present by caller."""
    right = "call" if sig.direction == "long" else "put"
    out = []
    for arm in ("ATM", "ATM-1", "ATM+1"):
        q = arms.get(arm)
        if q is None:
            continue
        out.append(Position(
            position_id=uuid.uuid4().hex,
            signal_id=signal_id,
            config_hash=config_hash,
            setup_id=setup.id,
            setup_version=setup.version,
            symbol=symbol,
            arm=arm,
            direction=sig.direction,
            right=right,
            strike=q["strike"],
            expiration=expiration.isoformat(),
            contracts=contracts,
            entry_ts=now.isoformat(),
            entry_bar_ts=bar_ts.isoformat(),
            entry_ask=q["ask"],
            entry_bid=q["bid"],
            entry_spread_pct=q.get("spread_pct_of_mid", 0.0),
            entry_underlying=spot_mid,
            iv_derived=q.get("iv_derived"),
            delta_derived=q.get("delta_derived"),
            open_interest=oi.get((q["strike"], right)),
            stop=sig.stop,
            target=sig.target,
            time_exit_min=sig.time_exit_min,
            bar_exit=sig.bar_exit,
            trailing=sig.trailing,
            state=dict(sig.state),
        ))
    return out


def position_from_dict(d: dict) -> Position:
    """Crash recovery: rebuild an open Position from positions_open.json."""
    fields = set(Position.__dataclass_fields__)
    return Position(**{k: v for k, v in d.items() if k in fields})
