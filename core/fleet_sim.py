"""Discrete-event fleet simulation for the dashboard: flights land, engines age, alerts fire.

Each aircraft flies a fixed rotation (cycles per day, one hour on the ground between
flights). Every landing adds one cycle to both engines and triggers a post-landing review.
An engine that reaches the end of its simulated life grounds its aircraft (AOG) until
authorised maintenance replaces it.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from core.planning import TIER_RANK, PlanningContext, assign_tier


@dataclass
class Aircraft:
    aircraft_id: str
    base: str
    cycles_per_day: float
    engines: list[str]
    last_landing: pd.Timestamp
    next_departure: pd.Timestamp
    next_landing: pd.Timestamp
    grounded_since: pd.Timestamp | None = None
    grounded_reason: str = ""

    @property
    def interval(self) -> pd.Timedelta:
        return pd.Timedelta(hours=24.0 / self.cycles_per_day)

    @property
    def flying(self) -> bool:
        return self.grounded_since is None


@dataclass
class FleetState:
    now: pd.Timestamp
    cycles: dict[str, int]
    aircraft: dict[str, Aircraft]
    turnaround: pd.Timedelta
    replaced: dict[str, pd.Timestamp] = field(default_factory=dict)
    log: list[dict] = field(default_factory=list)
    aog_closed_hours: float = 0.0

    def aircraft_of(self, engine_id: str) -> Aircraft:
        for ac in self.aircraft.values():
            if engine_id in ac.engines:
                return ac
        raise KeyError(engine_id)

    def active_engines(self) -> list[str]:
        return [e for ac in self.aircraft.values() for e in ac.engines if e not in self.replaced]


def init_state(ctx: PlanningContext, cfg: dict) -> FleetState:
    tz = cfg["timezone"]
    start = pd.Timestamp(cfg["simulation"]["start"]).tz_convert(tz)
    turn = pd.Timedelta(hours=float(cfg["simulation"]["turnaround_hours"]))
    aircraft, cycles = {}, {}
    for ac_id, g in ctx.tables.fleet.groupby("aircraft_id", sort=True):
        cpd = float(g["cycles_per_day"].iloc[0])
        dep = g["next_departure"].iloc[0]
        block = pd.Timedelta(hours=24.0 / cpd) - turn
        aircraft[ac_id] = Aircraft(
            aircraft_id=ac_id, base=str(g["base_location"].iloc[0]), cycles_per_day=cpd,
            engines=list(g["engine_id"]), last_landing=dep - turn, next_departure=dep, next_landing=dep + block,
        )
        for eid, cyc in zip(g["engine_id"], g["current_cycle"]):
            cycles[eid] = int(cyc)
    return FleetState(now=start, cycles=cycles, aircraft=aircraft, turnaround=turn)


def _land(state: FleetState, ctx: PlanningContext, cfg: dict, ac: Aircraft) -> dict:
    t = ac.next_landing
    engines = []
    for eid in ac.engines:
        if eid in state.replaced:
            continue
        last = ctx.predictions.last_cycle(eid)
        before = state.cycles[eid]
        after = min(before + 1, last)
        state.cycles[eid] = after
        p_before, p_after = ctx.predictions.get(eid, before), ctx.predictions.get(eid, after)
        engines.append({
            "engine_id": eid, "cycle": after,
            "tier_before": assign_tier(p_before["rul_lower"], cfg), "tier_after": assign_tier(p_after["rul_lower"], cfg),
            "rul_lower": float(p_after["rul_lower"]), "predicted_rul": float(p_after["predicted_rul"]),
            "end_of_life": after >= last,
        })
    ac.last_landing = t
    failed = [e["engine_id"] for e in engines if e["end_of_life"]]
    if failed:
        ac.grounded_since = t
        ac.grounded_reason = f"{', '.join(failed)} reached the end of its simulated life"
        ac.next_departure = pd.NaT
        ac.next_landing = pd.NaT
    else:
        ac.next_departure = t + state.turnaround
        ac.next_landing = ac.next_departure + (ac.interval - state.turnaround)
    event = {
        "time": t, "aircraft_id": ac.aircraft_id, "base": ac.base, "engines": engines,
        "next_departure": ac.next_departure, "grounded": bool(failed),
        "tier_changed": any(e["tier_before"] != e["tier_after"] for e in engines),
        "worst_tier": min((e["tier_after"] for e in engines), key=lambda x: TIER_RANK[x], default=None),
    }
    state.log.append(event)
    return event


def advance_to(state: FleetState, ctx: PlanningContext, cfg: dict, target: pd.Timestamp) -> list[dict]:
    """Process every landing up to and including `target`, in time order."""
    events = []
    while True:
        flying = [ac for ac in state.aircraft.values() if ac.flying and pd.notna(ac.next_landing)]
        if not flying:
            break
        ac = min(flying, key=lambda a: a.next_landing)
        if ac.next_landing > target:
            break
        events.append(_land(state, ctx, cfg, ac))
    state.now = max(state.now, target)
    return events


def land_next(state: FleetState, ctx: PlanningContext, cfg: dict, aircraft_id: str | None = None) -> list[dict]:
    """Advance to the next landing of one aircraft (or of the whole fleet when aircraft_id is None)."""
    flying = [ac for ac in state.aircraft.values() if ac.flying and pd.notna(ac.next_landing)]
    if aircraft_id is not None:
        ac = state.aircraft[aircraft_id]
        if not ac.flying:
            return []
        target = ac.next_landing
    elif flying:
        target = min(a.next_landing for a in flying)
    else:
        return []
    return advance_to(state, ctx, cfg, target)


def advance_hours(state: FleetState, ctx: PlanningContext, cfg: dict, hours: float) -> list[dict]:
    return advance_to(state, ctx, cfg, state.now + pd.Timedelta(hours=hours))


def complete_maintenance(state: FleetState, ctx: PlanningContext, engine_id: str) -> None:
    """Authorised maintenance replaced the engine now; return the aircraft to service if it was grounded."""
    ac = state.aircraft_of(engine_id)
    state.replaced[engine_id] = state.now
    still_failed = [
        e for e in ac.engines
        if e not in state.replaced and state.cycles[e] >= ctx.predictions.last_cycle(e)
    ]
    if not ac.flying and not still_failed:
        state.aog_closed_hours += (state.now - ac.grounded_since).total_seconds() / 3600
        ac.grounded_since = None
        ac.grounded_reason = ""
        ac.next_departure = state.now + state.turnaround
        ac.next_landing = ac.next_departure + (ac.interval - state.turnaround)


def open_aog_hours(state: FleetState) -> float:
    return sum(
        (state.now - ac.grounded_since).total_seconds() / 3600 for ac in state.aircraft.values() if not ac.flying
    )
