"""Reactive vs predictive spare-parts policy on the synthetic fleet (measured, not assumed).

Both policies start from the same fleet state and the same stock, and replay every fleet
engine to the end of its simulated life:

  Reactive    no forecast; when an engine reaches end of life the fastest available part
              is sourced (an AOG situation).
  Predictive  the planning engine runs after every landing. The first time an engine
              enters Plan or Review, or its order window opens, the recommended option is
              taken (stock reserved or order placed). Estimates are re-checked at every
              landing, and the engine is removed at the first landing where the lower edge of
              its band is within the safety buffer. If the part has not arrived, the aircraft
              waits (AOG). An engine that reaches end of life first counts as unscheduled.
              When several engines are reviewed at the same moment, the most urgent goes first.

Stock taken by one engine is unavailable to the others. Orders are dedicated to the engine.
This is a synthetic scenario: it shows the mechanism, not real-world savings.
"""
from __future__ import annotations

import math

import pandas as pd

from core.planning import (
    KIND_EXPEDITED,
    TIER_MONITOR,
    PlanningContext,
    assign_tier,
    best_option,
    describe_option,
    free_stock,
    need_by_time,
)


class EngineClock:
    """Maps cycles to landing times for one engine on its aircraft's rotation."""

    def __init__(self, start: pd.Timestamp, first_departure: pd.Timestamp, cycles_per_day: float,
                 turnaround: pd.Timedelta, start_cycle: int):
        self.start = start
        self.interval = pd.Timedelta(hours=24.0 / cycles_per_day)
        self.first_landing = first_departure + (self.interval - turnaround)
        self.c0 = int(start_cycle)

    def time_of(self, cycle: int) -> pd.Timestamp:
        """Landing time at which `cycle` completes (the start time for the starting cycle)."""
        if cycle <= self.c0:
            return self.start
        return self.first_landing + (cycle - self.c0 - 1) * self.interval

    def cycle_at(self, t: pd.Timestamp) -> int:
        if t < self.first_landing:
            return self.c0
        return self.c0 + 1 + int(math.floor((t - self.first_landing) / self.interval))


def _setup(ctx: PlanningContext, cfg: dict):
    tz = cfg["timezone"]
    start = pd.Timestamp(cfg["simulation"]["start"]).tz_convert(tz)
    turn = pd.Timedelta(hours=float(cfg["simulation"]["turnaround_hours"]))
    engines = {}
    for row in ctx.tables.fleet.to_dict("records"):
        eid = row["engine_id"]
        clock = EngineClock(start, row["next_departure"], float(row["cycles_per_day"]), turn, int(row["current_cycle"]))
        last = ctx.predictions.last_cycle(eid)
        engines[eid] = {
            "aircraft_id": row["aircraft_id"], "base": row["base_location"], "cpd": float(row["cycles_per_day"]),
            "clock": clock, "start_cycle": int(row["current_cycle"]), "last_cycle": last, "failure": clock.time_of(last),
        }
    return start, engines


def _record(eid, e, policy, trigger_time, trigger_cycle, opt, need_by, removal, unscheduled, aog_h, unused):
    return {
        "policy": policy, "engine_id": eid, "aircraft_id": e["aircraft_id"], "base": e["base"],
        "trigger_time": trigger_time, "trigger_cycle": trigger_cycle, "action": describe_option(opt),
        "kind": opt["kind"], "mode": opt["mode"], "need_by": need_by, "arrival": trigger_time + pd.Timedelta(hours=opt["lead_hours"]),
        "removal_time": removal, "failure_time": e["failure"], "unscheduled_removal": unscheduled,
        "aog_hours": aog_h, "cost_usd": float(opt["cost_usd"]), "co2e_kg": float(opt["co2e_kg"]),
        "air_shipment": opt["mode"] == "air", "expedited_order": opt["kind"] == KIND_EXPEDITED,
        "unused_cycles": unused,
    }


def run_reactive(ctx: PlanningContext, cfg: dict) -> pd.DataFrame:
    _, engines = _setup(ctx, cfg)
    stock = free_stock(ctx.tables.inventory)
    rows = []
    for eid, e in sorted(engines.items(), key=lambda kv: kv[1]["failure"]):
        part = ctx.part_for_engine(eid)
        options = ctx.build_options(part, e["base"], stock)
        opt = min(options, key=lambda o: (o["lead_hours"], o["cost_usd"]))
        if isinstance(opt["stock_location"], str):
            stock[(opt["stock_location"], part)] -= 1
        t_f = e["failure"]
        rows.append(_record(eid, e, "Reactive", t_f, e["last_cycle"], opt, t_f, t_f, True, float(opt["lead_hours"]), 0))
    return pd.DataFrame(rows)


def run_predictive(ctx: PlanningContext, cfg: dict, w_co2: float) -> pd.DataFrame:
    _, engines = _setup(ctx, cfg)
    stock = free_stock(ctx.tables.inventory)
    buffer = float(cfg["planning"]["safety_buffer_cycles"])
    window = pd.Timedelta(days=float(cfg["planning"]["order_window_days"]))

    events = []
    for eid, e in engines.items():
        for c in range(e["start_cycle"], e["last_cycle"] + 1):
            lower = float(ctx.predictions.get(eid, c)["rul_lower"])
            events.append((e["clock"].time_of(c), lower, eid, c))
    events.sort(key=lambda ev: (ev[0], ev[1], ev[2]))  # same moment: most urgent first

    secured: dict[str, dict] = {}
    rows = []
    for t, lower, eid, c in events:
        e = engines[eid]
        if e.get("removed"):
            continue
        part = ctx.part_for_engine(eid)
        if eid not in secured:
            tier = assign_tier(lower, cfg)
            need_by = need_by_time(t, lower, e["cpd"], buffer)
            opt, _ = best_option(ctx.build_options(part, e["base"], stock), t, need_by, w_co2)
            if opt is not None:
                order_by = need_by - pd.Timedelta(hours=opt["lead_hours"])
                if tier != TIER_MONITOR or order_by <= t + window or c == e["last_cycle"]:
                    if isinstance(opt["stock_location"], str):
                        stock[(opt["stock_location"], part)] -= 1
                    secured[eid] = {"opt": opt, "time": t, "cycle": c, "need_by": need_by}
        at_end = c == e["last_cycle"]
        if lower <= buffer or at_end:
            s = secured[eid]
            arrival = s["time"] + pd.Timedelta(hours=s["opt"]["lead_hours"])
            aog_h = max(0.0, (arrival - t).total_seconds() / 3600)
            rows.append(_record(
                eid, e, "Predictive", s["time"], s["cycle"], s["opt"], s["need_by"], t, at_end, aog_h,
                e["last_cycle"] - c,
            ))
            e["removed"] = True
    return pd.DataFrame(rows)


def summarise(df: pd.DataFrame) -> dict:
    return {
        "Unscheduled removals": int(df["unscheduled_removal"].sum()),
        "AOG hours waiting for parts": round(float(df["aog_hours"].sum()), 1),
        "Air shipments": int(df["air_shipment"].sum()),
        "Expedited orders": int(df["expedited_order"].sum()),
        "Parts cost incl. logistics (USD)": round(float(df["cost_usd"].sum())),
        "Freight emissions (kg CO2e)": round(float(df["co2e_kg"].sum()), 1),
        "Mean unused life (cycles)": round(float(df["unused_cycles"].mean()), 1),
    }


def compare(ctx: PlanningContext, cfg: dict, w_co2: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (per-engine rows for both policies, summary table with one column per policy)."""
    reactive = run_reactive(ctx, cfg)
    predictive = run_predictive(ctx, cfg, w_co2)
    summary = pd.DataFrame({"Reactive": summarise(reactive), "Predictive": summarise(predictive)})
    return pd.concat([reactive, predictive], ignore_index=True), summary
