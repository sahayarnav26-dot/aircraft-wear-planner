"""Planning engine: turns a post-landing RUL estimate into a ranked parts recommendation.

It always acts on the cautious lower edge of the RUL band. Nothing is reserved, moved or
ordered here: functions return recommendations, and a person approves or overrides them.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from core.data import Tables

TIER_REVIEW = "Review before next departure"
TIER_PLAN = "Plan maintenance"
TIER_MONITOR = "Monitor"
TIER_RANK = {TIER_REVIEW: 0, TIER_PLAN: 1, TIER_MONITOR: 2}

KIND_RESERVE = "Reserve local stock"
KIND_TRANSFER = "Transfer"
KIND_STANDARD = "Standard order"
KIND_EXPEDITED = "Expedited order"

ACTION_REVIEW = "Inspect before next departure and secure a part"
ACTION_PLAN = "Plan maintenance and secure a part"
ACTION_ORDER = "Order window open"
ACTION_MONITOR = "Monitor"

OPTION_COLUMNS = [
    "option_id", "kind", "source", "source_name", "mode", "lead_hours", "arrival", "on_time",
    "order_by", "cost_usd", "co2e_kg", "distance_km", "stock_location", "score", "recommended",
]


# ---------------------------------------------------------------------------------------
# Timing and tiers
# ---------------------------------------------------------------------------------------
def assign_tier(rul_lower: float, cfg: dict) -> str:
    p = cfg["planning"]
    if rul_lower <= p["review_max_rul_lower"]:
        return TIER_REVIEW
    if rul_lower <= p["plan_max_rul_lower"]:
        return TIER_PLAN
    return TIER_MONITOR


def need_by_time(landing_time: pd.Timestamp, rul_lower: float, cycles_per_day: float, buffer_cycles: float) -> pd.Timestamp:
    """t_need = t_landing + (RUL_lower - buffer) / cycles per day; never earlier than the landing."""
    usable = max(0.0, float(rul_lower) - float(buffer_cycles))
    return landing_time + pd.Timedelta(hours=24.0 * usable / float(cycles_per_day))


# ---------------------------------------------------------------------------------------
# Emissions
# ---------------------------------------------------------------------------------------
class EmissionFactors:
    """kg CO2e per tonne-km by mode; air legs pick short- or long-haul by distance."""

    def __init__(self, table: pd.DataFrame, long_haul_km: float):
        self.factors = dict(zip(table["mode"], table["kg_co2e_per_tonne_km"].astype(float)))
        self.long_haul_km = float(long_haul_km)

    def key(self, mode: str, distance_km: float) -> str:
        if mode == "air":
            return "air_long_haul" if distance_km > self.long_haul_km else "air_short_haul"
        return mode

    def factor(self, mode: str, distance_km: float) -> float:
        return self.factors[self.key(mode, distance_km)]

    def co2e_kg(self, weight_kg: float, distance_km: float, mode: str) -> float:
        if mode == "none" or distance_km <= 0:
            return 0.0
        return weight_kg / 1000.0 * distance_km * self.factor(mode, distance_km)


# ---------------------------------------------------------------------------------------
# Stock helpers
# ---------------------------------------------------------------------------------------
def free_stock(inventory: pd.DataFrame) -> dict[tuple[str, str], int]:
    """(location_id, part_id) -> qty_on_hand - qty_reserved, never below zero."""
    free = (inventory["qty_on_hand"] - inventory["qty_reserved"]).clip(lower=0).astype(int)
    return dict(zip(zip(inventory["location_id"], inventory["part_id"]), free))


def reserve_stock(inventory: pd.DataFrame, location_id: str, part_id: str, qty: int = 1) -> pd.DataFrame:
    """Return a copy of inventory with qty more units reserved at location_id."""
    inv = inventory.copy()
    mask = (inv["location_id"] == location_id) & (inv["part_id"] == part_id)
    if not mask.any():
        raise KeyError(f"No inventory row for {part_id} at {location_id}")
    free = int((inv.loc[mask, "qty_on_hand"] - inv.loc[mask, "qty_reserved"]).iloc[0])
    if free < qty:
        raise ValueError(f"Only {free} free {part_id} at {location_id}")
    inv.loc[mask, "qty_reserved"] = inv.loc[mask, "qty_reserved"] + qty
    return inv


# ---------------------------------------------------------------------------------------
# Context: lookups built once from the tables
# ---------------------------------------------------------------------------------------
class PlanningContext:
    def __init__(self, tables: Tables, cfg: dict):
        self.tables = tables
        self.cfg = cfg
        p = cfg["planning"]
        self.buffer = float(p["safety_buffer_cycles"])
        self.window = pd.Timedelta(days=float(p["order_window_days"]))
        self.ef = EmissionFactors(tables.emission_factors, p["air_long_haul_km"])
        self.parts = tables.parts.set_index("part_id")
        self.location_names = dict(zip(tables.locations["location_id"], tables.locations["name"]))
        self.transfers_to: dict[str, list[dict]] = {}
        for row in tables.transfers.to_dict("records"):
            self.transfers_to.setdefault(row["to_location"], []).append(row)
        self.suppliers_for: dict[str, list[dict]] = {}
        for row in tables.suppliers.to_dict("records"):
            self.suppliers_for.setdefault(row["part_id"], []).append(row)
        self.fleet = tables.fleet.set_index("engine_id")
        self.predictions = PredictionIndex(tables.predictions)

    def part_for_engine(self, engine_id: str) -> str:
        p = self.cfg["planning"]
        return p["component_part"][p["default_component"]]

    def build_options(self, part_id: str, base: str, stock: dict[tuple[str, str], int]) -> list[dict]:
        """All ways to get one unit of part_id to base, independent of time."""
        part = self.parts.loc[part_id]
        weight, value = float(part["unit_weight_kg"]), float(part["unit_cost_usd"])
        options = []
        if stock.get((base, part_id), 0) > 0:
            options.append(dict(
                kind=KIND_RESERVE, source=base, source_name=self.location_names.get(base, base), mode="none",
                lead_hours=0.0, cost_usd=value, co2e_kg=0.0, distance_km=0.0, stock_location=base,
            ))
        for t in self.transfers_to.get(base, []):
            src = t["from_location"]
            if stock.get((src, part_id), 0) <= 0:
                continue
            options.append(dict(
                kind=KIND_TRANSFER, source=src, source_name=self.location_names.get(src, src), mode=t["mode"],
                lead_hours=float(t["transit_hours"]), cost_usd=value + float(t["cost_usd"]),
                co2e_kg=self.ef.co2e_kg(weight, float(t["distance_km"]), t["mode"]),
                distance_km=float(t["distance_km"]), stock_location=src,
            ))
        for s in self.suppliers_for.get(part_id, []):
            dist = float(s["distance_km"])
            options.append(dict(
                kind=KIND_STANDARD, source=s["supplier_id"], source_name=f"{s['supplier_name']}, {s['origin_location']}",
                mode=s["standard_mode"], lead_hours=24.0 * float(s["lead_time_days_standard"]),
                cost_usd=float(s["unit_price_usd"]), co2e_kg=self.ef.co2e_kg(weight, dist, s["standard_mode"]),
                distance_km=dist, stock_location=None,
            ))
            options.append(dict(
                kind=KIND_EXPEDITED, source=s["supplier_id"], source_name=f"{s['supplier_name']}, {s['origin_location']}",
                mode="air", lead_hours=24.0 * float(s["lead_time_days_expedited"]),
                cost_usd=float(s["unit_price_usd"]) + float(s["expedite_fee_usd"]),
                co2e_kg=self.ef.co2e_kg(weight, dist, "air"), distance_km=dist, stock_location=None,
            ))
        return options


class PredictionIndex:
    """Fast (engine_id, cycle) -> prediction lookup."""

    def __init__(self, predictions: pd.DataFrame):
        self.by_engine: dict[str, pd.DataFrame] = {
            eid: g.set_index("cycle").sort_index() for eid, g in predictions.groupby("engine_id", sort=False)
        }
        self._rows: dict[tuple[str, int], dict] = {}
        for eid, g in self.by_engine.items():
            recs = g[["predicted_rul", "rul_lower", "rul_upper", "true_rul", "top_sensors"]].to_dict("index")
            for cycle, rec in recs.items():
                self._rows[(eid, int(cycle))] = rec

    def get(self, engine_id: str, cycle: int) -> dict:
        try:
            return self._rows[(engine_id, int(cycle))]
        except KeyError as exc:
            raise KeyError(f"No prediction for {engine_id} at cycle {cycle}") from exc

    def first_cycle(self, engine_id: str) -> int:
        return int(self.by_engine[engine_id].index.min())

    def last_cycle(self, engine_id: str) -> int:
        return int(self.by_engine[engine_id].index.max())

    def history(self, engine_id: str, upto_cycle: int | None = None) -> pd.DataFrame:
        g = self.by_engine[engine_id]
        return g if upto_cycle is None else g.loc[: int(upto_cycle)]


# ---------------------------------------------------------------------------------------
# Ranking
# ---------------------------------------------------------------------------------------
def _normalise(values: np.ndarray) -> np.ndarray:
    lo, hi = float(np.min(values)), float(np.max(values))
    return np.zeros_like(values) if hi - lo < 1e-9 else (values - lo) / (hi - lo)


def rank_options(options: list[dict], now: pd.Timestamp, need_by: pd.Timestamp, w_co2: float) -> pd.DataFrame:
    """Mark on-time options, score them, and sort best first.

    score = (1 - w_co2) x cost + w_co2 x CO2e, each normalised across the on-time options.
    Late options get no score; if nothing is on time, the fastest option is recommended.
    """
    if not options:
        return pd.DataFrame(columns=OPTION_COLUMNS)
    df = pd.DataFrame(options)
    df["arrival"] = [now + pd.Timedelta(hours=h) for h in df["lead_hours"]]
    df["on_time"] = df["arrival"] <= need_by
    df["order_by"] = [need_by - pd.Timedelta(hours=h) for h in df["lead_hours"]]
    df["score"] = np.nan
    feasible = df["on_time"].to_numpy()
    if feasible.any():
        cost_n = _normalise(df.loc[feasible, "cost_usd"].to_numpy(float))
        co2_n = _normalise(df.loc[feasible, "co2e_kg"].to_numpy(float))
        df.loc[feasible, "score"] = (1.0 - w_co2) * cost_n + w_co2 * co2_n
        df = df.sort_values(["on_time", "score", "lead_hours", "cost_usd"], ascending=[False, True, True, True])
    else:
        df = df.sort_values(["lead_hours", "cost_usd"])
    df = df.reset_index(drop=True)
    df["recommended"] = False
    df.loc[0, "recommended"] = True
    df["option_id"] = [f"opt-{i + 1}" for i in range(len(df))]
    return df[OPTION_COLUMNS]


def best_option(options: list[dict], now: pd.Timestamp, need_by: pd.Timestamp, w_co2: float) -> tuple[dict | None, bool]:
    """Fast path with the same rule as rank_options: (best option, on_time). Used by simulations."""
    if not options:
        return None, False
    avail_h = (need_by - now).total_seconds() / 3600.0
    feasible = [o for o in options if o["lead_hours"] <= avail_h + 1e-9]
    if not feasible:
        return min(options, key=lambda o: (o["lead_hours"], o["cost_usd"])), False
    cost_n = _normalise(np.array([o["cost_usd"] for o in feasible], dtype=float))
    co2_n = _normalise(np.array([o["co2e_kg"] for o in feasible], dtype=float))
    scores = (1.0 - w_co2) * cost_n + w_co2 * co2_n
    k = min(range(len(feasible)), key=lambda i: (scores[i], feasible[i]["lead_hours"], feasible[i]["cost_usd"]))
    return feasible[k], True


# ---------------------------------------------------------------------------------------
# Recommendation
# ---------------------------------------------------------------------------------------
@dataclass
class Recommendation:
    engine_id: str
    aircraft_id: str
    base: str
    cycle: int
    now: pd.Timestamp
    cycles_per_day: float
    predicted_rul: float
    rul_lower: float
    rul_upper: float
    true_rul: float
    top_sensors: list[str]
    tier: str
    need_by: pd.Timestamp
    part_id: str
    part_name: str
    options: pd.DataFrame
    escalated: bool
    shortfall_hours: float
    order_window_open: bool
    action: str
    rationale: str = ""
    next_departure: pd.Timestamp | None = None
    extra: dict = field(default_factory=dict)

    @property
    def chosen(self) -> dict | None:
        if self.options.empty:
            return None
        return self.options.iloc[0].to_dict()

    @property
    def needs_action(self) -> bool:
        return self.tier != TIER_MONITOR or self.order_window_open


def fmt_time(ts: pd.Timestamp) -> str:
    return ts.strftime("%a %d %b, %H:%M")


def fmt_hours(hours: float) -> str:
    hours = max(0.0, float(hours))
    if hours < 48:
        return f"{hours:.0f} h" if hours >= 10 else f"{hours:.1f} h"
    return f"{hours / 24:.1f} days"


def plan_engine(
    ctx: PlanningContext,
    engine_id: str,
    cycle: int,
    now: pd.Timestamp,
    stock: dict[tuple[str, str], int],
    w_co2: float,
    next_departure: pd.Timestamp | None = None,
) -> Recommendation:
    fleet_row = ctx.fleet.loc[engine_id]
    pred = ctx.predictions.get(engine_id, cycle)
    cpd = float(fleet_row["cycles_per_day"])
    base = str(fleet_row["base_location"])
    rul_lower = float(pred["rul_lower"])
    tier = assign_tier(rul_lower, ctx.cfg)
    need_by = need_by_time(now, rul_lower, cpd, ctx.buffer)
    part_id = ctx.part_for_engine(engine_id)
    options = rank_options(ctx.build_options(part_id, base, stock), now, need_by, w_co2)

    escalated, shortfall, window_open = False, 0.0, False
    if not options.empty:
        best = options.iloc[0]
        if not bool(best["on_time"]):
            escalated = True
            shortfall = (best["arrival"] - need_by).total_seconds() / 3600.0
        window_open = tier == TIER_MONITOR and best["order_by"] <= now + ctx.window

    if tier == TIER_REVIEW:
        action = ACTION_REVIEW
    elif tier == TIER_PLAN:
        action = ACTION_PLAN
    elif window_open:
        action = ACTION_ORDER
    else:
        action = ACTION_MONITOR

    true_rul = pred.get("true_rul")
    rec = Recommendation(
        engine_id=engine_id, aircraft_id=str(fleet_row["aircraft_id"]), base=base, cycle=int(cycle), now=now,
        cycles_per_day=cpd, predicted_rul=float(pred["predicted_rul"]), rul_lower=rul_lower,
        rul_upper=float(pred["rul_upper"]), true_rul=float(true_rul) if pd.notna(true_rul) else float("nan"),
        top_sensors=str(pred["top_sensors"]).split(";"), tier=tier, need_by=need_by, part_id=part_id,
        part_name=str(ctx.parts.loc[part_id, "part_name"]), options=options, escalated=escalated,
        shortfall_hours=shortfall, order_window_open=bool(window_open), action=action, next_departure=next_departure,
    )
    rec.rationale = rationale(rec)
    return rec


def describe_option(opt: dict) -> str:
    if opt["kind"] == KIND_RESERVE:
        return f"reserve the spare already at {opt['source']}"
    if opt["kind"] == KIND_TRANSFER:
        return f"transfer from {opt['source']} by {opt['mode']}"
    return f"{opt['kind'].lower()} from {opt['source_name']} by {opt['mode']}"


def rationale(rec: Recommendation) -> str:
    parts = [
        f"Lower-bound RUL is {rec.rul_lower:.0f} cycles (estimate {rec.predicted_rul:.0f}, band "
        f"{rec.rul_lower:.0f}-{rec.rul_upper:.0f}) at {rec.cycles_per_day:g} flights a day, so the "
        f"{rec.part_name} is needed at {rec.base} by {fmt_time(rec.need_by)}."
    ]
    if rec.tier == TIER_REVIEW and rec.next_departure is not None:
        parts.append(f"Recommend authorised inspection before the next departure at {fmt_time(rec.next_departure)}.")
    opt = rec.chosen
    if opt is None:
        parts.append("No source for this part exists in the network or supplier list.")
    elif rec.escalated:
        parts.append(
            f"No option arrives in time. Fastest: {describe_option(opt)}, arriving {fmt_time(opt['arrival'])}, "
            f"{fmt_hours(rec.shortfall_hours)} after the need-by time. Escalate."
        )
    else:
        slack = (rec.need_by - opt["arrival"]).total_seconds() / 3600
        parts.append(
            f"Recommended: {describe_option(opt)}, arriving {fmt_time(opt['arrival'])} ({fmt_hours(slack)} to spare), "
            f"about ${opt['cost_usd']:,.0f} including part value and {opt['co2e_kg']:,.1f} kg CO2e."
        )
        if rec.order_window_open:
            parts.append(f"Order by {fmt_time(opt['order_by'])} to keep this option.")
    parts.append("Decision support only: authorised maintenance personnel decide.")
    return " ".join(parts)
