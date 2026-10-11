"""Fleet-level allocation of scarce spares with PuLP and the HiGHS solver.

Per-engine ranking looks at one engine at a time, so two engines can both pick the same
spare. The optimizer chooses one option per engine jointly:

    minimise   sum over e, o of  score[e, o] * x[e, o]
    subject to sum over o of x[e, o] = 1                   for every engine e
               sum of x[e, o] drawing on location l <= s_l  for every location l
               x[e, o] in {0, 1}

score = (1 - w) * cost + w * CO2e, both normalised across all candidate options, plus a large
penalty (and the lateness in days) for options that arrive after the engine's need-by time,
so the model always has a solution and prefers being on time above everything else.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pulp

from core.planning import Recommendation, describe_option, rank_options

LATE_PENALTY = 1000.0


def _candidates(recs: list[Recommendation]) -> pd.DataFrame:
    rows = []
    for i, rec in enumerate(recs):
        for j, opt in rec.options.reset_index(drop=True).iterrows():
            lateness_h = max(0.0, (opt["arrival"] - rec.need_by).total_seconds() / 3600)
            rows.append({**opt.to_dict(), "i": i, "j": j, "engine_id": rec.engine_id, "part_id": rec.part_id, "lateness_h": lateness_h})
    return pd.DataFrame(rows)


def _scores(cand: pd.DataFrame, w_co2: float) -> np.ndarray:
    def norm(v: np.ndarray) -> np.ndarray:
        lo, hi = float(v.min()), float(v.max())
        return np.zeros_like(v) if hi - lo < 1e-9 else (v - lo) / (hi - lo)

    base = (1 - w_co2) * norm(cand["cost_usd"].to_numpy(float)) + w_co2 * norm(cand["co2e_kg"].to_numpy(float))
    late = ~cand["on_time"].to_numpy(bool)
    return base + late * (LATE_PENALTY + cand["lateness_h"].to_numpy(float) / 24.0)


def allocate(recs: list[Recommendation], stock: dict[tuple[str, str], int], w_co2: float) -> pd.DataFrame:
    """Return one chosen option per engine, allocated jointly. Empty frame if no engines."""
    recs = [r for r in recs if not r.options.empty]
    if not recs:
        return pd.DataFrame()
    cand = _candidates(recs)
    cand["obj"] = _scores(cand, w_co2)

    prob = pulp.LpProblem("fleet_spares", pulp.LpMinimize)
    # PuLP 4 creates variables through the problem rather than LpVariable(...).
    x = {k: prob.add_variable(f"x_{k}", lowBound=0, upBound=1, cat="Binary") for k in cand.index}
    prob += pulp.lpSum(cand.loc[k, "obj"] * x[k] for k in cand.index)
    for i, grp in cand.groupby("i"):
        prob += pulp.lpSum(x[k] for k in grp.index) == 1, f"one_option_{i}"
    stocked = cand[cand["stock_location"].notna()]
    for (loc, part), grp in stocked.groupby(["stock_location", "part_id"]):
        prob += pulp.lpSum(x[k] for k in grp.index) <= int(stock.get((loc, part), 0)), f"stock_{loc}_{part}"

    result = prob.solve(pulp.getSolver("HiGHS", msg=False))
    if result.status != pulp.LpSolveStatus.Optimal:
        raise RuntimeError(f"Fleet allocation failed: {result.status_str}")

    chosen = [k for k in cand.index if (x[k].value() or 0) > 0.5]
    picks = cand.loc[chosen].set_index("i")
    rows = []
    for i, rec in enumerate(recs):
        opt = picks.loc[i]
        solo = rec.options.iloc[0]
        rows.append({
            "engine_id": rec.engine_id, "aircraft_id": rec.aircraft_id, "base": rec.base, "tier": rec.tier,
            "need_by": rec.need_by, "fleet_choice": describe_option(opt), "kind": opt["kind"], "source": opt["source"],
            "mode": opt["mode"], "arrival": opt["arrival"], "on_time": bool(opt["on_time"]),
            "cost_usd": float(opt["cost_usd"]), "co2e_kg": float(opt["co2e_kg"]),
            "stock_location": opt["stock_location"], "per_engine_choice": describe_option(solo),
            "changed": describe_option(opt) != describe_option(solo),
        })
    return pd.DataFrame(rows)


def double_booked(recs: list[Recommendation], stock: dict[tuple[str, str], int]) -> int:
    """Units of stock that independent per-engine picks would claim beyond what exists."""
    claims: dict[tuple[str, str], int] = {}
    for rec in recs:
        if rec.options.empty:
            continue
        loc = rec.options.iloc[0]["stock_location"]
        if isinstance(loc, str):
            claims[(loc, rec.part_id)] = claims.get((loc, rec.part_id), 0) + 1
    return sum(max(0, n - int(stock.get(k, 0))) for k, n in claims.items())


def sequential_by_urgency(recs: list[Recommendation], stock: dict[tuple[str, str], int], w_co2: float) -> pd.DataFrame:
    """Baseline: most urgent engine first, each takes its best option from what is left."""
    left = dict(stock)
    rows = []
    for rec in sorted(recs, key=lambda r: r.need_by):
        if rec.options.empty:
            continue
        base_cols = ["kind", "source", "source_name", "mode", "lead_hours", "cost_usd", "co2e_kg", "distance_km", "stock_location"]
        avail = [
            o for o in rec.options[base_cols].to_dict("records")
            if not isinstance(o["stock_location"], str) or left.get((o["stock_location"], rec.part_id), 0) > 0
        ]
        ranked = rank_options(avail, rec.now, rec.need_by, w_co2)
        opt = ranked.iloc[0]
        if isinstance(opt["stock_location"], str):
            left[(opt["stock_location"], rec.part_id)] -= 1
        rows.append({
            "engine_id": rec.engine_id, "choice": describe_option(opt), "on_time": bool(opt["on_time"]),
            "cost_usd": float(opt["cost_usd"]), "co2e_kg": float(opt["co2e_kg"]),
        })
    return pd.DataFrame(rows)
