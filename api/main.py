"""Optional REST layer over the same core functions the dashboard uses.

Run from the repo root:  uvicorn api.main:app --reload
Interactive docs:        http://127.0.0.1:8000/docs

The simulation state lives in memory (single process, demo only). Decision support only:
nothing here reserves stock or places orders.
"""
from __future__ import annotations

import math

import pandas as pd
from fastapi import FastAPI, HTTPException, Query

from core.data import load_config, load_tables
from core.fleet_sim import init_state, land_next
from core.optimizer import allocate
from core.planning import PlanningContext, describe_option, free_stock, plan_engine

CFG = load_config()
CTX = PlanningContext(load_tables(tz=CFG["timezone"]), CFG)
STATE = {"sim": init_state(CTX, CFG)}

app = FastAPI(
    title="Aircraft Wear Planner API",
    version="1.0.0",
    description=(
        "Post-landing wear estimates and spare-parts recommendations. Decision support only: "
        "NASA C-MAPSS simulated engine data and synthetic supply-chain data."
    ),
)


def _clean(v):
    if isinstance(v, pd.Timestamp):
        return None if pd.isna(v) else v.isoformat()
    if isinstance(v, float) and math.isnan(v):
        return None
    return v


def _option_json(opt: dict) -> dict:
    keys = ("kind", "source", "source_name", "mode", "lead_hours", "arrival", "on_time", "order_by", "cost_usd", "co2e_kg", "score")
    out = {k: _clean(opt.get(k)) for k in keys}
    out["description"] = describe_option(opt)
    out["on_time"] = bool(opt.get("on_time"))
    return out


def _plan(engine_id: str, weight: float):
    sim = STATE["sim"]
    if engine_id not in sim.cycles:
        raise HTTPException(status_code=404, detail=f"Unknown fleet engine {engine_id}")
    if engine_id in sim.replaced:
        raise HTTPException(status_code=409, detail=f"{engine_id} has been replaced in this simulation")
    ac = sim.aircraft_of(engine_id)
    return plan_engine(CTX, engine_id, sim.cycles[engine_id], sim.now, free_stock(CTX.tables.inventory), weight,
                       next_departure=ac.next_departure if ac.flying else None)


def _plan_json(rec) -> dict:
    return {
        "engine_id": rec.engine_id, "aircraft_id": rec.aircraft_id, "base": rec.base, "cycle": rec.cycle,
        "as_of": _clean(rec.now), "predicted_rul": rec.predicted_rul, "rul_lower": rec.rul_lower, "rul_upper": rec.rul_upper,
        "tier": rec.tier, "action": rec.action, "need_by": _clean(rec.need_by), "part_id": rec.part_id,
        "escalated": rec.escalated, "order_window_open": rec.order_window_open, "rationale": rec.rationale,
        "recommendation": _option_json(rec.chosen) if rec.chosen else None,
        "options": [_option_json(o) for o in rec.options.to_dict("records")],
    }


@app.get("/engines")
def list_engines(weight: float = Query(0.5, ge=0, le=1)):
    sim = STATE["sim"]
    out = []
    for eid in sim.active_engines():
        rec = _plan(eid, weight)
        out.append({
            "engine_id": eid, "aircraft_id": rec.aircraft_id, "base": rec.base, "cycle": rec.cycle,
            "predicted_rul": rec.predicted_rul, "rul_lower": rec.rul_lower, "rul_upper": rec.rul_upper, "tier": rec.tier,
        })
    return {"as_of": _clean(sim.now), "engines": out}


@app.get("/engines/{engine_id}/health")
def engine_health(engine_id: str, cycle: int | None = Query(None, ge=1)):
    sim = STATE["sim"]
    if engine_id not in sim.cycles:
        raise HTTPException(status_code=404, detail=f"Unknown fleet engine {engine_id}")
    upto = sim.cycles[engine_id] if cycle is None else cycle
    if upto > sim.cycles[engine_id]:
        raise HTTPException(status_code=400, detail="Cannot look past the current cycle")
    try:
        now = CTX.predictions.get(engine_id, upto)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    hist = CTX.predictions.history(engine_id, upto).reset_index()
    return {
        "engine_id": engine_id, "cycle": upto, "predicted_rul": now["predicted_rul"], "rul_lower": now["rul_lower"],
        "rul_upper": now["rul_upper"], "true_rul_simulation_only": _clean(now["true_rul"]),
        "top_sensors": str(now["top_sensors"]).split(";"),
        "history": [{k: _clean(v) for k, v in row.items()} for row in hist[["cycle", "predicted_rul", "rul_lower", "rul_upper"]].to_dict("records")],
    }


@app.post("/engines/{engine_id}/landing")
def engine_landing(engine_id: str, weight: float = Query(0.5, ge=0, le=1)):
    """Advance the simulation to the next landing of this engine's aircraft and return the post-landing review."""
    sim = STATE["sim"]
    if engine_id not in sim.cycles:
        raise HTTPException(status_code=404, detail=f"Unknown fleet engine {engine_id}")
    ac = sim.aircraft_of(engine_id)
    if not ac.flying:
        raise HTTPException(status_code=409, detail=f"{ac.aircraft_id} is on ground: {ac.grounded_reason}")
    events = land_next(sim, CTX, CFG, ac.aircraft_id)
    last = events[-1]
    alerts = [e for e in last["engines"] if e["tier_before"] != e["tier_after"] or e["end_of_life"]]
    return {
        "landed": _clean(last["time"]), "aircraft_id": last["aircraft_id"], "base": last["base"],
        "next_departure": _clean(last["next_departure"]), "grounded": last["grounded"],
        "alerts": alerts, "landings_processed": len(events),
        "plan": None if engine_id in sim.replaced else _plan_json(_plan(engine_id, weight)),
    }


@app.get("/engines/{engine_id}/plan")
def engine_plan(engine_id: str, weight: float = Query(0.5, ge=0, le=1)):
    return _plan_json(_plan(engine_id, weight))


@app.get("/fleet/plan")
def fleet_plan(weight: float = Query(0.5, ge=0, le=1)):
    sim = STATE["sim"]
    recs = [_plan(e, weight) for e in sim.active_engines()]
    pending = [r for r in recs if r.needs_action]
    alloc = allocate(pending, free_stock(CTX.tables.inventory), weight)
    rows = [{k: _clean(v) for k, v in row.items()} for row in alloc.to_dict("records")] if not alloc.empty else []
    return {"as_of": _clean(sim.now), "engines_needing_action": len(pending), "allocation": rows}


@app.post("/reset")
def reset():
    STATE["sim"] = init_state(CTX, CFG)
    return {"as_of": _clean(STATE["sim"].now), "status": "reset"}
