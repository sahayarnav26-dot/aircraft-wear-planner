"""Aircraft Wear Planner: post-landing wear review and spare-parts planning (decision support).

Run from the repo root:  streamlit run app/streamlit_app.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.data import load_config, load_tables  # noqa: E402
from core.fleet_sim import (  # noqa: E402
    advance_hours,
    complete_maintenance,
    init_state,
    land_next,
    open_aog_hours,
)
from core.optimizer import allocate, double_booked  # noqa: E402
from core.planning import (  # noqa: E402
    TIER_MONITOR,
    TIER_PLAN,
    TIER_RANK,
    TIER_REVIEW,
    PlanningContext,
    describe_option,
    free_stock,
    plan_engine,
    reserve_stock,
)
from core.policy_sim import compare  # noqa: E402
from ml.data_prep import SENSORS, load_raw, sensor_label, unit_from_engine_id  # noqa: E402

st.set_page_config(page_title="Aircraft Wear Planner", page_icon="✈️", layout="wide")

VIEWS = ["Fleet overview", "Engine detail", "Simulate landing", "Parts plan", "Impact"]
TIER_STYLE = {
    TIER_REVIEW: {"badge": "red", "fg": "#9f1c14", "bg": "#fde7e6", "short": "Review"},
    TIER_PLAN: {"badge": "orange", "fg": "#8a4b00", "bg": "#fff1d6", "short": "Plan"},
    TIER_MONITOR: {"badge": "green", "fg": "#1d6b3a", "bg": "#e5f4ea", "short": "Monitor"},
}
BLUE, GREY, RED, AMBER = "#2563eb", "#6b7280", "#d92d20", "#dc8a00"
BANNER = (
    "**Decision support only.** Engine data is a NASA C-MAPSS simulation (FD001); supply-chain data is "
    "synthetic. Thresholds are illustrative, not airworthiness limits. Maintenance and airworthiness "
    "decisions stay with authorised personnel."
)


# ---------------------------------------------------------------------------------------
# Cached data
# ---------------------------------------------------------------------------------------
@st.cache_resource
def get_context() -> PlanningContext:
    cfg = load_config(ROOT)
    return PlanningContext(load_tables(ROOT, tz=cfg["timezone"]), cfg)


@st.cache_data
def raw_train() -> pd.DataFrame:
    return load_raw("train_FD001.txt")


@st.cache_data
def model_metrics() -> dict:
    path = ROOT / "outputs" / "metrics.json"
    return json.loads(path.read_text()) if path.exists() else {}


@st.cache_data
def model_comparison() -> dict:
    path = ROOT / "outputs" / "model_comparison.json"
    return json.loads(path.read_text()) if path.exists() else {}


@st.cache_data
def policy_results(w_co2: float, _ctx: PlanningContext) -> tuple[pd.DataFrame, pd.DataFrame]:
    return compare(_ctx, _ctx.cfg, w_co2)


CTX = get_context()
CFG = CTX.cfg


# ---------------------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------------------
def reset_scenario() -> None:
    st.session_state.sim = init_state(CTX, CFG)
    st.session_state.inventory = CTX.tables.inventory.copy()
    st.session_state.decisions = {}
    st.session_state.decision_log = []
    st.session_state.recent_events = []
    st.session_state.flash = "Scenario reset to the start of the simulation."


if "sim" not in st.session_state:
    reset_scenario()
    st.session_state.flash = None
st.session_state.setdefault("view", VIEWS[0])
st.session_state.setdefault("w_co2", float(CFG["weights"]["co2e_default"]))


# ---------------------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------------------
def t_short(ts) -> str:
    return "—" if ts is None or pd.isna(ts) else ts.strftime("%a %d %b %H:%M")


def t_compact(ts) -> str:
    return "—" if ts is None or pd.isna(ts) else ts.strftime("%d %b %H:%M")


def until(ts, now) -> str:
    if ts is None or pd.isna(ts):
        return "—"
    mins = int(round((ts - now).total_seconds() / 60))
    sign_past = mins < 0
    mins = abs(mins)
    d, rem = divmod(mins, 1440)
    h, m = divmod(rem, 60)
    text = f"{d} d {h} h" if d else (f"{h} h {m:02d} min" if h else f"{m} min")
    return f"{text} ago" if sign_past else f"in {text}"


def money(x: float) -> str:
    return f"${x:,.0f}"


def engine_label(eid: str) -> str:
    sim = st.session_state.sim
    ac = sim.aircraft_of(eid)
    tag = " · replaced" if eid in sim.replaced else ""
    return f"{ac.aircraft_id} · {eid} ({ac.base}){tag}"


def tier_badge(tier: str) -> None:
    s = TIER_STYLE[tier]
    st.badge(tier, color=s["badge"])


def goto(view: str, engine: str | None = None) -> None:
    st.session_state.view = view
    if engine:
        st.session_state.focus_engine = engine


def engine_picker(label: str, engines: list[str], key: str) -> str:
    """Select an engine by its display label; remembers the focus engine across views."""
    labels = [engine_label(e) for e in engines]
    focus = st.session_state.get("focus_engine")
    idx = engines.index(focus) if focus in engines else 0
    choice = st.selectbox(label, labels, index=idx, key=key)
    eid = engines[labels.index(choice)] if choice in labels else engines[idx]
    st.session_state.focus_engine = eid
    return eid


# ---------------------------------------------------------------------------------------
# Planning for the current simulation moment
# ---------------------------------------------------------------------------------------
def current_plans() -> dict:
    sim = st.session_state.sim
    stock = free_stock(st.session_state.inventory)
    plans = {}
    for eid in sim.active_engines():
        ac = sim.aircraft_of(eid)
        plans[eid] = plan_engine(
            CTX, eid, sim.cycles[eid], sim.now, stock, st.session_state.w_co2,
            next_departure=ac.next_departure if ac.flying else None,
        )
    return plans


def part_status(eid: str, rec) -> str:
    sim = st.session_state.sim
    d = st.session_state.decisions.get(eid)
    if d:
        if sim.now >= d["arrival"]:
            return f"Part at {rec.base}, ready to fit"
        return f"Secured, arrives {t_short(d['arrival'])}"
    if rec.needs_action:
        return "Awaiting review"
    return "—"


# ---------------------------------------------------------------------------------------
# Actions (callbacks run before the rerun)
# ---------------------------------------------------------------------------------------
def _remember(events: list[dict]) -> None:
    st.session_state.recent_events = (list(reversed(events)) + st.session_state.recent_events)[:40]
    if not events:
        st.session_state.flash = "No aircraft is scheduled to land in that window."
        return
    last = events[-1]
    worst = last["worst_tier"]
    msg = f"{last['aircraft_id']} landed at {last['base']} {t_short(last['time'])}."
    if last["grounded"]:
        msg += " Aircraft grounded: an engine reached the end of its simulated life."
    elif last["tier_changed"]:
        msg += f" Tier changed: now {worst}."
    if len(events) > 1:
        msg += f" ({len(events)} landings processed.)"
    st.session_state.flash = msg


def act_land(aircraft_id: str | None = None) -> None:
    _remember(land_next(st.session_state.sim, CTX, CFG, aircraft_id))


def act_advance(hours: float) -> None:
    _remember(advance_hours(st.session_state.sim, CTX, CFG, hours))


def act_approve(eid: str, opt: dict, decision: str, note: str, via: str) -> None:
    sim = st.session_state.sim
    part = CTX.part_for_engine(eid)
    stock_loc = opt.get("stock_location")
    if isinstance(stock_loc, str):
        st.session_state.inventory = reserve_stock(st.session_state.inventory, stock_loc, part)
    record = {
        "engine_id": eid, "decided_at": sim.now, "decision": decision, "via": via,
        "option": describe_option(opt), "kind": opt["kind"], "source": opt["source"], "mode": opt["mode"],
        "stock_location": stock_loc if isinstance(stock_loc, str) else None,
        "arrival": sim.now + pd.Timedelta(hours=float(opt["lead_hours"])),
        "cost_usd": float(opt["cost_usd"]), "co2e_kg": float(opt["co2e_kg"]), "note": note.strip(),
        "part_id": part, "status": "Secured",
    }
    st.session_state.decisions[eid] = record
    st.session_state.decision_log.append(dict(record))
    st.session_state.flash = f"{decision}: {record['option']} for {eid}."


def act_cancel(eid: str) -> None:
    d = st.session_state.decisions.pop(eid, None)
    if d and d["stock_location"]:
        inv = st.session_state.inventory.copy()
        mask = (inv["location_id"] == d["stock_location"]) & (inv["part_id"] == d["part_id"])
        inv.loc[mask, "qty_reserved"] = (inv.loc[mask, "qty_reserved"] - 1).clip(lower=0)
        st.session_state.inventory = inv
    if d:
        st.session_state.decision_log.append({**d, "decided_at": st.session_state.sim.now, "decision": "Cancelled", "status": "Cancelled"})
        st.session_state.flash = f"Cancelled the plan for {eid}; any reserved spare is released."


def act_complete(eid: str) -> None:
    sim = st.session_state.sim
    d = st.session_state.decisions.pop(eid)
    if d["stock_location"]:
        inv = st.session_state.inventory.copy()
        mask = (inv["location_id"] == d["stock_location"]) & (inv["part_id"] == d["part_id"])
        inv.loc[mask, "qty_on_hand"] = (inv.loc[mask, "qty_on_hand"] - 1).clip(lower=0)
        inv.loc[mask, "qty_reserved"] = (inv.loc[mask, "qty_reserved"] - 1).clip(lower=0)
        st.session_state.inventory = inv
    complete_maintenance(sim, CTX, eid)
    st.session_state.decision_log.append({**d, "decided_at": sim.now, "decision": "Maintenance completed", "status": "Completed"})
    st.session_state.flash = f"Maintenance on {eid} recorded as completed by authorised personnel."


def act_approve_fleet(rows: list[dict]) -> None:
    plans = current_plans()
    for row in rows:
        rec = plans[row["engine_id"]]
        opts = rec.options
        match = opts[opts.apply(describe_option, axis=1) == row["fleet_choice"]]
        if not match.empty:
            act_approve(row["engine_id"], match.iloc[0].to_dict(), "Approved", "Approved from the fleet plan", "Fleet plan")
    st.session_state.flash = f"Approved the fleet plan for {len(rows)} engines."


# ---------------------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------------------
sim = st.session_state.sim
with st.sidebar:
    st.title("✈️ Aircraft Wear Planner")
    st.caption("Post-landing wear review and spare-parts planning")
    st.radio("View", VIEWS, key="view")
    st.divider()
    st.caption("Simulation clock (IST)")
    st.markdown(f"#### {sim.now:%a %d %b · %H:%M}")
    st.button("Next landing in the fleet", on_click=act_land, type="primary", width="stretch")
    c1, c2 = st.columns(2)
    c1.button("+6 hours", on_click=act_advance, args=(6,), width="stretch")
    c2.button("+1 day", on_click=act_advance, args=(24,), width="stretch")
    st.button("Reset scenario", on_click=reset_scenario, width="stretch")
    st.divider()
    st.slider(
        "Weight on emissions", 0.0, 1.0, step=0.1, key="w_co2",
        help="How on-time options are ranked: 0 = lowest cost only, 1 = lowest freight CO2e only.",
    )
    st.caption(
        "Data: [NASA C-MAPSS](https://data.nasa.gov/dataset/cmapss-jet-engine-simulated-data) FD001 · "
        "synthetic network, stock and suppliers · emission factors from the "
        "[NZ Measuring Emissions Catalogue 2026](https://measuringemissionsguide.environment.govt.nz/8_freight.html)."
    )

st.info(BANNER, icon="ℹ️")
if st.session_state.get("flash"):
    st.toast(st.session_state.flash)
    st.session_state.flash = None

PLANS = current_plans()
VIEW = st.session_state.view


# ---------------------------------------------------------------------------------------
# View 1: Fleet overview
# ---------------------------------------------------------------------------------------
def view_fleet() -> None:
    st.header("Fleet overview")
    recs = sorted(PLANS.values(), key=lambda r: (TIER_RANK[r.tier], r.rul_lower))
    grounded = [ac for ac in sim.aircraft.values() if not ac.flying]
    k = st.columns(5)
    k[0].metric("Engines tracked", len(recs), border=True)
    k[1].metric("Review before departure", sum(r.tier == TIER_REVIEW for r in recs), border=True)
    k[2].metric("Plan maintenance", sum(r.tier == TIER_PLAN for r in recs), border=True)
    k[3].metric("Order window open", sum(r.order_window_open for r in recs), border=True)
    k[4].metric(
        "Aircraft on ground", len(grounded), border=True,
        help=f"Grounded because an engine reached the end of its simulated life. Open AOG time: {open_aog_hours(sim):.0f} h; "
             f"closed AOG time this session: {sim.aog_closed_hours:.0f} h.",
    )

    rows = []
    for r in recs:
        ac = sim.aircraft_of(r.engine_id)
        tier = r.tier if not (r.tier == TIER_MONITOR and r.order_window_open) else f"{TIER_MONITOR} · order soon"
        rows.append({
            "Aircraft": r.aircraft_id, "Engine": r.engine_id, "Base": r.base, "Cycle": r.cycle,
            "RUL estimate": round(r.predicted_rul), "Lower edge": round(r.rul_lower), "Tier": tier,
            "Need part by": t_compact(r.need_by),
            "Next departure": t_compact(ac.next_departure) if ac.flying else "Grounded (AOG)",
            "Part status": part_status(r.engine_id, r),
        })
    df = pd.DataFrame(rows)

    def style_tier(v):
        s = TIER_STYLE.get(str(v).split(" · ")[0])
        return f"background-color: {s['bg']}; color: {s['fg']}; font-weight: 600" if s else ""

    st.subheader("Engines by urgency")
    st.dataframe(
        df.style.map(style_tier, subset=["Tier"]), hide_index=True,
        column_config={
            "Cycle": st.column_config.NumberColumn(width="small"),
            "RUL estimate": st.column_config.NumberColumn("RUL est.", help="Estimated remaining useful life, in cycles (flights)."),
            "Lower edge": st.column_config.NumberColumn("Lower edge", help="Cautious edge of the 80% band, in cycles. The planner acts on this."),
        },
    )
    ready = [e for e, d in st.session_state.decisions.items() if sim.now >= d["arrival"] and e in PLANS]
    if ready:
        c1, c2 = st.columns([3, 1])
        c1.success(f"**{len(ready)} part(s) have arrived and are ready to fit:** {', '.join(ready)}. Record the maintenance once it is done.")
        c2.button("Open the first one", on_click=goto, args=("Parts plan", ready[0]), width="stretch")
    urgent = [r for r in recs if r.needs_action and r.engine_id not in st.session_state.decisions]
    if urgent:
        r0 = urgent[0]
        c1, c2 = st.columns([3, 1])
        c1.warning(f"**{len(urgent)} engine(s) await review.** Most urgent: {r0.engine_id} on {r0.aircraft_id} at {r0.base} ({r0.tier}).")
        c2.button("Open its parts plan", on_click=goto, args=("Parts plan", r0.engine_id), width="stretch", type="primary")

    st.subheader("Spare HPC blade sets across the network")
    part = CTX.part_for_engine(recs[0].engine_id) if recs else "P-HPC-BLD"
    inv = st.session_state.inventory
    inv_p = inv[inv["part_id"] == part].assign(free=lambda d: (d["qty_on_hand"] - d["qty_reserved"]).clip(lower=0))
    loc = CTX.tables.locations.merge(inv_p, on="location_id")
    based = pd.Series([ac.base for ac in sim.aircraft.values()]).value_counts()
    loc["aircraft"] = loc["location_id"].map(based).fillna(0).astype(int)
    c1, c2 = st.columns([3, 2])
    with c1:
        loc = loc.assign(label=[f"{c} · {n} ({a} aircraft)" for c, n, a in zip(loc["location_id"], loc["name"], loc["aircraft"])])
        fig = go.Figure()
        fig.add_trace(go.Bar(y=loc["label"], x=loc["free"], orientation="h", name="Free", marker_color=BLUE,
                             text=loc["free"], textposition="inside", hovertemplate="%{y}: %{x} free<extra></extra>"))
        fig.add_trace(go.Bar(y=loc["label"], x=loc["qty_reserved"], orientation="h", name="Reserved", marker_color="#b8c2d0",
                             hovertemplate="%{y}: %{x} reserved<extra></extra>"))
        fig.update_layout(barmode="stack", height=320, margin=dict(l=10, r=10, t=10, b=10), xaxis_title="Spares on hand",
                          yaxis=dict(autorange="reversed"), legend=dict(orientation="h", y=1.1), xaxis=dict(dtick=1))
        st.plotly_chart(fig)
        st.caption("Stations with aircraft but no free spare depend on transfers or orders.")
    with c2:
        st.dataframe(
            loc[["location_id", "type", "qty_on_hand", "qty_reserved", "free"]].rename(columns={
                "location_id": "Code", "type": "Type", "qty_on_hand": "On hand", "qty_reserved": "Reserved", "free": "Free",
            }),
            hide_index=True,
        )


# ---------------------------------------------------------------------------------------
# View 2: Engine detail
# ---------------------------------------------------------------------------------------
def rul_figure(eid: str, upto: int, show_truth: bool) -> go.Figure:
    hist = CTX.predictions.history(eid, upto)
    x = hist.index
    p = CFG["planning"]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=hist["rul_upper"], line=dict(width=0), hoverinfo="skip", showlegend=False))
    fig.add_trace(go.Scatter(x=x, y=hist["rul_lower"], line=dict(width=0), fill="tonexty",
                             fillcolor="rgba(37,99,235,0.16)", name="80% band", hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=x, y=hist["predicted_rul"], line=dict(color=BLUE, width=2.2), name="Estimated RUL",
                             customdata=hist[["rul_lower", "rul_upper"]],
                             hovertemplate="Cycle %{x}<br>Estimate %{y:.0f} (band %{customdata[0]:.0f}–%{customdata[1]:.0f})<extra></extra>"))
    if show_truth and hist["true_rul"].notna().any():
        fig.add_trace(go.Scatter(x=x, y=hist["true_rul"], line=dict(color=GREY, width=1.5, dash="dot"),
                                 name="True RUL (simulation only)", hovertemplate="True RUL %{y:.0f}<extra></extra>"))
    fig.add_hline(y=p["review_max_rul_lower"], line=dict(color=RED, dash="dash", width=1),
                  annotation_text="Review threshold", annotation_position="top left")
    fig.add_hline(y=p["plan_max_rul_lower"], line=dict(color=AMBER, dash="dash", width=1),
                  annotation_text="Plan threshold", annotation_position="top left")
    ymax = max(150.0, float(hist["rul_upper"].max()) + 10)
    fig.update_layout(height=380, margin=dict(l=10, r=10, t=30, b=10), legend=dict(orientation="h", y=1.12),
                      xaxis_title="Cycle (one cycle = one flight)", yaxis_title="Remaining useful life (cycles)",
                      yaxis_range=[0, ymax], hovermode="x unified")
    return fig


def sensor_figure(eid: str, upto: int, sensors: list[str]) -> go.Figure:
    _, unit = unit_from_engine_id(eid)
    raw = raw_train()
    eng = raw[(raw["unit"] == unit) & (raw["cycle"] <= upto)]
    fig = make_subplots(rows=len(sensors), cols=1, shared_xaxes=True, vertical_spacing=0.08,
                        subplot_titles=[sensor_label(s) for s in sensors])
    for i, s in enumerate(sensors, start=1):
        fig.add_trace(go.Scatter(x=eng["cycle"], y=eng[s], line=dict(color=GREY, width=1), name=s, showlegend=False,
                                 hovertemplate=f"{s}: " + "%{y:.2f}<extra></extra>"), row=i, col=1)
        smooth = eng[s].rolling(10, min_periods=1).mean()
        fig.add_trace(go.Scatter(x=eng["cycle"], y=smooth, line=dict(color=BLUE, width=2), showlegend=False,
                                 hoverinfo="skip"), row=i, col=1)
    fig.update_layout(height=170 * len(sensors) + 40, margin=dict(l=10, r=10, t=40, b=10))
    fig.update_xaxes(title_text="Cycle", row=len(sensors), col=1)
    return fig


def view_engine() -> None:
    st.header("Engine detail")
    engines = sorted(PLANS, key=lambda e: (TIER_RANK[PLANS[e].tier], PLANS[e].rul_lower))
    if not engines:
        st.success("Every fleet engine has been replaced in this run. Reset the scenario to start again.")
        return
    eid = engine_picker("Engine", engines, key="pick_engine_detail")
    rec = PLANS[eid]
    ac = sim.aircraft_of(eid)

    c = st.columns(5)
    c[0].metric("Cycle", rec.cycle, help="Flights flown", border=True)
    c[1].metric("RUL estimate", f"{rec.predicted_rul:.0f}", help="Cycles", border=True)
    c[2].metric("80% band", f"{rec.rul_lower:.0f}–{rec.rul_upper:.0f}", border=True)
    c[3].metric(f"Need part by · {rec.need_by:%a %d %b}", rec.need_by.strftime("%H:%M"), border=True)
    if ac.flying:
        c[4].metric(f"Departs · {ac.next_departure:%a %d %b}", ac.next_departure.strftime("%H:%M"), border=True)
    else:
        c[4].metric("Next departure", "AOG", border=True)
    tier_badge(rec.tier)

    show_truth = st.toggle("Show true RUL (only known in simulation)", value=True)
    st.plotly_chart(rul_figure(eid, rec.cycle, show_truth))
    st.caption(
        "Labels are capped at 125 cycles during training, so early-life estimates flatten near the cap. "
        "The band narrows as wear progresses: errors near end of life are smaller."
    )

    st.subheader("What is driving the estimate")
    names = ", ".join(f"**{sensor_label(s)}**" for s in rec.top_sensors)
    explainer = model_metrics().get("explainer", "model attribution")
    st.write(f"Top sensors at cycle {rec.cycle}, by {explainer}: {names}.")
    sensor_labels = {sensor_label(s): s for s in SENSORS}
    picked_labels = st.multiselect(
        "Sensors to plot", list(sensor_labels), default=[sensor_label(s) for s in rec.top_sensors],
        max_selections=5, key=f"sensors_{eid}",
    )
    picked = [sensor_labels[x] for x in picked_labels]
    if picked:
        st.plotly_chart(sensor_figure(eid, rec.cycle, picked))
        st.caption("Grey: raw reading each flight. Blue: 10-flight average.")
    st.button("Open the parts plan for this engine", on_click=goto, args=("Parts plan", eid))


# ---------------------------------------------------------------------------------------
# View 3: Simulate landing
# ---------------------------------------------------------------------------------------
def review_card(event: dict) -> None:
    rec_by_engine = PLANS
    with st.container(border=True):
        st.markdown(f"**Post-landing review · {event['aircraft_id']} at {event['base']}** — landed {t_short(event['time'])}")
        if event["grounded"]:
            st.error("Aircraft grounded (AOG): an engine reached the end of its simulated life. Source a part and record the maintenance in the parts plan.")
        else:
            st.write(f"Next departure {t_short(event['next_departure'])} ({until(event['next_departure'], sim.now)}).")
        for e in event["engines"]:
            rec = rec_by_engine.get(e["engine_id"])
            arrow = "" if e["tier_before"] == e["tier_after"] else f" (was {e['tier_before']})"
            line = f"`{e['engine_id']}` · cycle {e['cycle']} · lower edge {e['rul_lower']:.0f} cycles · **{e['tier_after']}**{arrow}"
            if e["tier_after"] == TIER_REVIEW:
                st.error(line)
            elif e["tier_after"] == TIER_PLAN:
                st.warning(line)
            else:
                st.success(line)
            if rec is not None and e["tier_after"] != TIER_MONITOR and e["engine_id"] not in st.session_state.decisions:
                st.caption(rec.rationale)


def view_landing() -> None:
    st.header("Simulate landing")
    st.write("Each landing adds one flight cycle to both engines, refreshes the wear estimate and runs the planner before the next departure.")
    aircraft_ids = sorted(sim.aircraft)
    focus = st.session_state.get("focus_engine")
    default_ac = sim.aircraft_of(focus).aircraft_id if focus in PLANS else aircraft_ids[0]
    ac_labels = [f"{a} · based at {sim.aircraft[a].base}" for a in aircraft_ids]
    ac_choice = st.selectbox("Aircraft", ac_labels, index=aircraft_ids.index(default_ac), key="pick_aircraft")
    ac_id = aircraft_ids[ac_labels.index(ac_choice)]
    ac = sim.aircraft[ac_id]

    left, right = st.columns([1, 1.4])
    with left:
        with st.container(border=True):
            if ac.flying:
                st.markdown(f"**{ac_id}** · in service · {ac.cycles_per_day:g} flights a day")
                st.write(f"Last landing {t_short(ac.last_landing)} · next departure {t_short(ac.next_departure)} ({until(ac.next_departure, sim.now)})")
                st.write(f"Next landing {t_short(ac.next_landing)} ({until(ac.next_landing, sim.now)})")
            else:
                st.markdown(f"**{ac_id}** · **on ground (AOG)** since {t_short(ac.grounded_since)}")
                st.write(ac.grounded_reason)
            for eid in ac.engines:
                if eid in sim.replaced:
                    st.write(f"`{eid}` replaced {t_short(sim.replaced[eid])}")
                    continue
                r = PLANS[eid]
                st.write(f"`{eid}` · cycle {r.cycle} · estimate {r.predicted_rul:.0f} (band {r.rul_lower:.0f}–{r.rul_upper:.0f}) · {TIER_STYLE[r.tier]['short']}")
        st.button(f"Land {ac_id} (complete its next flight)", on_click=act_land, args=(ac_id,), type="primary",
                  disabled=not ac.flying, width="stretch")
        st.button("Next landing anywhere in the fleet", on_click=act_land, width="stretch")
        st.button("Jump ahead one day", on_click=act_advance, args=(24,), width="stretch")
    with right:
        mine = [e for e in st.session_state.recent_events if e["aircraft_id"] == ac_id]
        if mine:
            review_card(mine[0])
        else:
            st.info(f"No landing of {ac_id} yet in this session. Land it to run a post-landing review.")

    st.subheader("Landing log")
    log = st.session_state.recent_events
    if log:
        rows = []
        for e in log:
            for x in e["engines"]:
                rows.append({
                    "Landed": t_short(e["time"]), "Aircraft": e["aircraft_id"], "Base": e["base"], "Engine": x["engine_id"],
                    "Cycle": x["cycle"], "Lower edge": round(x["rul_lower"]), "Tier": x["tier_after"],
                    "Tier change": "" if x["tier_before"] == x["tier_after"] else f"{x['tier_before']} → {x['tier_after']}",
                    "Grounded": "Yes" if e["grounded"] else "",
                })
        st.dataframe(pd.DataFrame(rows), hide_index=True)
    else:
        st.caption("Landings you simulate appear here.")


# ---------------------------------------------------------------------------------------
# View 4: Parts plan
# ---------------------------------------------------------------------------------------
def options_table(rec) -> pd.DataFrame:
    o = rec.options
    return pd.DataFrame({
        "Rank": range(1, len(o) + 1),
        "Option": o.apply(describe_option, axis=1),
        "Arrives": [t_short(a) for a in o["arrival"]],
        "On time": ["Yes" if ok else "Late" for ok in o["on_time"]],
        "Order by": [t_short(a) if ok else "—" for a, ok in zip(o["order_by"], o["on_time"])],
        "Cost (USD)": [money(x) for x in o["cost_usd"]],
        "Freight CO2e (kg)": [f"{x:,.1f}" for x in o["co2e_kg"]],
        "Score": [f"{s:.3f}" if pd.notna(s) else "—" for s in o["score"]],
    })


def options_figure(rec) -> go.Figure:
    o = rec.options
    labels = o.apply(describe_option, axis=1)
    fig = go.Figure()
    for ok, name, color in ((True, "Arrives on time", BLUE), (False, "Arrives late", GREY)):
        sel = o["on_time"] == ok
        fig.add_trace(go.Scatter(
            x=o.loc[sel, "cost_usd"], y=o.loc[sel, "co2e_kg"], mode="markers", name=name,
            marker=dict(size=11, color=color, opacity=0.85), text=labels[sel],
            hovertemplate="%{text}<br>$%{x:,.0f} · %{y:.1f} kg CO2e<extra></extra>",
        ))
    best = o.iloc[0]
    fig.add_trace(go.Scatter(x=[best["cost_usd"]], y=[best["co2e_kg"]], mode="markers", name="Recommended",
                             marker=dict(size=20, color="rgba(0,0,0,0)", line=dict(color=RED if rec.escalated else "#111827", width=2)),
                             hoverinfo="skip"))
    fig.update_layout(height=320, margin=dict(l=10, r=10, t=10, b=10), xaxis_title="Cost incl. part value (USD)",
                      yaxis_title="Freight emissions (kg CO2e)", legend=dict(orientation="h", y=1.15))
    return fig


def view_parts() -> None:
    st.header("Parts plan")
    engines = sorted(PLANS, key=lambda e: (e in st.session_state.decisions, TIER_RANK[PLANS[e].tier], PLANS[e].rul_lower))
    if not engines:
        st.success("Every fleet engine has been replaced in this run.")
        return
    eid = engine_picker("Engine", engines, key="pick_engine_parts")
    rec = PLANS[eid]
    decision = st.session_state.decisions.get(eid)

    with st.container(border=True):
        tier_badge(rec.tier)
        st.markdown(f"**{rec.engine_id}** on **{rec.aircraft_id}** at **{rec.base}** · cycle {rec.cycle} · part: {rec.part_name}")
        c = st.columns(4)
        c[0].metric("Lower edge of RUL", f"{rec.rul_lower:.0f} cycles")
        c[1].metric("Need part by", rec.need_by.strftime("%d %b %H:%M"))
        c[2].metric("Action", "Escalate" if rec.escalated else rec.action.split(" and ")[0])
        c[3].metric("Next departure", rec.next_departure.strftime("%d %b %H:%M") if rec.next_departure is not None and pd.notna(rec.next_departure) else "Grounded")
        if decision is None:
            (st.error if rec.escalated else st.info)(rec.rationale)

    if decision is not None:
        st.subheader("Decision on record")
        arrived = sim.now >= decision["arrival"]
        if arrived:
            where = f"Part is at {rec.base}."
        else:
            where = f"Arrives {t_short(decision['arrival'])} ({until(decision['arrival'], sim.now)})."
        note_text = f" Note: {decision['note']}" if decision["note"] else ""
        st.success(f"{decision['decision']} {t_short(decision['decided_at'])}: {decision['option']}. {where}{note_text}")
        b1, b2 = st.columns(2)
        b1.button("Record maintenance completed", on_click=act_complete, args=(eid,), type="primary", disabled=not arrived,
                  help="Available once the part has arrived. Authorised maintenance personnel complete the work.", width="stretch")
        b2.button("Cancel this plan", on_click=act_cancel, args=(eid,), width="stretch")
    elif not rec.options.empty:
        st.subheader("Options")
        st.dataframe(options_table(rec), hide_index=True, column_config={
            "Cost (USD)": st.column_config.TextColumn(help="Part value plus logistics. Using a spare from stock is costed at its replacement value."),
            "Score": st.column_config.TextColumn(help="Lower is better. Only on-time options are scored."),
        })
        st.plotly_chart(options_figure(rec))

        st.subheader("Approve or override")
        labels = [
            f"{i + 1}. {d}" + (" (recommended)" if i == 0 else "")
            for i, d in enumerate(rec.options.apply(describe_option, axis=1))
        ]
        pick_label = st.radio("Option to secure", labels, key=f"pick_option_{eid}")
        pick = labels.index(pick_label) if pick_label in labels else 0
        note = st.text_input("Note (required for an override)", key=f"note_{eid}")
        chosen = rec.options.iloc[pick].to_dict()
        if pick == 0:
            st.button("Approve recommendation", on_click=act_approve, args=(eid, chosen, "Approved", note, "Per-engine"), type="primary")
        else:
            st.button("Approve override", on_click=act_approve, args=(eid, chosen, "Override", note, "Per-engine"), type="primary",
                      disabled=not note.strip(), help="Add a note explaining the override.")

    st.divider()
    st.subheader("Fleet plan: share scarce spares across engines")
    pending = [r for r in PLANS.values() if r.needs_action and r.engine_id not in st.session_state.decisions]
    if not pending:
        st.caption("No engine is waiting for a parts decision right now.")
    else:
        stock = free_stock(st.session_state.inventory)
        clash = double_booked(pending, stock)
        alloc = allocate(pending, stock, st.session_state.w_co2)
        if clash:
            st.warning(f"Planning each engine alone would claim **{clash} more spare(s) than exist**. The optimizer (PuLP with the HiGHS solver) allocates them jointly.")
        else:
            st.caption("Per-engine picks do not clash on stock right now; the optimizer confirms the allocation.")
        show = alloc.assign(
            **{"Need part by": alloc["need_by"].map(t_short), "Arrives": alloc["arrival"].map(t_short),
               "Changed by optimizer": alloc["changed"].map({True: "Yes", False: ""})}
        )[["engine_id", "aircraft_id", "base", "tier", "fleet_choice", "Arrives", "Need part by", "on_time", "per_engine_choice", "Changed by optimizer"]]
        st.dataframe(show.rename(columns={
            "engine_id": "Engine", "aircraft_id": "Aircraft", "base": "Base", "tier": "Tier", "fleet_choice": "Fleet plan",
            "on_time": "On time", "per_engine_choice": "Per-engine pick",
        }), hide_index=True)
        st.button(f"Approve the fleet plan for {len(alloc)} engines", on_click=act_approve_fleet,
                  args=(alloc.to_dict("records"),), type="primary")

    if st.session_state.decision_log:
        st.subheader("Decision log")
        log = pd.DataFrame(st.session_state.decision_log)
        log = log.assign(When=log["decided_at"].map(t_short), Arrives=log["arrival"].map(t_short))
        st.dataframe(log[["When", "engine_id", "decision", "option", "Arrives", "via", "note"]].rename(columns={
            "engine_id": "Engine", "decision": "Decision", "option": "Option", "via": "Via", "note": "Note",
        }).iloc[::-1], hide_index=True)


# ---------------------------------------------------------------------------------------
# View 5: Impact
# ---------------------------------------------------------------------------------------
def view_impact() -> None:
    st.header("Impact")
    st.write(
        "Both policies start from the same fleet, stock and suppliers and replay all 20 fleet engines to the end of "
        "their simulated life. **Reactive** sources a part only when an engine fails. **Predictive** runs this planner "
        "after every landing. This is a synthetic scenario: it shows the mechanism, not real-world savings."
    )
    rows, summary = policy_results(float(st.session_state.w_co2), CTX)
    r, p = summary["Reactive"], summary["Predictive"]

    k = st.columns(5)
    k[0].metric("Unplanned removals", int(p["Unscheduled removals"]), delta=int(p["Unscheduled removals"] - r["Unscheduled removals"]),
                delta_color="inverse", border=True)
    k[1].metric("AOG hours", f"{p['AOG hours waiting for parts']:,.0f}",
                delta=f"{p['AOG hours waiting for parts'] - r['AOG hours waiting for parts']:,.0f}", delta_color="inverse", border=True)
    k[2].metric("Air shipments", int(p["Air shipments"]), delta=int(p["Air shipments"] - r["Air shipments"]),
                delta_color="inverse", border=True)
    k[3].metric("Freight kg CO2e", f"{p['Freight emissions (kg CO2e)']:,.0f}",
                delta=f"{p['Freight emissions (kg CO2e)'] - r['Freight emissions (kg CO2e)']:,.0f}", delta_color="inverse", border=True)
    k[4].metric("Parts cost", f"${p['Parts cost incl. logistics (USD)'] / 1e6:,.2f}M",
                delta=f"{(p['Parts cost incl. logistics (USD)'] - r['Parts cost incl. logistics (USD)']) / 1e3:,.0f}k",
                delta_color="inverse", border=True)
    st.caption(
        "Predictive values, with the change against reactive underneath (green is better). AOG hours = hours aircraft "
        "wait on the ground for a part. Parts cost includes part value and logistics."
    )

    metrics = ["Unscheduled removals", "AOG hours waiting for parts", "Air shipments", "Freight emissions (kg CO2e)"]
    fig = make_subplots(rows=1, cols=4, subplot_titles=metrics, horizontal_spacing=0.07)
    for i, m in enumerate(metrics, start=1):
        fig.add_trace(go.Bar(x=["Reactive", "Predictive"], y=[r[m], p[m]], marker_color=[GREY, BLUE],
                             text=[f"{r[m]:,.0f}", f"{p[m]:,.0f}"], textposition="outside", showlegend=False,
                             hovertemplate="%{x}: %{y:,.1f}<extra></extra>"), row=1, col=i)
        fig.update_yaxes(range=[0, max(r[m], p[m]) * 1.25 + 1], showticklabels=False, row=1, col=i)
    fig.update_layout(height=320, margin=dict(l=10, r=10, t=40, b=10))
    st.plotly_chart(fig)

    st.warning(
        f"**The trade-off:** removing engines before they fail leaves life unused, on average "
        f"**{p['Mean unused life (cycles)']:.0f} cycles per engine** in this scenario. Reactive uses every cycle but "
        f"grounds each aircraft until a part arrives."
    )
    with st.expander("Per-engine results"):
        show = rows.assign(**{c: rows[c].map(t_short) for c in ("trigger_time", "removal_time")})
        st.dataframe(show[["policy", "engine_id", "base", "trigger_time", "action", "removal_time", "unscheduled_removal",
                           "aog_hours", "cost_usd", "co2e_kg", "unused_cycles"]], hide_index=True)

    st.subheader("Model quality")
    m = model_metrics()
    if m:
        t, v = m["test_last_cycle"], m["replay_engines_all_cycles"]
        c = st.columns(4)
        c[0].metric("Test RMSE (cycles)", f"{t['rmse_vs_capped_true']:.1f}", help="100 official FD001 test engines, last cycle, true RUL capped at 125.", border=True)
        c[1].metric("NASA score (test)", f"{t['nasa_score_vs_capped_true']:,.0f}", help="PHM08 scoring function; penalises late predictions more. Lower is better.", border=True)
        c[2].metric("Band coverage (replay)", f"{v['band_coverage_vs_capped_true']:.0%}", help="Target 80%. 20 held-out engines, every cycle.", border=True)
        c[3].metric("Mean band width", f"{v['mean_band_width_cycles']:.0f} cycles", border=True)
        st.caption(f"In use: {m.get('model_name', m['model_version'])}. {m['band']}. Simulated engine data; not validated on airline data.")

    comp = model_comparison()
    if comp:
        st.markdown("**Models compared on the same engines**")
        table = pd.DataFrame([{
            "Model": r["label"] + (" · in use" if r["primary"] else ""),
            "Test RMSE": f"{r['test_rmse']:.1f} cycles",
            "NASA score": f"{r['test_nasa_score']:,.0f}",
            "Band coverage": f"{r['replay_band_coverage']:.0%}",
            "Band width": f"{r['replay_band_width']:.0f} cycles",
            "Plan alert lead": f"{r['plan_alert_lead_median']:.0f} flights",
            "Review alert lead": f"{r['review_alert_lead_median']:.0f} flights",
        } for r in comp["models"]])
        st.dataframe(table, hide_index=True)
        st.caption(
            "Alert leads are medians over the 20 held-out engines: flights left when the lower edge first reaches "
            "the Plan and Review thresholds. Selection rule, fixed before comparing: " + comp["rule"]
        )

    st.subheader("Assumptions")
    st.markdown(
        "- One C-MAPSS cycle is treated as one flight; each aircraft flies a fixed daily rotation from its base.\n"
        "- Using a spare from stock is costed at its replacement value, so options compare on equal terms.\n"
        "- Freight emissions = part weight × distance × mode factor (kg CO2e per tonne-km); air factors include radiative forcing. "
        "Source: [NZ Ministry for the Environment, Measuring Emissions Catalogue 2026](https://measuringemissionsguide.environment.govt.nz/8_freight.html).\n"
        "- Thresholds, network, stock, prices and lead times are synthetic and illustrative."
    )


{"Fleet overview": view_fleet, "Engine detail": view_engine, "Simulate landing": view_landing,
 "Parts plan": view_parts, "Impact": view_impact}[VIEW]()
