import pandas as pd

from core.fleet_sim import advance_hours, complete_maintenance, init_state, land_next
from core.policy_sim import EngineClock, compare


def test_init_state_matches_fleet_table(ctx, cfg, tables):
    state = init_state(ctx, cfg)
    assert state.now == pd.Timestamp(cfg["simulation"]["start"]).tz_convert(cfg["timezone"])
    for row in tables.fleet.itertuples():
        assert state.cycles[row.engine_id] == row.current_cycle


def test_landing_adds_one_cycle_to_that_aircraft_only(ctx, cfg):
    state = init_state(ctx, cfg)
    before = dict(state.cycles)
    ac = state.aircraft["AC-05"]
    expected_time = ac.next_landing
    events = land_next(state, ctx, cfg, "AC-05")
    assert events[-1]["aircraft_id"] == "AC-05" and events[-1]["time"] == expected_time
    for eid in ac.engines:
        assert state.cycles[eid] == before[eid] + 1
    landed_others = {e["aircraft_id"] for e in events[:-1]}
    for other in state.aircraft.values():
        if other.aircraft_id not in landed_others | {"AC-05"}:
            for eid in other.engines:
                assert state.cycles[eid] == before[eid]


def test_end_of_life_grounds_aircraft_until_maintenance(ctx, cfg):
    state = init_state(ctx, cfg)
    advance_hours(state, ctx, cfg, 24 * 4)  # long enough for the most worn engine to reach end of life
    grounded = [ac for ac in state.aircraft.values() if not ac.flying]
    assert grounded, "expected at least one aircraft on ground"
    ac = grounded[0]
    failed = [e for e in ac.engines if state.cycles[e] >= ctx.predictions.last_cycle(e)]
    for eid in failed:
        complete_maintenance(state, ctx, eid)
    assert ac.flying and pd.notna(ac.next_departure)
    assert state.aog_closed_hours > 0


def test_engine_clock_round_trip(now):
    clock = EngineClock(now, now + pd.Timedelta(hours=1), 4, pd.Timedelta(hours=1), start_cycle=100)
    assert clock.time_of(100) == now
    for c in (101, 105, 130):
        assert clock.cycle_at(clock.time_of(c)) == c


def test_predictive_beats_reactive_on_disruption(ctx, cfg):
    rows, summary = compare(ctx, cfg, 0.5)
    r, p = summary["Reactive"], summary["Predictive"]
    assert (rows["policy"] == "Reactive").sum() == 20 and (rows["policy"] == "Predictive").sum() == 20
    assert p["Unscheduled removals"] < r["Unscheduled removals"]
    assert p["AOG hours waiting for parts"] < r["AOG hours waiting for parts"]
    assert p["Freight emissions (kg CO2e)"] < r["Freight emissions (kg CO2e)"]
    assert p["Mean unused life (cycles)"] > 0  # the honest trade-off
