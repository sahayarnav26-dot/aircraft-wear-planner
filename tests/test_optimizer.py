import pandas as pd

from core.optimizer import allocate, double_booked, sequential_by_urgency
from core.planning import KIND_RESERVE, KIND_STANDARD, TIER_PLAN, Recommendation, rank_options


def make_rec(engine_id, now, need_by, options):
    return Recommendation(
        engine_id=engine_id, aircraft_id="AC-T", base="DEL", cycle=100, now=now, cycles_per_day=4,
        predicted_rul=30, rul_lower=20, rul_upper=40, true_rul=float("nan"), top_sensors=["s11"], tier=TIER_PLAN,
        need_by=need_by, part_id="P-HPC-BLD", part_name="HPC blade set",
        options=rank_options(options, now, need_by, 0.5), escalated=False, shortfall_hours=0.0,
        order_window_open=False, action="Plan",
    )


def spare_and_order():
    return [
        dict(kind=KIND_RESERVE, source="DEL", source_name="DEL", mode="none", lead_hours=0.0, cost_usd=240_000.0,
             co2e_kg=0.0, distance_km=0.0, stock_location="DEL"),
        dict(kind=KIND_STANDARD, source="SUP", source_name="SUP", mode="road", lead_hours=72.0, cost_usd=249_600.0,
             co2e_kg=11.0, distance_km=1750.0, stock_location=None),
    ]


def test_one_spare_goes_to_the_engine_that_cannot_wait():
    now = pd.Timestamp("2026-10-11 06:00", tz="Asia/Kolkata")
    urgent = make_rec("A", now, now + pd.Timedelta(hours=10), spare_and_order())  # only the spare is on time
    patient = make_rec("B", now, now + pd.Timedelta(days=30), spare_and_order())  # both options on time
    stock = {("DEL", "P-HPC-BLD"): 1}

    # Planned one at a time, both engines want the single spare.
    assert double_booked([urgent, patient], stock) == 1

    alloc = allocate([patient, urgent], stock, 0.5).set_index("engine_id")
    assert alloc.loc["A", "kind"] == KIND_RESERVE
    assert alloc.loc["B", "kind"] == KIND_STANDARD
    assert alloc["on_time"].all()
    assert bool(alloc.loc["B", "changed"])

    seq = sequential_by_urgency([patient, urgent], stock, 0.5).set_index("engine_id")
    assert seq["on_time"].all()


def test_allocation_never_exceeds_stock(ctx, tables, now):
    from core.planning import free_stock, plan_engine

    stock = free_stock(tables.inventory)
    recs = [plan_engine(ctx, r.engine_id, r.current_cycle, now, stock, 0.5) for r in tables.fleet.itertuples()]
    pending = [r for r in recs if r.needs_action]
    alloc = allocate(pending, stock, 0.5)
    assert len(alloc) == len(pending)
    used = alloc.dropna(subset=["stock_location"]).groupby("stock_location").size()
    for loc, n in used.items():
        assert n <= stock[(loc, "P-HPC-BLD")]
