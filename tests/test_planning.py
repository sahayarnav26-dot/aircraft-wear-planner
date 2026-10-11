import pandas as pd
import pytest

from core.planning import (
    KIND_EXPEDITED,
    KIND_RESERVE,
    KIND_STANDARD,
    TIER_MONITOR,
    TIER_PLAN,
    TIER_REVIEW,
    EmissionFactors,
    assign_tier,
    best_option,
    free_stock,
    need_by_time,
    plan_engine,
    rank_options,
    reserve_stock,
)


def opt(kind, lead_hours, cost, co2, stock_location=None, source="X", mode="road"):
    return dict(kind=kind, source=source, source_name=source, mode=mode, lead_hours=lead_hours, cost_usd=cost,
                co2e_kg=co2, distance_km=100.0, stock_location=stock_location)


@pytest.mark.parametrize("lower,tier", [
    (0.0, TIER_REVIEW), (10.0, TIER_REVIEW), (10.01, TIER_PLAN), (40.0, TIER_PLAN), (40.01, TIER_MONITOR), (120.0, TIER_MONITOR),
])
def test_tier_boundaries(cfg, lower, tier):
    assert assign_tier(lower, cfg) == tier


def test_need_by_uses_lower_edge_buffer_and_utilisation(now):
    # (25 - 5) cycles at 4 per day = 5 days
    assert need_by_time(now, 25, 4, 5) == now + pd.Timedelta(days=5)


def test_need_by_never_before_landing(now):
    assert need_by_time(now, 3, 6, 5) == now


def test_emission_factor_picks_haul_by_distance(tables):
    ef = EmissionFactors(tables.emission_factors, 3700)
    assert ef.key("air", 1200) == "air_short_haul"
    assert ef.key("air", 4150) == "air_long_haul"
    assert ef.key("road", 4150) == "road"
    # 60 kg over 1,000 km by road at 0.105 kg CO2e per tonne-km
    assert ef.co2e_kg(60, 1000, "road") == pytest.approx(6.3)
    assert ef.co2e_kg(60, 0, "none") == 0.0


def test_rank_scores_only_on_time_options(now):
    options = [
        opt(KIND_STANDARD, 24 * 7, 230_000, 4.0),
        opt(KIND_EXPEDITED, 24 * 2, 260_000, 120.0, mode="air"),
        opt(KIND_RESERVE, 0, 240_000, 0.0, stock_location="DEL"),
    ]
    ranked = rank_options(options, now, now + pd.Timedelta(days=3), w_co2=0.5)
    assert ranked["recommended"].sum() == 1
    late = ranked[ranked["kind"] == KIND_STANDARD].iloc[0]
    assert not late["on_time"] and pd.isna(late["score"])
    assert ranked.iloc[0]["kind"] == KIND_RESERVE  # on time, cheapest-ish and zero freight
    assert ranked.iloc[0]["on_time"]


def test_weight_changes_choice_when_cost_and_emissions_conflict(now):
    options = [opt(KIND_STANDARD, 24, 100.0, 50.0), opt(KIND_STANDARD, 24, 200.0, 10.0, source="Y")]
    need_by = now + pd.Timedelta(days=5)
    assert rank_options(options, now, need_by, 0.0).iloc[0]["source"] == "X"
    assert rank_options(options, now, need_by, 1.0).iloc[0]["source"] == "Y"
    assert best_option(options, now, need_by, 0.0)[0]["source"] == "X"
    assert best_option(options, now, need_by, 1.0)[0]["source"] == "Y"


def test_escalates_to_fastest_when_nothing_is_on_time(now):
    options = [opt(KIND_STANDARD, 72, 100.0, 1.0), opt(KIND_EXPEDITED, 24, 150.0, 90.0, mode="air")]
    ranked = rank_options(options, now, now + pd.Timedelta(hours=6), 0.5)
    assert not ranked["on_time"].any()
    assert ranked.iloc[0]["kind"] == KIND_EXPEDITED
    chosen, on_time = best_option(options, now, now + pd.Timedelta(hours=6), 0.5)
    assert chosen["kind"] == KIND_EXPEDITED and not on_time


def test_reserve_stock_reduces_free_stock(tables):
    inv = tables.inventory
    before = free_stock(inv)[("DEL", "P-HPC-BLD")]
    after = free_stock(reserve_stock(inv, "DEL", "P-HPC-BLD"))[("DEL", "P-HPC-BLD")]
    assert after == before - 1
    with pytest.raises(ValueError):
        reserve_stock(inv, "HYD", "P-HPC-BLD")  # nothing free at HYD


def test_plan_engine_for_every_fleet_engine(ctx, tables, now):
    stock = free_stock(tables.inventory)
    for row in tables.fleet.itertuples():
        rec = plan_engine(ctx, row.engine_id, row.current_cycle, now, stock, 0.5)
        assert rec.need_by >= now
        assert rec.rul_lower <= rec.predicted_rul <= rec.rul_upper
        assert not rec.options.empty
        assert "authorised maintenance personnel decide" in rec.rationale


def test_most_urgent_engine_at_start_is_escalated(ctx, tables, now):
    stock = free_stock(tables.inventory)
    recs = [plan_engine(ctx, r.engine_id, r.current_cycle, now, stock, 0.5) for r in tables.fleet.itertuples()]
    worst = min(recs, key=lambda r: r.rul_lower)
    assert worst.tier == TIER_REVIEW
    assert worst.escalated and worst.shortfall_hours > 0
    assert worst.chosen["lead_hours"] == worst.options["lead_hours"].min()
