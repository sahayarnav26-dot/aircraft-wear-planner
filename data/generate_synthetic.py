"""Generate the synthetic supply-chain tables the planning engine reads.

All output is SYNTHETIC demonstration data with a fixed seed. Station names and
coordinates are real Indian airports, but the network, stock, prices, suppliers and
fleet are invented. The scenario is designed so the demo shows every option type:
reserve local stock, transfer, standard order and expedited order.

Run from the repo root, after outputs/predictions.csv exists:
    python data/generate_synthetic.py
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from faker import Faker

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "synthetic"
SEED = 42

# location_id, name, type, lat, lon
LOCATIONS = [
    ("DEL", "Delhi hub", "hub", 28.5562, 77.1000),
    ("BOM", "Mumbai line station", "line station", 19.0896, 72.8656),
    ("BLR", "Bengaluru line station", "line station", 13.1986, 77.7066),
    ("HYD", "Hyderabad line station", "line station", 17.2403, 78.4294),
    ("MAA", "Chennai line station", "line station", 12.9941, 80.1709),
    ("CCU", "Kolkata line station", "line station", 22.6547, 88.4467),
    ("NAG", "Nagpur central warehouse", "warehouse", 21.0922, 79.0472),
]

# part_id, part_name, component, unit_weight_kg, unit_cost_usd (replacement value)
PARTS = [
    ("P-HPC-BLD", "HPC blade set", "HPC", 60.0, 240000),
    ("P-HPC-VNE", "HPC stator vane set", "HPC", 45.0, 150000),
    ("P-HPC-SEL", "HPC seal kit", "HPC", 8.0, 18000),
    ("P-FAN-BLD", "Fan blade set", "FAN", 150.0, 480000),
    ("P-FAN-ACL", "Fan case acoustic liner", "FAN", 35.0, 60000),
]

# Stock of the critical spare (HPC blade set), set by hand so stock is scarce.
# (location_id): (qty_on_hand, qty_reserved, reorder_point)
CRITICAL_STOCK = {
    "DEL": (3, 1, 1),
    "BOM": (1, 0, 0),
    "BLR": (0, 0, 0),
    "HYD": (0, 0, 0),
    "MAA": (1, 0, 0),
    "CCU": (0, 0, 0),
    "NAG": (2, 0, 1),
}

# Suppliers per component family: (origin, distance_km to the network, standard mode,
# standard lead days, expedited lead days, price factor vs replacement value, expedite fee share)
SUPPLIER_PROFILES = {
    "HPC": [
        ("Bengaluru (IN)", 1750, "road", 7, 2, 1.04, 0.06),
        ("Singapore (SG)", 4150, "sea", 24, 3, 0.96, 0.10),
    ],
    "FAN": [
        ("Bengaluru (IN)", 1750, "road", 10, 3, 1.05, 0.06),
        ("Hamburg (DE)", 6100, "sea", 35, 4, 0.95, 0.10),
    ],
}

# Target true cycles left at the scenario start, one per fleet engine (most to least urgent).
TARGET_REMAINING = [7, 16, 24, 31, 38, 46, 55, 63, 72, 80, 90, 100, 110, 120, 135, 150, 165, 180, 200, 220]
# Bases for aircraft AC-01..AC-10. AC-01 carries the most urgent engine at a base with no spare.
AIRCRAFT_BASES = ["HYD", "DEL", "BOM", "CCU", "BLR", "MAA", "DEL", "BOM", "DEL", "BLR"]


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = p2 - p1, math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    Faker.seed(SEED)
    fake = Faker()
    cfg = yaml.safe_load((ROOT / "config.yaml").read_text())
    tz = cfg["timezone"]
    start = pd.Timestamp(cfg["simulation"]["start"]).tz_convert(tz)

    locations = pd.DataFrame(LOCATIONS, columns=["location_id", "name", "type", "lat", "lon"])
    parts = pd.DataFrame(PARTS, columns=["part_id", "part_name", "component", "unit_weight_kg", "unit_cost_usd"])

    # Inventory: hand-set for the critical spare, small random stock for the rest.
    inv_rows = []
    for loc in locations["location_id"]:
        for part in parts["part_id"]:
            if part == "P-HPC-BLD":
                on_hand, reserved, reorder = CRITICAL_STOCK[loc]
            else:
                on_hand = int(rng.integers(0, 4))
                reserved = int(rng.integers(0, on_hand + 1)) if on_hand and rng.random() < 0.2 else 0
                reorder = 1 if loc in ("DEL", "NAG") else 0
            inv_rows.append((loc, part, on_hand, reserved, reorder))
    inventory = pd.DataFrame(inv_rows, columns=["location_id", "part_id", "qty_on_hand", "qty_reserved", "reorder_point"])

    # Transfers: every ordered pair of locations, by road and by air.
    tr_rows = []
    for a in LOCATIONS:
        for b in LOCATIONS:
            if a[0] == b[0]:
                continue
            gc = haversine_km(a[3], a[4], b[3], b[4])
            road_km = round(gc * 1.3)
            air_km = round(gc * 1.05)
            tr_rows.append((a[0], b[0], "road", round(road_km / 45 + 6, 1), round(350 + 0.85 * road_km), road_km))
            tr_rows.append((a[0], b[0], "air", round(6 + air_km / 700, 1), round(1200 + 2.2 * air_km), air_km))
    transfers = pd.DataFrame(
        tr_rows, columns=["from_location", "to_location", "mode", "transit_hours", "cost_usd", "distance_km"]
    )

    # Suppliers: a domestic and an international source for every part.
    sup_rows = []
    n = 1
    for part in PARTS:
        for origin, dist, std_mode, lt_std, lt_exp, price_f, fee_share in SUPPLIER_PROFILES[part[2]]:
            name = f"{fake.last_name()} {'Aero Components' if '(IN)' in origin else 'Turbine Spares'}"
            price = round(part[4] * price_f)
            sup_rows.append(
                (f"SUP-{n:02d}", name, part[0], origin, lt_std, lt_exp, price, round(price * fee_share), std_mode, dist)
            )
            n += 1
    suppliers = pd.DataFrame(
        sup_rows,
        columns=[
            "supplier_id", "supplier_name", "part_id", "origin_location", "lead_time_days_standard",
            "lead_time_days_expedited", "unit_price_usd", "expedite_fee_usd", "standard_mode", "distance_km",
        ],
    )

    # Fleet: the 20 replay engines on 10 aircraft. Urgent engines are paired with healthy ones.
    preds = pd.read_csv(ROOT / "outputs" / "predictions.csv")
    replay = preds[preds["engine_id"].str.contains("-R-")]
    last_cycle = replay.groupby("engine_id")["cycle"].max()
    first_cycle = replay.groupby("engine_id")["cycle"].min()
    engines = list(last_cycle.index)
    rng.shuffle(engines)
    current = {}
    for eid, target in zip(engines, TARGET_REMAINING):
        current[eid] = int(max(first_cycle[eid], last_cycle[eid] - target))
    ordered = engines  # engines[i] carries TARGET_REMAINING[i]
    fleet_rows = []
    for i in range(10):
        ac = f"AC-{i + 1:02d}"
        base = AIRCRAFT_BASES[i]
        cpd = int(rng.integers(3, 7))
        dep = start + pd.Timedelta(minutes=int(rng.integers(6, 48)) * 5)
        for eid in (ordered[i], ordered[19 - i]):
            fleet_rows.append((ac, eid, base, cpd, dep.isoformat(), current[eid]))
    fleet = pd.DataFrame(
        fleet_rows, columns=["aircraft_id", "engine_id", "base_location", "cycles_per_day", "next_departure", "current_cycle"]
    )

    for name, df in [
        ("locations", locations), ("parts", parts), ("inventory", inventory),
        ("transfers", transfers), ("suppliers", suppliers), ("fleet", fleet),
    ]:
        df.to_csv(OUT / f"{name}.csv", index=False)
        print(f"{name}.csv: {len(df)} rows")


if __name__ == "__main__":
    main()
