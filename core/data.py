"""Load and validate every table the planning engine reads."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]

SCHEMA = {
    "predictions": ["engine_id", "cycle", "predicted_rul", "rul_lower", "rul_upper", "true_rul", "top_sensors", "model_version"],
    "parts": ["part_id", "part_name", "component", "unit_weight_kg", "unit_cost_usd"],
    "locations": ["location_id", "name", "type", "lat", "lon"],
    "inventory": ["location_id", "part_id", "qty_on_hand", "qty_reserved", "reorder_point"],
    "transfers": ["from_location", "to_location", "mode", "transit_hours", "cost_usd", "distance_km"],
    "suppliers": [
        "supplier_id", "supplier_name", "part_id", "origin_location", "lead_time_days_standard",
        "lead_time_days_expedited", "unit_price_usd", "expedite_fee_usd", "standard_mode", "distance_km",
    ],
    "fleet": ["aircraft_id", "engine_id", "base_location", "cycles_per_day", "next_departure", "current_cycle"],
    "emission_factors": ["mode", "kg_co2e_per_tonne_km", "source_url"],
}

PATHS = {
    "predictions": "outputs/predictions.csv",
    "parts": "data/synthetic/parts.csv",
    "locations": "data/synthetic/locations.csv",
    "inventory": "data/synthetic/inventory.csv",
    "transfers": "data/synthetic/transfers.csv",
    "suppliers": "data/synthetic/suppliers.csv",
    "fleet": "data/synthetic/fleet.csv",
    "emission_factors": "data/reference/emission_factors.csv",
}


class SchemaError(ValueError):
    """A table is missing columns the planning engine needs."""


@dataclass(frozen=True)
class Tables:
    predictions: pd.DataFrame
    parts: pd.DataFrame
    locations: pd.DataFrame
    inventory: pd.DataFrame
    transfers: pd.DataFrame
    suppliers: pd.DataFrame
    fleet: pd.DataFrame
    emission_factors: pd.DataFrame


def load_config(root: Path = ROOT) -> dict:
    return yaml.safe_load((Path(root) / "config.yaml").read_text())


def validate(name: str, df: pd.DataFrame) -> None:
    missing = [c for c in SCHEMA[name] if c not in df.columns]
    if missing:
        raise SchemaError(f"{PATHS[name]} is missing columns {missing}. Fix the file at its source, not in the app.")


def load_tables(root: Path = ROOT, tz: str = "Asia/Kolkata") -> Tables:
    root = Path(root)
    frames = {}
    for name, rel in PATHS.items():
        df = pd.read_csv(root / rel)
        validate(name, df)
        frames[name] = df
    fleet = frames["fleet"].copy()
    fleet["next_departure"] = pd.to_datetime(fleet["next_departure"], utc=True).dt.tz_convert(tz)
    frames["fleet"] = fleet
    preds = frames["predictions"].copy()
    preds["cycle"] = preds["cycle"].astype(int)
    frames["predictions"] = preds
    return Tables(**frames)
