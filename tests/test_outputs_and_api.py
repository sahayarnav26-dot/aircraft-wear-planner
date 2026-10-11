import json
from pathlib import Path

import pandas as pd
from fastapi.testclient import TestClient

from core.data import SCHEMA

ROOT = Path(__file__).resolve().parents[1]


def test_predictions_contract():
    p = pd.read_csv(ROOT / "outputs" / "predictions.csv")
    assert list(p.columns) == SCHEMA["predictions"]
    assert p["cycle"].min() >= 30
    assert (p["rul_lower"] <= p["predicted_rul"] + 1e-6).all()
    assert (p["predicted_rul"] <= p["rul_upper"] + 1e-6).all()
    replay = p[p["engine_id"].str.contains("-R-")]
    test = p[p["engine_id"].str.contains("-T-")]
    assert replay["engine_id"].nunique() == 20 and replay["true_rul"].notna().all()
    assert test["engine_id"].nunique() == 100 and test["true_rul"].isna().all()
    assert (p["top_sensors"].str.count(";") == 2).all()


def test_metrics_file_reports_honest_coverage():
    m = json.loads((ROOT / "outputs" / "metrics.json").read_text())
    assert 0.6 <= m["replay_engines_all_cycles"]["band_coverage_vs_capped_true"] <= 0.95
    assert m["test_last_cycle"]["n_engines"] == 100


def test_synthetic_tables_are_consistent(tables):
    sup = tables.suppliers
    assert (sup["lead_time_days_expedited"] < sup["lead_time_days_standard"]).all()
    assert set(tables.parts["part_id"]) <= set(sup["part_id"])
    inv = tables.inventory
    assert (inv["qty_reserved"] <= inv["qty_on_hand"]).all() and (inv[["qty_on_hand", "qty_reserved"]] >= 0).all().all()
    assert set(tables.fleet["engine_id"]) <= set(tables.predictions["engine_id"])
    assert tables.emission_factors["source_url"].str.startswith("https://").all()


def test_api_endpoints():
    from api.main import app

    client = TestClient(app)
    client.post("/reset")
    engines = client.get("/engines").json()["engines"]
    assert len(engines) == 20
    eid = min(engines, key=lambda e: e["rul_lower"])["engine_id"]
    health = client.get(f"/engines/{eid}/health").json()
    assert health["history"] and health["rul_lower"] <= health["rul_upper"]
    plan = client.get(f"/engines/{eid}/plan", params={"weight": 0.3}).json()
    assert plan["recommendation"] and plan["options"]
    landing = client.post(f"/engines/{eid}/landing")
    assert landing.status_code in (200, 409)
    fleet = client.get("/fleet/plan").json()
    assert fleet["engines_needing_action"] >= 1 and fleet["allocation"]
    assert client.get("/engines/NOPE/plan").status_code == 404
