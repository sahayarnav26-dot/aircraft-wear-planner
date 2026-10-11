"""predict_rul() must reproduce the batch predictions the planner reads, and the promotion must follow its rule."""
import json
from pathlib import Path

import pandas as pd
import pytest

from ml.inference import engine_history, predict_rul, primary_model

ROOT = Path(__file__).resolve().parents[1]
CASES = [("FD001-R-074", 150), ("FD001-R-008", 60), ("FD001-T-003", None)]


@pytest.mark.parametrize("model", ["lgbm", "lstm"])
def test_predict_rul_matches_batch_predictions(model):
    if model == "lstm":
        pytest.importorskip("torch")
    preds = pd.read_csv(ROOT / "outputs" / f"predictions_{model}.csv")
    for eid, cycle in CASES:
        res = predict_rul(engine_history(eid, cycle), model)
        row = preds[(preds["engine_id"] == eid) & (preds["cycle"] == res["cycle"])].iloc[0]
        for k in ("predicted_rul", "rul_lower", "rul_upper"):
            assert res[k] == pytest.approx(row[k], abs=0.05), (eid, k)
        assert res["rul_lower"] <= res["predicted_rul"] <= res["rul_upper"]
        assert res["top_sensors"][0] == row["top_sensors"].split(";")[0]
        assert res["model_version"] == row["model_version"]


def test_bad_histories_are_rejected():
    hist = engine_history("FD001-T-003")
    with pytest.raises(ValueError):
        predict_rul(hist.head(10), "lgbm")
    with pytest.raises(ValueError):
        predict_rul(hist.drop(columns=["s11"]), "lgbm")


def test_promotion_follows_the_rule():
    comp = json.loads((ROOT / "outputs" / "model_comparison.json").read_text())
    metrics = json.loads((ROOT / "outputs" / "metrics.json").read_text())
    models = comp["models"]
    assert sum(m["primary"] for m in models) == 1
    winner = next(m for m in models if m["primary"])
    eligible = [m for m in models if m["eligible"]] or models
    assert winner["test_rmse"] == min(m["test_rmse"] for m in eligible)
    assert winner["key"] == primary_model() == metrics["selection"]["primary"]
    promoted = pd.read_csv(ROOT / "outputs" / "predictions.csv")
    assert (promoted["model_version"] == winner["model_version"]).all()
