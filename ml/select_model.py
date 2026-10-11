"""Compare the candidate RUL models on the same engines and promote the winner.

Rule, fixed before the comparison: among models whose 80% band covers 70-90% of true RUL on the
20 held-out replay engines, choose the lowest RMSE on the 100 official test engines (last cycle,
true RUL capped at 125). The winner's files become outputs/predictions.csv and
outputs/metrics.json, which the planner and dashboard read.

Also measures alert lead time on the replay engines: how many cycles (flights) before the end of
simulated life each engine first enters "Plan maintenance" and "Review before next departure".

Run from the repo root, after ml.baseline_model and ml.sequence_model:  python -m ml.select_model
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = ROOT / "outputs"
CANDIDATES = {
    "lgbm": {"label": "LightGBM (rolling features)", "preds": "predictions_lgbm.csv", "metrics": "metrics_lgbm.json"},
    "lstm": {"label": "LSTM (30-cycle windows)", "preds": "predictions_lstm.csv", "metrics": "metrics_lstm.json"},
}
COVERAGE_RANGE = (0.70, 0.90)
RULE = (
    "Among models whose 80% band covers 70-90% of true RUL on the 20 held-out replay engines, "
    "choose the lowest RMSE on the 100 official test engines (last cycle, true RUL capped at 125)."
)


def alert_lead_times(preds: pd.DataFrame, plan_max: float, review_max: float) -> pd.DataFrame:
    """Per replay engine: true cycles left when the lower edge first reaches each tier threshold."""
    rows = []
    replay = preds[preds["engine_id"].str.contains("-R-")]
    for eid, g in replay.sort_values("cycle").groupby("engine_id"):
        def first_hit(threshold: float) -> float:
            hit = g[g["rul_lower"] <= threshold]
            return float(hit["true_rul"].iloc[0]) if len(hit) else float("nan")

        rows.append({"engine_id": eid, "plan_lead_cycles": first_hit(plan_max), "review_lead_cycles": first_hit(review_max)})
    return pd.DataFrame(rows)


def summarise(key: str, cfg: dict) -> dict:
    spec = CANDIDATES[key]
    m = json.loads((OUTPUTS / spec["metrics"]).read_text())
    preds = pd.read_csv(OUTPUTS / spec["preds"])
    p = cfg["planning"]
    leads = alert_lead_times(preds, p["plan_max_rul_lower"], p["review_max_rul_lower"])
    t, r = m["test_last_cycle"], m["replay_engines_all_cycles"]
    cov = r["band_coverage_vs_capped_true"]
    return {
        "key": key,
        "label": spec["label"],
        "model_version": m["model_version"],
        "model_name": m["model_name"],
        "band_method": m["band"],
        "explainer": m["explainer"],
        "test_rmse": t["rmse_vs_capped_true"],
        "test_rmse_uncapped": t["rmse_vs_uncapped_true"],
        "test_nasa_score": t["nasa_score_vs_capped_true"],
        "test_band_coverage": t["band_coverage_vs_capped_true"],
        "test_band_width": t["mean_band_width_cycles"],
        "replay_rmse": r["rmse_vs_capped_true"],
        "replay_band_coverage": cov,
        "replay_band_width": r["mean_band_width_cycles"],
        "plan_alert_lead_median": float(np.nanmedian(leads["plan_lead_cycles"])),
        "review_alert_lead_median": float(np.nanmedian(leads["review_lead_cycles"])),
        "review_alert_lead_min": float(np.nanmin(leads["review_lead_cycles"])),
        "review_alert_lead_max": float(np.nanmax(leads["review_lead_cycles"])),
        "engines_reviewed_with_buffer": int((leads["review_lead_cycles"] >= p["safety_buffer_cycles"]).sum()),
        "engines": int(len(leads)),
        "eligible": bool(COVERAGE_RANGE[0] <= cov <= COVERAGE_RANGE[1]),
    }


def markdown_table(rows: list[dict]) -> str:
    head = (
        "| Model | Test RMSE (cycles) | NASA score | Band coverage, replay (target 80%) | Mean band width | "
        "Plan alert, median lead | Review alert, median lead (min) | Primary |\n"
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | :---: |"
    )
    lines = [head]
    for r in rows:
        lines.append(
            f"| {r['label']} | {r['test_rmse']:.1f} | {r['test_nasa_score']:,.0f} | {r['replay_band_coverage']:.0%} | "
            f"{r['replay_band_width']:.0f} cycles | {r['plan_alert_lead_median']:.0f} cycles | "
            f"{r['review_alert_lead_median']:.0f} ({r['review_alert_lead_min']:.0f}) cycles | {'yes' if r['primary'] else ''} |"
        )
    return "\n".join(lines)


def main() -> None:
    cfg = yaml.safe_load((ROOT / "config.yaml").read_text())
    rows = [summarise(k, cfg) for k in CANDIDATES if (OUTPUTS / CANDIDATES[k]["metrics"]).exists()]
    eligible = [r for r in rows if r["eligible"]] or rows
    winner = min(eligible, key=lambda r: (r["test_rmse"], r["test_nasa_score"]))
    for r in rows:
        r["primary"] = r["key"] == winner["key"]

    spec = CANDIDATES[winner["key"]]
    shutil.copyfile(OUTPUTS / spec["preds"], OUTPUTS / "predictions.csv")
    metrics = json.loads((OUTPUTS / spec["metrics"]).read_text())
    metrics["selection"] = {"rule": RULE, "primary": winner["key"], "candidates": [r["key"] for r in rows]}
    (OUTPUTS / "metrics.json").write_text(json.dumps(metrics, indent=2))

    comparison = {"rule": RULE, "primary": winner["key"], "models": rows}
    (OUTPUTS / "model_comparison.json").write_text(json.dumps(comparison, indent=2))
    (OUTPUTS / "model_comparison.md").write_text(markdown_table(rows) + "\n")
    print(markdown_table(rows))
    print(f"\nPrimary model: {winner['label']} ({winner['model_version']}) -> outputs/predictions.csv")


if __name__ == "__main__":
    main()
