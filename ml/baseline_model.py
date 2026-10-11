"""Baseline: LightGBM RUL model with an 80% band from cross-validated errors.

Method
  1. Point model: LightGBM regression on 84 rolling sensor features, labels capped at 125 cycles.
  2. Band: the same 5 engine-wise folds as the LSTM give out-of-fold errors. Within each
     predicted-RUL range, the 10th and 90th percentiles of those errors set the band (a binned,
     split-conformal style interval), so the band narrows as wear progresses.
  3. Explanation: exact TreeSHAP contributions (LightGBM pred_contrib), summed per sensor.
  4. The 20 replay engines are never used for fitting, tuning or calibration.

Writes outputs/predictions_lgbm.csv, outputs/metrics_lgbm.json, models/lgbm_rul.txt and
models/band_calibration.json. ml/select_model.py then promotes the better model to
outputs/predictions.csv, which the planner reads.

Run from the repo root:  python -m ml.baseline_model
"""
from __future__ import annotations

import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from ml.data_prep import (
    CAP,
    FEATURES,
    FIRST_PREDICTED_CYCLE,
    SEED,
    SENSORS,
    add_features,
    add_rul_labels,
    engine_id,
    load_fd001,
    replay_units,
)
from ml.metrics import (
    BAND_QUANTILES,
    BIN_EDGES,
    N_FOLDS,
    bin_index,
    coverage,
    coverage_by_range,
    engine_folds,
    nasa_score,
    rmse,
    sensor_shares,
)

ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = ROOT / "outputs"
MODELS = ROOT / "models"
MODEL_VERSION = "lgbm-cvband-v1"

PARAMS = {
    "objective": "regression",
    "learning_rate": 0.03,
    "num_leaves": 31,
    "min_data_in_leaf": 50,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "lambda_l2": 1.0,
    "seed": SEED,
    "deterministic": True,
    "force_row_wise": True,
    "num_threads": 1,
    "verbose": -1,
}


def fit_band_calibration(pred: np.ndarray, y: np.ndarray) -> dict:
    """Error quantiles (y - pred) per predicted-RUL bin."""
    resid = y - pred
    bins = bin_index(pred)
    table = []
    for b in range(len(BIN_EDGES)):
        r = resid[bins == b]
        table.append({
            "pred_from": BIN_EDGES[b],
            "pred_to": BIN_EDGES[b + 1] if b + 1 < len(BIN_EDGES) else None,
            "n": int(r.size),
            "err_lo": float(np.quantile(r, BAND_QUANTILES[0])),
            "err_hi": float(np.quantile(r, BAND_QUANTILES[1])),
        })
    return {"quantiles": list(BAND_QUANTILES), "bins": table}


def apply_band(pred: np.ndarray, calibration: dict) -> tuple[np.ndarray, np.ndarray]:
    bins = bin_index(pred)
    err_lo = np.array([row["err_lo"] for row in calibration["bins"]])[bins]
    err_hi = np.array([row["err_hi"] for row in calibration["bins"]])[bins]
    lower = np.clip(pred + np.minimum(err_lo, 0.0), 0.0, None)
    upper = pred + np.maximum(err_hi, 0.0)
    return lower, upper


def predict_point(booster: lgb.Booster, X: pd.DataFrame) -> np.ndarray:
    return np.clip(booster.predict(X), 0.0, None)


def sensor_contributions(booster: lgb.Booster, X: pd.DataFrame) -> np.ndarray:
    """(n, n_sensors) TreeSHAP contributions (LightGBM pred_contrib), summed over each sensor's features."""
    contrib = booster.predict(X, pred_contrib=True)[:, :-1]
    sensor_of = np.array([n.split("_")[0] for n in booster.feature_name()])
    return np.column_stack([contrib[:, sensor_of == s].sum(axis=1) for s in SENSORS])


def top_sensors(booster: lgb.Booster, X: pd.DataFrame, k: int = 3) -> list[str]:
    """Top-k sensors by absolute TreeSHAP contribution."""
    per_sensor = sensor_contributions(booster, X)
    order = np.argsort(-np.abs(per_sensor), axis=1)[:, :k]
    sensors = np.array(SENSORS)
    return [";".join(sensors[row]) for row in order]


def cross_validate(fit: pd.DataFrame, fold_of: dict[int, int]) -> tuple[np.ndarray, list[int]]:
    """Out-of-fold predictions on the shared engine-wise folds; returns (oof_pred, best_iterations)."""
    oof = np.zeros(len(fit))
    best_iters = []
    folds = fit["unit"].map(fold_of).to_numpy()
    for k in range(N_FOLDS):
        ho_idx, tr_idx = np.where(folds == k)[0], np.where(folds != k)[0]
        tr, ho = fit.iloc[tr_idx], fit.iloc[ho_idx]
        dtr = lgb.Dataset(tr[FEATURES], label=tr["rul"])
        dho = lgb.Dataset(ho[FEATURES], label=ho["rul"], reference=dtr)
        booster = lgb.train(
            PARAMS, dtr, num_boost_round=3000, valid_sets=[dho],
            callbacks=[lgb.early_stopping(100, verbose=False)],
        )
        best_iters.append(int(booster.best_iteration))
        oof[ho_idx] = predict_point(booster, ho[FEATURES])
    return oof, best_iters


def main() -> None:
    OUTPUTS.mkdir(exist_ok=True)
    MODELS.mkdir(exist_ok=True)

    train_raw, test_raw, test_rul = load_fd001()
    train = add_features(add_rul_labels(train_raw))
    test = add_features(test_raw)

    replay = replay_units(train_raw)
    fit = train[~train["unit"].isin(replay)].reset_index(drop=True)
    val = train[train["unit"].isin(replay)].reset_index(drop=True)

    # 1) Engine-wise cross-validation on the 80 fitting engines -> out-of-fold errors.
    oof, best_iters = cross_validate(fit, engine_folds(fit))
    use = fit["cycle"].to_numpy() >= FIRST_PREDICTED_CYCLE
    calibration = fit_band_calibration(oof[use], fit["rul"].to_numpy(float)[use])

    # 2) Final point model on all 80 fitting engines, rounds = mean of the fold optima.
    n_rounds = int(np.mean(best_iters))
    booster = lgb.train(PARAMS, lgb.Dataset(fit[FEATURES], label=fit["rul"]), num_boost_round=n_rounds)
    booster.save_model(str(MODELS / "lgbm_rul.txt"))
    calibration.update({"model_version": MODEL_VERSION, "n_rounds": n_rounds, "fold_best_iterations": best_iters})
    (MODELS / "band_calibration.json").write_text(json.dumps(calibration, indent=2))

    def score_rows(rows: pd.DataFrame, prefix: str) -> tuple[pd.DataFrame, np.ndarray]:
        rows = rows[rows["cycle"] >= FIRST_PREDICTED_CYCLE].copy()
        pred = predict_point(booster, rows[FEATURES])
        lower, upper = apply_band(pred, calibration)
        rows["predicted_rul"], rows["rul_lower"], rows["rul_upper"] = pred, lower, upper
        contrib = sensor_contributions(booster, rows[FEATURES])
        order = np.argsort(-np.abs(contrib), axis=1)[:, :3]
        rows["top_sensors"] = [";".join(np.array(SENSORS)[r]) for r in order]
        rows["engine_id"] = [engine_id(prefix, u) for u in rows["unit"]]
        return rows, contrib

    # Replay engines: true RUL known (uncapped) so the dashboard can show reality.
    val_rows, val_contrib = score_rows(val, "R")
    val_rows["true_rul"] = val_rows["rul_true"].astype(float)
    # Test engines: true RUL withheld from the planner (blank).
    test_rows, _ = score_rows(test, "T")
    test_rows["true_rul"] = np.nan

    cols = ["engine_id", "cycle", "predicted_rul", "rul_lower", "rul_upper", "true_rul", "top_sensors", "model_version"]
    preds = pd.concat([val_rows, test_rows], ignore_index=True)
    preds["model_version"] = MODEL_VERSION
    for c in ("predicted_rul", "rul_lower", "rul_upper"):
        preds[c] = preds[c].round(2)
    preds[cols].to_csv(OUTPUTS / "predictions_lgbm.csv", index=False)

    # ---- Metrics -------------------------------------------------------------------------
    last = (
        test_rows.sort_values("cycle").groupby("unit").tail(1)[["unit", "predicted_rul", "rul_lower", "rul_upper"]]
        .merge(test_rul.rename(columns={"true_rul": "rul_from_file"}), on="unit")
    )
    t_uncapped = last["rul_from_file"].to_numpy(float)
    t_capped = np.minimum(t_uncapped, CAP)
    p_last = last["predicted_rul"].to_numpy(float)
    v_true = val_rows["rul"].to_numpy(float)

    metrics = {
        "model_version": MODEL_VERSION,
        "model_name": "LightGBM gradient-boosted trees on rolling sensor features",
        "explainer": "TreeSHAP",
        "dataset": "NASA C-MAPSS FD001 (simulated engine data)",
        "split": "80 engines for fitting and 5-fold calibration; 20 held-out replay engines; 100 official test engines",
        "label_cap_cycles": CAP,
        "band": "80% band: 10th-90th percentile of engine-wise cross-validated errors, per predicted-RUL range",
        "test_last_cycle": {
            "n_engines": int(len(last)),
            "rmse_vs_capped_true": round(rmse(t_capped, p_last), 2),
            "rmse_vs_uncapped_true": round(rmse(t_uncapped, p_last), 2),
            "nasa_score_vs_capped_true": round(nasa_score(t_capped, p_last), 1),
            "nasa_score_vs_uncapped_true": round(nasa_score(t_uncapped, p_last), 1),
            "band_coverage_vs_capped_true": round(coverage(last["rul_lower"], last["rul_upper"], t_capped), 3),
            "mean_band_width_cycles": round(float(np.mean(last["rul_upper"] - last["rul_lower"])), 1),
        },
        "replay_engines_all_cycles": {
            "n_engines": int(val_rows["unit"].nunique()),
            "n_rows": int(len(val_rows)),
            "rmse_vs_capped_true": round(rmse(v_true, val_rows["predicted_rul"]), 2),
            "band_coverage_vs_capped_true": round(coverage(val_rows["rul_lower"], val_rows["rul_upper"], v_true), 3),
            "mean_band_width_cycles": round(float(np.mean(val_rows["rul_upper"] - val_rows["rul_lower"])), 1),
            "coverage_by_range": coverage_by_range(val_rows["predicted_rul"], val_rows["rul_lower"], val_rows["rul_upper"], v_true),
        },
        "cross_validation": {
            "folds": N_FOLDS,
            "oof_rmse_vs_capped_true": round(rmse(fit["rul"].to_numpy(float)[use], oof[use]), 2),
            "fold_best_iterations": best_iters,
            "final_rounds": n_rounds,
        },
        "sensor_importance": sensor_shares(np.abs(val_contrib), SENSORS),
        "replay_units": [int(u) for u in replay],
    }
    (OUTPUTS / "metrics_lgbm.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps({k: metrics[k] for k in ("test_last_cycle", "cross_validation")}, indent=2))
    print(f"Wrote {len(preds)} rows to outputs/predictions_lgbm.csv")


if __name__ == "__main__":
    main()
