"""Score one engine's sensor history: the live counterpart of outputs/predictions.csv.

    from ml.inference import predict_rul
    result = predict_rul(history)           # the promoted (primary) model
    result = predict_rul(history, "lgbm")   # or a named model: "lgbm" or "lstm"

history: one engine's raw C-MAPSS rows up to the latest completed cycle, at least 30 rows, with a
`cycle` column and the 14 sensor columns (s2 ... s21). Other columns are ignored.
Returns predicted_rul, rul_lower, rul_upper (cycles), top_sensors (3 names) and model_version.
The LSTM needs PyTorch (requirements-dev.txt); the LightGBM path does not.

Command line, from the repo root:
    python -m ml.inference --engine FD001-T-003            # official test engine, last cycle
    python -m ml.inference --engine FD001-R-074 --cycle 150 --model lgbm
"""
from __future__ import annotations

import argparse
import json
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from ml.data_prep import FEATURES, SENSORS, add_features, load_raw, sensor_label, unit_from_engine_id

ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT / "models"
WINDOW = 30
KINDS = ("lgbm", "lstm")


def primary_model() -> str:
    """The model promoted by ml.select_model."""
    version = json.loads((ROOT / "outputs" / "metrics.json").read_text())["model_version"]
    return "lstm" if version.startswith("lstm") else "lgbm"


def _validate(history: pd.DataFrame) -> pd.DataFrame:
    missing = [c for c in ["cycle"] + SENSORS if c not in history.columns]
    if missing:
        raise ValueError(f"history is missing columns {missing}")
    h = history.sort_values("cycle").reset_index(drop=True)
    if h["cycle"].duplicated().any():
        raise ValueError("history has repeated cycles")
    if len(h) < WINDOW:
        raise ValueError(f"need at least {WINDOW} cycles of history, got {len(h)}")
    if h[SENSORS].isna().any().any():
        raise ValueError("history has missing sensor readings")
    return h.assign(unit=0)


@lru_cache(maxsize=1)
def _lgbm():
    import lightgbm as lgb

    booster = lgb.Booster(model_file=str(MODELS_DIR / "lgbm_rul.txt"))
    calibration = json.loads((MODELS_DIR / "band_calibration.json").read_text())
    return booster, calibration


@lru_cache(maxsize=1)
def _lstm():
    from ml.sequence_model import LSTMEnsemble

    return LSTMEnsemble.load(MODELS_DIR / "lstm")


def _predict_lgbm(h: pd.DataFrame) -> dict:
    from ml.baseline_model import MODEL_VERSION, apply_band, predict_point, top_sensors

    booster, calibration = _lgbm()
    x = add_features(h)[FEATURES].iloc[[-1]]
    pred = predict_point(booster, x)
    lower, upper = apply_band(pred, calibration)
    return {
        "predicted_rul": float(pred[0]), "rul_lower": float(lower[0]), "rul_upper": float(upper[0]),
        "top_sensors": top_sensors(booster, x)[0].split(";"), "model_version": MODEL_VERSION,
    }


def _predict_lstm(h: pd.DataFrame) -> dict:
    from ml.sequence_model import MODEL_VERSION, top3

    ens = _lstm()
    X = np.ascontiguousarray(h[SENSORS].to_numpy(np.float32)[-WINDOW:][None])
    mu, sd = ens.mc(X)
    lower, upper = ens.band(mu, sd)
    return {
        "predicted_rul": float(mu[0]), "rul_lower": float(lower[0]), "rul_upper": float(upper[0]),
        "top_sensors": top3(ens.occlusion(X))[0].split(";"), "model_version": MODEL_VERSION,
    }


def predict_rul(history: pd.DataFrame, model: str | None = None) -> dict:
    kind = model or primary_model()
    if kind not in KINDS:
        raise ValueError(f"model must be one of {KINDS}")
    h = _validate(history)
    out = _predict_lgbm(h) if kind == "lgbm" else _predict_lstm(h)
    for k in ("predicted_rul", "rul_lower", "rul_upper"):
        out[k] = round(out[k], 2)
    out["cycle"] = int(h["cycle"].iloc[-1])
    return out


def engine_history(eid: str, cycle: int | None = None) -> pd.DataFrame:
    """Raw history of a C-MAPSS FD001 engine: 'R' engines come from the training file, 'T' from the test file."""
    prefix, unit = unit_from_engine_id(eid)
    raw = load_raw("train_FD001.txt" if prefix == "R" else "test_FD001.txt")
    h = raw[raw["unit"] == unit]
    if h.empty:
        raise ValueError(f"no engine {eid}")
    return h if cycle is None else h[h["cycle"] <= cycle]


def main() -> None:
    ap = argparse.ArgumentParser(description="Estimate remaining useful life for one FD001 engine.")
    ap.add_argument("--engine", required=True, help="e.g. FD001-T-003 (test) or FD001-R-074 (held-out replay)")
    ap.add_argument("--cycle", type=int, default=None, help="score the history up to this cycle (default: last)")
    ap.add_argument("--model", choices=KINDS, default=None, help="default: the promoted model")
    args = ap.parse_args()
    res = predict_rul(engine_history(args.engine, args.cycle), args.model)
    print(f"{args.engine} at cycle {res['cycle']} ({res['model_version']})")
    print(f"  RUL estimate {res['predicted_rul']:.1f} cycles, 80% band {res['rul_lower']:.1f} to {res['rul_upper']:.1f}")
    print("  Top sensors: " + ", ".join(sensor_label(s) for s in res["top_sensors"]))


if __name__ == "__main__":
    main()
