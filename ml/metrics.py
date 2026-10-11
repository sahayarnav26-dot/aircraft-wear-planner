"""Shared evaluation helpers, so every model is scored the same way on the same engines and folds."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

N_FOLDS = 5
BIN_EDGES = [0, 15, 30, 50, 75, 100]   # predicted-RUL ranges (cycles) for band calibration and reporting
BAND_QUANTILES = (0.10, 0.90)          # 80% band


def nasa_score(y_true, y_pred) -> float:
    """PHM08 scoring function: d = predicted - true; late predictions (d > 0) cost more."""
    d = np.asarray(y_pred, dtype=float) - np.asarray(y_true, dtype=float)
    return float(np.sum(np.where(d < 0, np.exp(-d / 13) - 1, np.exp(d / 10) - 1)))


def rmse(y_true, y_pred) -> float:
    d = np.asarray(y_pred, dtype=float) - np.asarray(y_true, dtype=float)
    return float(np.sqrt(np.mean(d * d)))


def bin_index(pred) -> np.ndarray:
    return np.digitize(np.asarray(pred, dtype=float), BIN_EDGES[1:], right=False)


def bin_labels() -> list[str]:
    edges = BIN_EDGES + [None]
    return [f"{a}-{b}" if b is not None else f"{a}+" for a, b in zip(edges[:-1], edges[1:])]


def coverage(lower, upper, y) -> float:
    y = np.asarray(y, dtype=float)
    return float(np.mean((np.asarray(lower) <= y) & (y <= np.asarray(upper))))


def coverage_by_range(pred, lower, upper, y) -> dict[str, dict]:
    """Band coverage within each predicted-RUL range."""
    bins = bin_index(pred)
    out = {}
    for b, label in enumerate(bin_labels()):
        m = bins == b
        out[label] = {
            "n": int(m.sum()),
            "coverage": round(coverage(np.asarray(lower)[m], np.asarray(upper)[m], np.asarray(y)[m]), 3) if m.any() else None,
        }
    return out


def engine_folds(rows: pd.DataFrame, n_splits: int = N_FOLDS) -> dict[int, int]:
    """unit -> fold, from GroupKFold on per-cycle rows. Deterministic, so every model shares the folds."""
    fold_of = {}
    for k, (_, ho_idx) in enumerate(GroupKFold(n_splits=n_splits).split(rows, groups=rows["unit"])):
        for u in rows["unit"].iloc[ho_idx].unique():
            fold_of[int(u)] = k
    return fold_of


def sensor_shares(per_sensor_abs: np.ndarray, sensors: list[str]) -> dict[str, float]:
    """Mean absolute attribution per sensor, as shares that sum to 1."""
    mean_abs = np.asarray(per_sensor_abs, dtype=float).mean(axis=0)
    total = float(mean_abs.sum()) or 1.0
    return {s: round(float(v) / total, 4) for s, v in zip(sensors, mean_abs)}
