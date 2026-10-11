"""Load NASA C-MAPSS FD001, label remaining useful life (RUL) and build features.

Shared by the baseline model (ml/baseline_quantile.py), the inference helper
(ml/inference.py) and the dashboard's sensor charts.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"

COLS = ["unit", "cycle", "setting_1", "setting_2", "setting_3"] + [f"s{i}" for i in range(1, 22)]

# In FD001, s1, s5, s6, s10, s16, s18, s19 and the settings are constant or nearly so.
SENSORS = ["s2", "s3", "s4", "s7", "s8", "s9", "s11", "s12", "s13", "s14", "s15", "s17", "s20", "s21"]

# Sensor symbols and descriptions from Saxena et al. (2008), Table 2.
SENSOR_INFO = {
    "s2": ("T24", "Total temperature at LPC outlet"),
    "s3": ("T30", "Total temperature at HPC outlet"),
    "s4": ("T50", "Total temperature at LPT outlet"),
    "s7": ("P30", "Total pressure at HPC outlet"),
    "s8": ("Nf", "Physical fan speed"),
    "s9": ("Nc", "Physical core speed"),
    "s11": ("Ps30", "Static pressure at HPC outlet"),
    "s12": ("phi", "Ratio of fuel flow to Ps30"),
    "s13": ("NRf", "Corrected fan speed"),
    "s14": ("NRc", "Corrected core speed"),
    "s15": ("BPR", "Bypass ratio"),
    "s17": ("htBleed", "Bleed enthalpy"),
    "s20": ("W31", "HPT coolant bleed"),
    "s21": ("W32", "LPT coolant bleed"),
}

CAP = 125                 # piecewise-linear RUL cap, standard practice for C-MAPSS
N_REPLAY = 20             # held-out training engines: validation set and dashboard replay engines
SEED = 42
WINDOWS = (5, 10)         # rolling mean / std windows, in cycles
SLOPE_WINDOW = 10         # rolling least-squares slope window, in cycles
FIRST_PREDICTED_CYCLE = 30  # FD001's shortest test engine has 31 cycles

FEATURES = (
    SENSORS
    + [f"{s}_{stat}{w}" for s in SENSORS for w in WINDOWS for stat in ("mean", "std")]
    + [f"{s}_slope{SLOPE_WINDOW}" for s in SENSORS]
)


def sensor_label(sensor: str) -> str:
    """'s11' -> 'Ps30 (Static pressure at HPC outlet)'."""
    symbol, desc = SENSOR_INFO.get(sensor, (sensor, ""))
    return f"{symbol} ({desc})" if desc else symbol


def load_raw(name: str, raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    # sep=r"\s+" copes with the trailing spaces in the raw files
    return pd.read_csv(Path(raw_dir) / name, sep=r"\s+", header=None, names=COLS)


def load_fd001(raw_dir: Path = RAW_DIR) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return (train, test, test_rul); test_rul has columns unit, true_rul."""
    train = load_raw("train_FD001.txt", raw_dir)
    test = load_raw("test_FD001.txt", raw_dir)
    test_rul = pd.read_csv(Path(raw_dir) / "RUL_FD001.txt", sep=r"\s+", header=None, names=["true_rul"])
    test_rul["unit"] = np.arange(1, len(test_rul) + 1)
    return train, test, test_rul


def add_rul_labels(train: pd.DataFrame, cap: int = CAP) -> pd.DataFrame:
    """rul_true = cycles left before failure; rul = rul_true capped at `cap`."""
    out = train.copy()
    last = out.groupby("unit")["cycle"].transform("max")
    out["rul_true"] = last - out["cycle"]
    out["rul"] = out["rul_true"].clip(upper=cap)
    return out


def replay_units(train: pd.DataFrame, n: int = N_REPLAY, seed: int = SEED) -> np.ndarray:
    """Engines held out from training. Split by engine, never by row, to avoid leakage."""
    rng = np.random.default_rng(seed)
    units = np.sort(train["unit"].unique())
    return np.sort(rng.choice(units, size=n, replace=False))


def _rolling_sum(series: pd.Series, units: pd.Series, window: int) -> pd.Series:
    out = series.groupby(units, sort=False).rolling(window, min_periods=1).sum()
    return out.reset_index(level=0, drop=True)


def _rolling_slope(df: pd.DataFrame, col: str, window: int) -> pd.Series:
    """Least-squares slope of `col` against cycle over the last `window` cycles of each engine."""
    units = df["unit"]
    t = df["cycle"].astype(float)
    x = df[col].astype(float)
    n = df.groupby("unit", sort=False).cumcount().add(1).clip(upper=window).astype(float)
    s_t = _rolling_sum(t, units, window)
    s_x = _rolling_sum(x, units, window)
    s_tx = _rolling_sum(t * x, units, window)
    s_tt = _rolling_sum(t * t, units, window)
    den = n * s_tt - s_t * s_t
    num = n * s_tx - s_t * s_x
    slope = np.where(den > 1e-9, num / den.where(den > 1e-9, 1.0), 0.0)
    return pd.Series(slope, index=df.index)


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add rolling features per engine. Uses only current and past cycles (no look-ahead)."""
    df = df.sort_values(["unit", "cycle"]).reset_index(drop=True)
    grouped = df.groupby("unit", sort=False)
    new_cols: dict[str, pd.Series] = {}
    for s in SENSORS:
        for w in WINDOWS:
            roll = grouped[s].rolling(w, min_periods=1)
            new_cols[f"{s}_mean{w}"] = roll.mean().reset_index(level=0, drop=True)
            new_cols[f"{s}_std{w}"] = roll.std().reset_index(level=0, drop=True).fillna(0.0)
        new_cols[f"{s}_slope{SLOPE_WINDOW}"] = _rolling_slope(df, s, SLOPE_WINDOW)
    return pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)


def engine_id(prefix: str, unit: int) -> str:
    """prefix 'R' = replay (held-out training engine), 'T' = test engine."""
    return f"FD001-{prefix}-{int(unit):03d}"


def unit_from_engine_id(eid: str) -> tuple[str, int]:
    """'FD001-R-017' -> ('R', 17)."""
    _, prefix, unit = eid.split("-")
    return prefix, int(unit)
