"""LSTM sequence model for RUL, with Monte Carlo dropout and a conformal-calibrated 80% band.

Method
  1. Input: the last 30 cycles of the 14 informative sensors, standardised with statistics from
     each fold's training engines. Labels capped at 125 cycles (same as the baseline).
  2. Model: two stacked LSTM layers (64 units) -> dense 32 -> RUL. Variational dropout (p = 0.25,
     one mask per sequence shared across time steps) between the layers and before the head.
  3. Training: the same 5 engine-wise folds of the 80 fitting engines as the LightGBM baseline.
     Each fold model early-stops on its held-out engines; the 5 fold models form the ensemble.
  4. Uncertainty: Monte Carlo dropout with 25 fixed dropout masks per model gives every estimate
     a mean and a spread. Read as a Gaussian 80% band, that spread is too narrow, so the band is
     calibrated on out-of-fold errors with per-RUL-range normalised conformal prediction:
         lower = mean + q10 x spread,  upper = mean + q90 x spread.
     Fixed masks make each estimate deterministic and independent of batch size.
  5. Explanation: occlusion sensitivity. Each sensor's 30-cycle window is replaced by its
     training mean and the change in the estimate is measured; the top 3 sensors are reported.
  6. The 20 replay engines are never used for fitting, early stopping or calibration.

Writes outputs/predictions_lstm.csv, outputs/metrics_lstm.json and models/lstm/.
Run from the repo root:  python -m ml.sequence_model   (about 10 minutes on 2 CPU cores)
"""
from __future__ import annotations

import copy
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

from ml.data_prep import (
    CAP,
    FIRST_PREDICTED_CYCLE,
    SEED,
    SENSORS,
    add_rul_labels,
    engine_id,
    load_fd001,
    replay_units,
)
from ml.metrics import (
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
MODEL_DIR = ROOT / "models" / "lstm"
MODEL_VERSION = "lstm-mc-v1"

WINDOW = 30
HIDDEN = 64
DENSE = 32
DROPOUT = 0.25
BATCH = 256
LR = 1e-3
MAX_EPOCHS = 60
PATIENCE = 10
MC_PASSES = 25
PRED_BATCH = 4096
Z80 = 1.2816  # standard-normal 90th percentile: a Gaussian 80% band is mean +/- 1.2816 x spread


# ---------------------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------------------
class LockedDropout(nn.Module):
    """Variational dropout: one mask per sequence, shared across time steps.

    Training draws a fresh random mask per sequence. For Monte Carlo inference, set_mc_masks()
    fixes mask number t for every sequence, so estimates do not depend on batching.
    """

    def __init__(self, size: int, p: float):
        super().__init__()
        self.size, self.p = size, p
        self.fixed: torch.Tensor | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.fixed is not None:
            return x * self.fixed
        if not self.training or self.p == 0:
            return x
        shape = (x.shape[0], 1, x.shape[-1]) if x.dim() == 3 else (x.shape[0], x.shape[-1])
        return x * (x.new_empty(shape).bernoulli_(1 - self.p) / (1 - self.p))


class RULNet(nn.Module):
    def __init__(self, n_features: int = len(SENSORS), hidden: int = HIDDEN, dense: int = DENSE, p: float = DROPOUT):
        super().__init__()
        self.lstm1 = nn.LSTM(n_features, hidden, batch_first=True)
        self.drop1 = LockedDropout(hidden, p)
        self.lstm2 = nn.LSTM(hidden, hidden, batch_first=True)
        self.drop2 = LockedDropout(hidden, p)
        self.fc = nn.Linear(hidden, dense)
        self.drop3 = LockedDropout(dense, p)
        self.out = nn.Linear(dense, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, _ = self.lstm1(x)
        h, _ = self.lstm2(self.drop1(h))
        h = self.drop2(h[:, -1, :])
        h = self.drop3(torch.relu(self.fc(h)))
        return self.out(h).squeeze(-1)  # RUL / CAP


def set_mc_masks(model: nn.Module, t: int, seed: int) -> None:
    g = torch.Generator().manual_seed(seed * 1000 + t)
    for m in model.modules():
        if isinstance(m, LockedDropout):
            m.fixed = torch.empty(m.size).bernoulli_(1 - m.p, generator=g) / (1 - m.p)


def clear_mc_masks(model: nn.Module) -> None:
    for m in model.modules():
        if isinstance(m, LockedDropout):
            m.fixed = None


@torch.no_grad()
def _forward(model: nn.Module, X: np.ndarray) -> np.ndarray:
    model.eval()
    parts = [model(torch.from_numpy(X[i:i + PRED_BATCH])) for i in range(0, len(X), PRED_BATCH)]
    return torch.cat(parts).numpy().astype(np.float64) * CAP


def predict_det(model: nn.Module, X: np.ndarray) -> np.ndarray:
    """Deterministic estimate (no dropout)."""
    clear_mc_masks(model)
    return _forward(model, X)


def mc_samples(model: nn.Module, X: np.ndarray, seed: int, passes: int = MC_PASSES) -> np.ndarray:
    """(passes, n) Monte Carlo dropout estimates with fixed masks."""
    out = np.empty((passes, len(X)))
    for t in range(passes):
        set_mc_masks(model, t, seed)
        out[t] = _forward(model, X)
    clear_mc_masks(model)
    return out


# ---------------------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------------------
def build_windows(df: pd.DataFrame, first_cycle: int = FIRST_PREDICTED_CYCLE) -> tuple[np.ndarray, pd.DataFrame]:
    """Unscaled sensor windows ending at every cycle >= first_cycle.

    Returns X with shape (n, WINDOW, n_sensors) and keys with unit, cycle (and rul, rul_true if present).
    """
    xs, keys = [], []
    label_cols = [c for c in ("rul", "rul_true") if c in df.columns]
    for _, g in df.sort_values(["unit", "cycle"]).groupby("unit", sort=True):
        if len(g) < WINDOW:
            continue
        x = g[SENSORS].to_numpy(np.float32)
        windows = np.lib.stride_tricks.sliding_window_view(x, WINDOW, axis=0).transpose(0, 2, 1)
        ends = g.iloc[WINDOW - 1:]
        keep = ends["cycle"].to_numpy() >= first_cycle
        xs.append(windows[keep])
        keys.append(ends.loc[keep, ["unit", "cycle"] + label_cols])
    return np.ascontiguousarray(np.concatenate(xs)), pd.concat(keys, ignore_index=True)


def fit_scaler(rows: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    mean = rows[SENSORS].mean().to_numpy(np.float32)
    std = rows[SENSORS].std().to_numpy(np.float32)
    std[std < 1e-6] = 1.0
    return mean, std


# ---------------------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------------------
def train_one(Xtr, ytr, Xva, yva, seed: int) -> tuple[RULNet, int, float, list[float]]:
    torch.manual_seed(seed)
    model = RULNet()
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=4)
    xt = torch.from_numpy(Xtr)
    yt = torch.from_numpy((np.asarray(ytr, np.float32) / CAP).astype(np.float32))
    gen = torch.Generator().manual_seed(seed)
    best, best_state, best_epoch, bad, curve = float("inf"), None, 0, 0, []
    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        perm = torch.randperm(len(xt), generator=gen)
        for i in range(0, len(perm), BATCH):
            b = perm[i:i + BATCH]
            opt.zero_grad()
            loss = nn.functional.mse_loss(model(xt[b]), yt[b])
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        val = rmse(yva, predict_det(model, Xva))
        curve.append(round(val, 3))
        sched.step(val)
        if val < best - 1e-3:
            best, best_epoch, bad = val, epoch, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            bad += 1
            if bad >= PATIENCE:
                break
    model.load_state_dict(best_state)
    model.eval()
    return model, best_epoch, best, curve


# ---------------------------------------------------------------------------------------
# Ensemble, calibration and explanation
# ---------------------------------------------------------------------------------------
class LSTMEnsemble:
    """The 5 fold models; each keeps its own scaler and dropout-mask seed."""

    def __init__(self, members: list[dict], calibration: dict | None = None):
        self.members = members
        self.calibration = calibration

    @classmethod
    def load(cls, model_dir: Path = MODEL_DIR) -> "LSTMEnsemble":
        members = []
        for k in range(N_FOLDS):
            ckpt = torch.load(Path(model_dir) / f"fold{k}.pt", map_location="cpu", weights_only=False)
            model = RULNet()
            model.load_state_dict(ckpt["state_dict"])
            model.eval()
            members.append({"model": model, "mean": np.asarray(ckpt["mean"], np.float32),
                            "std": np.asarray(ckpt["std"], np.float32), "seed": int(ckpt["seed"])})
        calibration = json.loads((Path(model_dir) / "calibration.json").read_text())
        return cls(members, calibration)

    def scaled(self, member: dict, X_raw: np.ndarray) -> np.ndarray:
        return ((X_raw - member["mean"]) / member["std"]).astype(np.float32)

    def mc(self, X_raw: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Ensemble mean of the MC means, and the average within-model MC spread."""
        mus, sds = [], []
        for m in self.members:
            s = mc_samples(m["model"], self.scaled(m, X_raw), m["seed"])
            mus.append(s.mean(axis=0))
            sds.append(s.std(axis=0))
        return np.clip(np.mean(mus, axis=0), 0.0, None), np.mean(sds, axis=0)

    def occlusion(self, X_raw: np.ndarray) -> np.ndarray:
        """(n, n_sensors) mean |change in estimate| when a sensor's window is set to its training mean."""
        total = np.zeros((len(X_raw), len(SENSORS)))
        for m in self.members:
            Xs = self.scaled(m, X_raw)
            base = predict_det(m["model"], Xs)
            for j in range(len(SENSORS)):
                Xo = Xs.copy()
                Xo[:, :, j] = 0.0  # standardised training mean
                total[:, j] += np.abs(predict_det(m["model"], Xo) - base)
        return total / len(self.members)

    def band(self, mu: np.ndarray, sd: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return apply_calibration(mu, sd, self.calibration)


def fit_calibration(mu: np.ndarray, sd: np.ndarray, y: np.ndarray) -> dict:
    floor = float(max(0.5, np.quantile(sd, 0.05)))
    r = (y - mu) / np.maximum(sd, floor)
    bins = bin_index(mu)
    table = []
    for b in range(len(BIN_EDGES)):
        rb = r[bins == b]
        table.append({
            "pred_from": BIN_EDGES[b], "pred_to": BIN_EDGES[b + 1] if b + 1 < len(BIN_EDGES) else None,
            "n": int(rb.size), "q_lo": float(np.quantile(rb, 0.10)), "q_hi": float(np.quantile(rb, 0.90)),
        })
    return {
        "method": "Per-RUL-range normalised conformal calibration of the MC-dropout spread on out-of-fold errors",
        "quantiles": [0.10, 0.90], "spread_floor": floor, "bins": table,
    }


def apply_calibration(mu: np.ndarray, sd: np.ndarray, cal: dict) -> tuple[np.ndarray, np.ndarray]:
    spread = np.maximum(sd, cal["spread_floor"])
    b = bin_index(mu)
    q_lo = np.array([row["q_lo"] for row in cal["bins"]])[b]
    q_hi = np.array([row["q_hi"] for row in cal["bins"]])[b]
    lower = np.clip(mu + np.minimum(q_lo, 0.0) * spread, 0.0, None)
    upper = mu + np.maximum(q_hi, 0.0) * spread
    return lower, upper


def top3(attrib: np.ndarray) -> list[str]:
    order = np.argsort(-attrib, axis=1)[:, :3]
    names = np.array(SENSORS)
    return [";".join(names[row]) for row in order]


# ---------------------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------------------
def main() -> None:
    torch.set_num_threads(max(1, os.cpu_count() or 1))
    OUTPUTS.mkdir(exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    t_start = time.time()

    train_raw, test_raw, test_rul = load_fd001()
    train = add_rul_labels(train_raw)
    replay = replay_units(train_raw)
    fit = train[~train["unit"].isin(replay)].reset_index(drop=True)
    val = train[train["unit"].isin(replay)].reset_index(drop=True)
    fold_of = engine_folds(fit)

    X_fit, k_fit = build_windows(fit)
    X_rep, k_rep = build_windows(val)
    X_test, k_test = build_windows(test_raw)
    y_fit = k_fit["rul"].to_numpy(float)

    # 1) Five fold models, out-of-fold MC estimates for calibration.
    oof_mu, oof_sd, oof_det = np.zeros(len(X_fit)), np.zeros(len(X_fit)), np.zeros(len(X_fit))
    members, folds_info = [], []
    for k in range(N_FOLDS):
        t0 = time.time()
        ho_units = sorted(u for u, f in fold_of.items() if f == k)
        ho = k_fit["unit"].isin(ho_units).to_numpy()
        mean, std = fit_scaler(fit[~fit["unit"].isin(ho_units)])
        Xtr = ((X_fit[~ho] - mean) / std).astype(np.float32)
        Xho = ((X_fit[ho] - mean) / std).astype(np.float32)
        seed = SEED + k
        model, best_epoch, best_val, curve = train_one(Xtr, y_fit[~ho], Xho, y_fit[ho], seed)
        samples = mc_samples(model, Xho, seed)
        oof_mu[ho], oof_sd[ho] = np.clip(samples.mean(axis=0), 0, None), samples.std(axis=0)
        oof_det[ho] = predict_det(model, Xho)
        torch.save({"state_dict": model.state_dict(), "mean": mean, "std": std, "seed": seed,
                    "config": {"window": WINDOW, "hidden": HIDDEN, "dense": DENSE, "dropout": DROPOUT,
                               "sensors": SENSORS, "cap": CAP}},
                   MODEL_DIR / f"fold{k}.pt")
        members.append({"model": model, "mean": mean, "std": std, "seed": seed})
        folds_info.append({"fold": k, "holdout_engines": len(ho_units), "best_epoch": best_epoch,
                           "holdout_rmse": round(best_val, 2), "epochs_run": len(curve),
                           "seconds": round(time.time() - t0, 1)})
        print(f"fold {k}: best epoch {best_epoch}, held-out RMSE {best_val:.2f}, {time.time() - t0:.0f}s", flush=True)

    calibration = fit_calibration(oof_mu, oof_sd, y_fit)
    calibration.update({"model_version": MODEL_VERSION, "mc_passes": MC_PASSES})
    (MODEL_DIR / "calibration.json").write_text(json.dumps(calibration, indent=2))
    ens = LSTMEnsemble(members, calibration)

    # 2) Replay and test engines: ensemble MC estimate, calibrated band, occlusion explanation.
    def score(X: np.ndarray, keys: pd.DataFrame, prefix: str) -> pd.DataFrame:
        mu, sd = ens.mc(X)
        lower, upper = ens.band(mu, sd)
        occ = ens.occlusion(X)
        out = keys[["unit", "cycle"]].copy()
        out["engine_id"] = [engine_id(prefix, u) for u in out["unit"]]
        out["predicted_rul"], out["rul_lower"], out["rul_upper"] = mu, lower, upper
        out["raw_lower"], out["raw_upper"] = np.clip(mu - Z80 * sd, 0, None), mu + Z80 * sd
        out["top_sensors"] = top3(occ)
        return out, occ

    rep, occ_rep = score(X_rep, k_rep, "R")
    rep["true_rul"] = k_rep["rul_true"].to_numpy(float)
    rep["rul_capped"] = k_rep["rul"].to_numpy(float)
    tst, _ = score(X_test, k_test, "T")
    tst["true_rul"] = np.nan

    cols = ["engine_id", "cycle", "predicted_rul", "rul_lower", "rul_upper", "true_rul", "top_sensors", "model_version"]
    preds = pd.concat([rep, tst], ignore_index=True)
    preds["model_version"] = MODEL_VERSION
    for c in ("predicted_rul", "rul_lower", "rul_upper"):
        preds[c] = preds[c].round(2)
    preds[cols].to_csv(OUTPUTS / "predictions_lstm.csv", index=False)

    # 3) Metrics, identical definitions to the baseline.
    last = tst.sort_values("cycle").groupby("unit").tail(1).merge(
        test_rul.rename(columns={"true_rul": "rul_from_file"}), on="unit")
    t_unc = last["rul_from_file"].to_numpy(float)
    t_cap = np.minimum(t_unc, CAP)
    p_last = last["predicted_rul"].to_numpy(float)
    v_true = rep["rul_capped"].to_numpy(float)
    oof_use = k_fit["cycle"].to_numpy() >= FIRST_PREDICTED_CYCLE
    oof_raw_lo, oof_raw_hi = oof_mu - Z80 * oof_sd, oof_mu + Z80 * oof_sd

    metrics = {
        "model_version": MODEL_VERSION,
        "model_name": "LSTM sequence model (2 x 64 units), 5-fold ensemble",
        "explainer": "occlusion sensitivity",
        "dataset": "NASA C-MAPSS FD001 (simulated engine data)",
        "split": "80 engines for 5-fold training and calibration; 20 held-out replay engines; 100 official test engines",
        "label_cap_cycles": CAP,
        "band": "80% band: Monte Carlo dropout spread calibrated with per-RUL-range conformal prediction on out-of-fold errors",
        "test_last_cycle": {
            "n_engines": int(len(last)),
            "rmse_vs_capped_true": round(rmse(t_cap, p_last), 2),
            "rmse_vs_uncapped_true": round(rmse(t_unc, p_last), 2),
            "nasa_score_vs_capped_true": round(nasa_score(t_cap, p_last), 1),
            "nasa_score_vs_uncapped_true": round(nasa_score(t_unc, p_last), 1),
            "band_coverage_vs_capped_true": round(coverage(last["rul_lower"], last["rul_upper"], t_cap), 3),
            "mean_band_width_cycles": round(float(np.mean(last["rul_upper"] - last["rul_lower"])), 1),
        },
        "replay_engines_all_cycles": {
            "n_engines": int(rep["unit"].nunique()),
            "n_rows": int(len(rep)),
            "rmse_vs_capped_true": round(rmse(v_true, rep["predicted_rul"]), 2),
            "band_coverage_vs_capped_true": round(coverage(rep["rul_lower"], rep["rul_upper"], v_true), 3),
            "mean_band_width_cycles": round(float(np.mean(rep["rul_upper"] - rep["rul_lower"])), 1),
            "coverage_by_range": coverage_by_range(rep["predicted_rul"], rep["rul_lower"], rep["rul_upper"], v_true),
        },
        "raw_mc_dropout_band": {
            "description": "Gaussian 80% band from the MC-dropout spread alone (mean +/- 1.28 x spread), before calibration",
            "oof_coverage": round(coverage(oof_raw_lo[oof_use], oof_raw_hi[oof_use], y_fit[oof_use]), 3),
            "replay_coverage": round(coverage(rep["raw_lower"], rep["raw_upper"], v_true), 3),
            "replay_mean_width_cycles": round(float(np.mean(rep["raw_upper"] - rep["raw_lower"])), 1),
            "replay_coverage_by_range": coverage_by_range(rep["predicted_rul"], rep["raw_lower"], rep["raw_upper"], v_true),
        },
        "cross_validation": {
            "folds": N_FOLDS,
            "oof_rmse_vs_capped_true": round(rmse(y_fit[oof_use], oof_mu[oof_use]), 2),
            "oof_rmse_deterministic": round(rmse(y_fit[oof_use], oof_det[oof_use]), 2),
            "fold_details": folds_info,
        },
        "sensor_importance": sensor_shares(occ_rep, SENSORS),
        "training": {
            "window_cycles": WINDOW, "hidden_units": HIDDEN, "dense_units": DENSE, "dropout": DROPOUT,
            "batch": BATCH, "learning_rate": LR, "max_epochs": MAX_EPOCHS, "patience": PATIENCE,
            "mc_passes_per_model": MC_PASSES, "torch": torch.__version__,
            "total_seconds": round(time.time() - t_start, 1),
        },
        "replay_units": [int(u) for u in replay],
    }
    (OUTPUTS / "metrics_lstm.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps({k: metrics[k] for k in ("test_last_cycle", "raw_mc_dropout_band")}, indent=2))
    rp = {k: v for k, v in metrics["replay_engines_all_cycles"].items() if k != "coverage_by_range"}
    print(json.dumps(rp, indent=2))
    print(f"Wrote {len(preds)} rows to outputs/predictions_lstm.csv in {time.time() - t_start:.0f}s")


if __name__ == "__main__":
    main()
