"""Deck and README charts from the model outputs (static PNGs in docs/images/).

Run from the repo root after ml.select_model:  python -m ml.make_charts
Colours follow the entity in every chart: LSTM = blue, LightGBM = orange, raw MC-dropout band = aqua.
"""
from __future__ import annotations

import json
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import yaml  # noqa: E402
from matplotlib import font_manager  # noqa: E402

from ml.data_prep import CAP, SENSORS, load_fd001  # noqa: E402
from ml.metrics import bin_labels  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
IMG = ROOT / "docs" / "images"

SURFACE, INK, INK2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
GRID, AXIS = "#e1e0d9", "#c3c2b7"
COLOR = {"lstm": "#2a78d6", "lgbm": "#eb6834", "raw": "#1baf7a"}
CRITICAL, WARNING = "#d03b3b", "#c98500"
LABEL = {"lstm": "LSTM", "lgbm": "LightGBM"}
SHORT = {
    "s2": "T24 · LPC outlet temp", "s3": "T30 · HPC outlet temp", "s4": "T50 · LPT outlet temp",
    "s7": "P30 · HPC outlet pressure", "s8": "Nf · fan speed", "s9": "Nc · core speed",
    "s11": "Ps30 · HPC static pressure", "s12": "phi · fuel flow / Ps30", "s13": "NRf · corrected fan speed",
    "s14": "NRc · corrected core speed", "s15": "BPR · bypass ratio", "s17": "htBleed · bleed enthalpy",
    "s20": "W31 · HPT coolant bleed", "s21": "W32 · LPT coolant bleed",
}
DEMO_ENGINE = "FD001-R-074"  # AC-01's engine in the dashboard scenario


def setup_fonts() -> None:
    names = {f.name for f in font_manager.fontManager.ttflist}
    plt.rcParams["font.family"] = "Inter" if "Inter" in names else "DejaVu Sans"
    plt.rcParams.update({"axes.titlesize": 12, "font.size": 10.5, "savefig.facecolor": SURFACE})


def new_fig(title: str, subtitle: str, size=(10, 5.6), top: float = 0.80):
    fig, ax = plt.subplots(figsize=size, dpi=200)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    lines = textwrap.wrap(subtitle, width=128)
    step = 0.22 / size[1]
    fig.subplots_adjust(left=0.08, right=0.97, top=top - step * (len(lines) - 1), bottom=0.12)
    fig.text(0.08, 1 - 0.40 / size[1], title, fontsize=15, fontweight=600, color=INK, ha="left")
    for i, line in enumerate(lines):
        fig.text(0.08, 1 - 0.70 / size[1] - i * step, line, fontsize=10, color=INK2, ha="left")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
    ax.tick_params(colors=MUTED, labelcolor=INK2, length=0)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    return fig, ax


def save(fig, name: str) -> None:
    IMG.mkdir(parents=True, exist_ok=True)
    fig.savefig(IMG / name, facecolor=SURFACE)
    plt.close(fig)
    print("wrote docs/images/" + name)


def load():
    comparison = json.loads((OUT / "model_comparison.json").read_text())
    preds = {k: pd.read_csv(OUT / f"predictions_{k}.csv") for k in ("lstm", "lgbm") if (OUT / f"predictions_{k}.csv").exists()}
    metrics = {k: json.loads((OUT / f"metrics_{k}.json").read_text()) for k in preds}
    cfg = yaml.safe_load((ROOT / "config.yaml").read_text())
    return comparison, preds, metrics, cfg


def chart_test_engines(primary: str, preds: dict, metrics: dict) -> None:
    _, _, test_rul = load_fd001()
    p = preds[primary]
    last = p[p["engine_id"].str.contains("-T-")].sort_values("cycle").groupby("engine_id").tail(1).copy()
    last["unit"] = last["engine_id"].str[-3:].astype(int)
    df = last.merge(test_rul.rename(columns={"true_rul": "rul_file"}), on="unit")
    df["true_cap"] = df["rul_file"].clip(upper=CAP)
    df = df.sort_values(["true_cap", "predicted_rul"]).reset_index(drop=True)
    x = np.arange(1, len(df) + 1)
    t = metrics[primary]["test_last_cycle"]
    fig, ax = new_fig(
        f"{LABEL[primary]} estimates track true remaining life on 100 unseen test engines",
        f"Official FD001 test engines, sorted by true remaining life (capped at {CAP} cycles). "
        f"RMSE {t['rmse_vs_capped_true']:.1f} cycles; {t['band_coverage_vs_capped_true']:.0%} of true values fall inside the 80% band.",
    )
    c = COLOR[primary]
    ax.vlines(x, df["rul_lower"], df["rul_upper"], color=c, alpha=0.28, linewidth=2.4, label="80% band")
    ax.plot(x, df["true_cap"], color=INK, linewidth=2, label="True remaining life", zorder=2)
    ax.scatter(x, df["predicted_rul"], s=30, color=c, edgecolor=SURFACE, linewidth=1.4, zorder=3, label=f"{LABEL[primary]} estimate")
    ax.set_xlim(0, len(df) + 1)
    ax.set_ylim(0, max(150, float(df["rul_upper"].max()) + 5))
    ax.set_xlabel("Test engines, sorted by true remaining life", color=INK2)
    ax.set_ylabel("Remaining useful life (cycles = flights)", color=INK2)
    ax.legend(loc="upper left", frameon=False, labelcolor=INK2)
    save(fig, "model_test_engines.png")


def chart_demo_engine(primary: str, preds: dict, cfg: dict) -> dict:
    p = preds[primary]
    g = p[p["engine_id"] == DEMO_ENGINE].sort_values("cycle")
    plan_th, review_th = cfg["planning"]["plan_max_rul_lower"], cfg["planning"]["review_max_rul_lower"]
    first_plan = g[g["rul_lower"] <= plan_th].iloc[0]
    first_review = g[g["rul_lower"] <= review_th].iloc[0]
    end = int(g["cycle"].max())
    fleet = pd.read_csv(ROOT / "data" / "synthetic" / "fleet.csv")
    cpd = float(fleet.loc[fleet["engine_id"] == DEMO_ENGINE, "cycles_per_day"].iloc[0])
    fig, ax = new_fig(
        f"Alerts arrive {first_plan['true_rul']:.0f} and {first_review['true_rul']:.0f} flights before the end of simulated life",
        f"Held-out engine {DEMO_ENGINE} (AC-01 in the dashboard), {LABEL[primary]} estimate after every flight. "
        f"At AC-01's {cpd:g} flights a day that is about {first_plan['true_rul'] / cpd:.0f} and "
        f"{first_review['true_rul'] / cpd:.0f} days of notice. The planner acts on the lower edge of the band.",
    )
    c = COLOR[primary]
    ax.fill_between(g["cycle"], g["rul_lower"], g["rul_upper"], color=c, alpha=0.13, linewidth=0, label="80% band")
    ax.plot(g["cycle"], g["true_rul"], color=INK, linewidth=2, label="True remaining life (known only in simulation)")
    ax.plot(g["cycle"], g["predicted_rul"], color=c, linewidth=2, label=f"{LABEL[primary]} estimate")
    for th, txt, col in ((plan_th, f"Plan maintenance: lower edge ≤ {plan_th}", WARNING), (review_th, f"Review before next departure: ≤ {review_th}", CRITICAL)):
        ax.axhline(th, color=col, linewidth=1, alpha=0.9)
        ax.text(g["cycle"].min() + 1, th + 2.5, txt, color=INK2, fontsize=9.5)
    start = float(g["cycle"].min())
    for row, col, txt, pos in (
        (first_plan, WARNING, "Plan alert", (start + 0.23 * (end - start), 30)),
        (first_review, CRITICAL, "Review alert", (start + 0.37 * (end - start), 21)),
    ):
        ax.scatter([row["cycle"]], [row["rul_lower"]], s=70, color=col, edgecolor=SURFACE, linewidth=2, zorder=4)
        ax.annotate(
            f"{txt} after flight {int(row['cycle'])}: {row['true_rul']:.0f} flights left",
            xy=(row["cycle"], row["rul_lower"]), xytext=pos, ha="left", va="center",
            color=INK, fontsize=9.5, arrowprops=dict(arrowstyle="-", color=MUTED, linewidth=1, shrinkA=4, shrinkB=6),
        )
    ax.set_xlim(g["cycle"].min() - 1, end + 2)
    ax.set_ylim(0, max(150, float(g["true_rul"].max()) + 5))
    ax.set_xlabel("Flight cycle", color=INK2)
    ax.set_ylabel("Remaining useful life (cycles = flights)", color=INK2)
    ax.legend(loc="upper right", frameon=False, labelcolor=INK2)
    save(fig, "demo_engine_band.png")
    return {"engine": DEMO_ENGINE, "plan_alert_cycle": int(first_plan["cycle"]), "plan_alert_flights_left": float(first_plan["true_rul"]),
            "review_alert_cycle": int(first_review["cycle"]), "review_alert_flights_left": float(first_review["true_rul"]), "end_of_life_cycle": end}


def chart_sensor_importance(primary: str, metrics: dict) -> dict:
    keys = [primary] + [k for k in metrics if k != primary]
    imp = {k: metrics[k]["sensor_importance"] for k in keys}
    order = sorted(SENSORS, key=lambda s: imp[primary][s])  # ascending -> largest at the top
    tops = {k: set(sorted(SENSORS, key=lambda s: -imp[k][s])[:5]) for k in keys}
    shared = set.intersection(*tops.values())
    core_speed = {"s9", "s14"}  # Nc and NRc: physical and corrected speed of the same core shaft
    if {"s4", "s11"} <= shared and all(t & core_speed for t in tops.values()):
        title = "Both models rank core speed, T50 and Ps30 among their top five sensors"
    elif len(shared) >= 2:
        title = "Both models lean on several of the same sensors"
    else:
        title = "The models weigh the sensors differently"
    fig, ax = new_fig(
        title,
        "Share of each model's attribution on the 20 held-out engines ("
        + "; ".join(f"{LABEL[k]}: {metrics[k]['explainer']}" for k in keys)
        + "). Nc and NRc measure the same shaft.",
        size=(10, 6.4),
    )
    fig.subplots_adjust(left=0.30, top=0.83)
    y = np.arange(len(order))
    h = 0.38
    for i, k in enumerate(keys):
        vals = np.array([imp[k][s] * 100 for s in order])
        offset = 0.2 if i == 0 else -0.2
        ax.barh(y + offset, vals, height=h, color=COLOR[k], label=LABEL[k])
        for j in np.argsort(-vals)[:3]:  # label each model's own top three
            ax.text(vals[j] + 0.25, y[j] + offset, f"{vals[j]:.0f}%", va="center", fontsize=9, color=INK2)
    ax.set_yticks(y, [SHORT[s] for s in order], color=INK2)
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    top_value = max(imp[k][s] for k in keys for s in SENSORS) * 100
    ax.set_xlim(0, top_value * 1.12)
    ax.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(5))
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:.0f}%"))
    ax.set_xlabel("Share of attribution", color=INK2)
    ax.legend(loc="lower right", frameon=False, labelcolor=INK2)
    save(fig, "sensor_importance.png")
    return {"shared_top3": sorted(shared), "top3": {k: sorted(v) for k, v in tops.items()}}


def chart_calibration(metrics: dict) -> None:
    labels = bin_labels()
    series = []
    if "lstm" in metrics:
        raw = metrics["lstm"]["raw_mc_dropout_band"]
        series.append(("raw", f"LSTM, raw MC-dropout band ({raw['replay_coverage']:.0%} overall)", raw["replay_coverage_by_range"]))
        cal = metrics["lstm"]["replay_engines_all_cycles"]
        series.append(("lstm", f"LSTM, calibrated band ({cal['band_coverage_vs_capped_true']:.0%})", cal["coverage_by_range"]))
    if "lgbm" in metrics:
        lg = metrics["lgbm"]["replay_engines_all_cycles"]
        series.append(("lgbm", f"LightGBM, cross-validated band ({lg['band_coverage_vs_capped_true']:.0%})", lg["coverage_by_range"]))
    fig, ax = new_fig(
        "Calibration turns an over-confident band into an honest one",
        "Share of true remaining life inside each 80% band, by predicted-RUL range, on the 20 held-out engines "
        "(never used for training or calibration).",
        top=0.72,
    )
    fig.subplots_adjust(right=0.88)
    x = np.arange(len(labels))
    w = 0.26
    for i, (key, name, cov) in enumerate(series):
        vals = [100 * (cov[l]["coverage"] or 0) for l in labels]
        ax.bar(x + (i - (len(series) - 1) / 2) * (w + 0.02), vals, width=w, color=COLOR[key], label=name)
    ax.axhline(80, color=INK, linewidth=1.2)
    ax.text(1.01, 80, "Target 80%", transform=ax.get_yaxis_transform(), va="center", ha="left", color=INK, fontsize=9.5)
    ax.set_xticks(x, labels)
    ax.set_ylim(0, 100)
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:.0f}%"))
    ax.set_xlabel("Predicted remaining life (cycles)", color=INK2)
    ax.set_ylabel("True values inside the band", color=INK2)
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.01), ncol=3, frameon=False, labelcolor=INK2, fontsize=9.5,
              handlelength=1.2, columnspacing=1.4)
    save(fig, "band_calibration.png")


def main() -> None:
    setup_fonts()
    comparison, preds, metrics, cfg = load()
    primary = comparison["primary"]
    chart_test_engines(primary, preds, metrics)
    demo = chart_demo_engine(primary, preds, cfg)
    sens = chart_sensor_importance(primary, metrics)
    chart_calibration(metrics)
    facts = {"primary": primary, "demo_engine": demo, "sensors": sens}
    (OUT / "chart_facts.json").write_text(json.dumps(facts, indent=2))
    print(json.dumps(facts, indent=2))


if __name__ == "__main__":
    main()
