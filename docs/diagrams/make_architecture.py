"""Draw the system architecture diagram for the README and the deck.

Writes docs/images/architecture.svg, architecture.png (with title, 3200 x 1800) and
architecture_slide.png (no title, for a slide that has its own heading).
Run from the repo root:  python docs/diagrams/make_architecture.py
Needs Pillow (text measurement) and Playwright with Chromium (PNG export); the Inter font is used when installed.
"""
from __future__ import annotations

import html
from pathlib import Path

from PIL import ImageFont

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "docs" / "images"
FONT_DIR = Path("/usr/share/fonts/opentype/inter")
FONT_FILES = {400: "Inter-Regular.otf", 500: "Inter-Medium.otf", 600: "Inter-SemiBold.otf", 700: "Inter-Bold.otf"}
FAMILY = "Inter, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif"

W, H = 1600, 900
INK, INK2, MUTED, LINE, BORDER = "#0f172a", "#334155", "#64748b", "#94a3b8", "#e2e8f0"
BLUE, BLUE_D, BLUE_T = "#2a78d6", "#1c5cab", "#eef4fc"
VIOLET, VIOLET_D, VIOLET_T = "#6250d6", "#4a3aa7", "#f1effc"
AQUA, AQUA_D, AQUA_T = "#1baf7a", "#13795a", "#e9f7f1"
ORANGE, ORANGE_D, ORANGE_T = "#eb6834", "#b8461a", "#fdf0ea"
RED, RED_T, RED_TXT = "#d03b3b", "#fdeceb", "#9f1c14"
SLATE_T = "#f8fafc"

ICONS = {  # Material Design icons (Apache 2.0), 24 x 24
    "plane": "M21 16v-2l-8-5V3.5c0-.83-.67-1.5-1.5-1.5S10 2.67 10 3.5V9l-8 5v2l8-2.5V19l-2 1.5V22l3.5-1 3.5 1v-1.5L13 19v-5.5l8 2.5z",
    "chip": "M15 9H9v6h6V9zm-2 4h-2v-2h2v2zm8-2V9h-2V7c0-1.1-.9-2-2-2h-2V3h-2v2h-2V3H9v2H7c-1.1 0-2 .9-2 2v2H3v2h2v2H3v2h2v2c0 1.1.9 2 2 2h2v2h2v-2h2v2h2v-2h2c1.1 0 2-.9 2-2v-2h2v-2h-2v-2h2zm-4 6H7V7h10v10z",
    "tune": "M3 17v2h6v-2H3zM3 5v2h10V5H3zm10 16v-2h8v-2h-8v-2h-2v6h2zM7 9v2H3v2h4v2h2V9H7zm14 4v-2H11v2h10zm-6-4h2V7h4V5h-4V3h-2v6z",
    "dashboard": "M3 13h8V3H3v10zm0 8h8v-6H3v6zm10 0h8V11h-8v10zm0-18v6h8V3h-8z",
    "person": "M12 12c2.21 0 4-1.79 4-4s-1.79-4-4-4-4 1.79-4 4 1.79 4 4 4zm0 2c-2.67 0-8 1.34-8 4v2h16v-2c0-2.66-5.33-4-8-4z",
}

_fonts: dict[tuple[int, float], ImageFont.FreeTypeFont] = {}


def text_w(s: str, size: float, weight: int = 400) -> float:
    key = (weight, size)
    if key not in _fonts:
        path = FONT_DIR / FONT_FILES[weight]
        _fonts[key] = ImageFont.truetype(str(path), size=int(round(size * 4))) if path.exists() else None
    f = _fonts[key]
    return f.getlength(s) / 4 if f else len(s) * size * 0.55


def wrap(s: str, size: float, max_w: float, weight: int = 400) -> list[str]:
    lines, cur = [], ""
    for word in s.split():
        trial = f"{cur} {word}".strip()
        if text_w(trial, size, weight) <= max_w or not cur:
            cur = trial
        else:
            lines.append(cur)
            cur = word
    return lines + ([cur] if cur else [])


def t(x, y, s, size=12, weight=400, fill=INK2, anchor="start", extra=""):
    return (f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" font-weight="{weight}" fill="{fill}" '
            f'text-anchor="{anchor}" {extra}>{html.escape(s)}</text>')


def halo(x, y, s, size=11.5, fill=MUTED, anchor="start", weight=500):
    return t(x, y, s, size, weight, fill, anchor, 'stroke="#ffffff" stroke-width="5" paint-order="stroke" stroke-linejoin="round"')


def chip(x_right, y, label, fill, color=INK, size=11, weight=600, h=22):
    w = text_w(label, size, weight) + 18
    x = x_right - w
    return (f'<rect x="{x:.1f}" y="{y}" width="{w:.1f}" height="{h}" rx="{h / 2}" fill="{fill}"/>'
            + t(x + w / 2, y + h / 2 + size * 0.36, label, size, weight, color, "middle")), w


def icon(cx, cy, name, bg, r=15):
    s = (r * 1.25) / 24
    return (f'<circle cx="{cx}" cy="{cy}" r="{r}" fill="{bg}"/>'
            f'<g transform="translate({cx - 12 * s:.2f},{cy - 12 * s:.2f}) scale({s:.4f})"><path d="{ICONS[name]}" fill="#ffffff"/></g>')


def badge(cx, cy, n, color):
    return (f'<circle cx="{cx}" cy="{cy}" r="13" fill="{color}" stroke="#ffffff" stroke-width="3"/>'
            + t(cx, cy + 4.6, str(n), 13, 700, "#ffffff", "middle"))


def card(x, y, w, title, body, accent, chip_spec=None, primary=False, dashed=False, title_size=14.5):
    """A white card with an accent bar, a title, an optional chip and wrapped body text. Returns (svg, height)."""
    body_lines = wrap(body, 12, w - 40)
    h = 32 + 16 * len(body_lines) + 12
    stroke = accent if primary else BORDER
    sw = 1.6 if primary else 1
    dash = ' stroke-dasharray="5 4"' if dashed else ""
    fill = "#ffffff"
    out = [f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="10" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"{dash}/>',
           f'<rect x="{x + 8}" y="{y + 12}" width="3.5" height="{h - 24}" rx="1.75" fill="{accent}" opacity="{1 if not dashed else 0.45}"/>',
           t(x + 20, y + 24, title, title_size, 600, INK if not dashed else INK2)]
    if chip_spec:
        label, cfill, ccolor = chip_spec
        c, _ = chip(x + w - 10, y + 9, label, cfill, ccolor)
        out.append(c)
    for i, line in enumerate(body_lines):
        out.append(t(x + 20, y + 44 + 16 * i, line, 12, 400, INK2 if not dashed else MUTED))
    return "".join(out), h


def container(x, y, w, h, tint, edge):
    return f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="16" fill="{tint}" stroke="{edge}" stroke-opacity="0.45" stroke-width="1.4"/>'


def header(x, y, w, n, step_color, icon_name, icon_bg, title, chip_label=None, chip_fill="#ffffff"):
    out = [icon(x + 32, y + 30, icon_name, icon_bg), t(x + 56, y + 36, title, 17, 600, INK)]
    if chip_label:
        c, _ = chip(x + w - 16, y + 19, chip_label, chip_fill, INK2, 11, 600)
        out.append(c)
    out.append(badge(x, y, n, step_color))
    return "".join(out)


def pill(cx, y, label, fill):
    w = text_w(label, 12, 600) + 28
    return (f'<rect x="{cx - w / 2:.1f}" y="{y}" width="{w:.1f}" height="30" rx="15" fill="{fill}"/>'
            + t(cx, y + 19.5, label, 12, 600, "#ffffff", "middle"))


def arrow(d, color=INK2, dashed=False, marker="arrow-dark", width=1.8):
    dash = ' stroke-dasharray="5 4"' if dashed else ""
    return f'<path d="{d}" fill="none" stroke="{color}" stroke-width="{width}"{dash} marker-end="url(#{marker})" stroke-linecap="round" stroke-linejoin="round"/>'


def cylinder(x, y, w, h, title, sub_lines, color=MUTED):
    e = 9
    body = (f'<path d="M{x},{y + e} a{w / 2},{e} 0 0 1 {w},0 v{h - 2 * e} a{w / 2},{e} 0 0 1 {-w},0 z" '
            f'fill="#ffffff" stroke="{LINE}" stroke-width="1.2"/>'
            f'<ellipse cx="{x + w / 2}" cy="{y + e}" rx="{w / 2}" ry="{e}" fill="{SLATE_T}" stroke="{LINE}" stroke-width="1.2"/>')
    out = [body, t(x + w / 2, y + e + 30, title, 13.5, 600, INK, "middle")]
    for i, s in enumerate(sub_lines):
        out.append(t(x + w / 2, y + e + 48 + 15 * i, s, 11.5, 400, MUTED, "middle"))
    return "".join(out)


def build() -> str:
    s: list[str] = []
    s.append(f'<rect x="0" y="0" width="{W}" height="{H}" fill="#ffffff"/>')

    # Title (cropped away in the slide version)
    s.append(t(40, 56, "From landing to a reviewed parts plan", 28, 700, INK))
    s.append(t(40, 88, "One pass after every landing: sensor history → wear estimate with uncertainty → parts plan → a person decides.",
               15, 400, MUTED))

    TOP, BOT = 140, 600
    MID = (TOP + BOT) // 2
    # ---- 1 Landing event --------------------------------------------------------------
    ax, aw, ah = 40, 190, 128
    ay = MID - ah / 2
    s.append(f'<rect x="{ax}" y="{ay}" width="{aw}" height="{ah}" rx="12" fill="#ffffff" stroke="{BORDER}" stroke-width="1.2"/>')
    s.append(icon(ax + 30, ay + 32, "plane", INK, 16))
    s.append(t(ax + 54, ay + 37.5, "Landing event", 15, 600, INK))
    for i, line in enumerate(wrap("A flight completes and new sensor readings arrive for both engines.", 12, aw - 32)):
        s.append(t(ax + 16, ay + 70 + 16 * i, line, 12, 400, INK2))
    s.append(badge(ax, ay, 1, INK))

    # ---- 2 Wear estimation ------------------------------------------------------------
    bx, bw = 280, 420
    s.append(container(bx, TOP, bw, BOT - TOP, BLUE_T, BLUE))
    s.append(header(bx, TOP, bw, 2, BLUE_D, "chip", BLUE, "Wear estimation · AI", "PyTorch · LightGBM"))
    cx, cw, y = bx + 16, bw - 32, TOP + 56
    for title, body, chip_spec, primary, dashed in [
        ("Sensor processing", "Drop 7 flat sensors · standardise · 30-flight windows", None, False, False),
        ("LSTM ensemble · in use", "Last 30 flights × 14 sensors → 2 LSTM layers (64 units) → remaining useful life; 5 fold models averaged",
         ("RMSE 13.6 cycles", BLUE_T, BLUE_D), True, False),
        ("Calibrated 80% band", "Monte Carlo dropout spread + conformal calibration", ("85% coverage", BLUE_T, BLUE_D), False, False),
        ("Explainability", "Occlusion sensitivity → top 3 sensors for every estimate", None, False, False),
        ("LightGBM baseline", "Compared under a selection rule fixed in advance", ("RMSE 18.4", "#f1f5f9", MUTED), False, True),
    ]:
        svg, h = card(cx, y, cw, title, body, BLUE, chip_spec, primary, dashed)
        s.append(svg)
        y += h + 8
    s.append(pill(bx + bw / 2, BOT - 42, "Output: RUL estimate · 80% band · top sensors", BLUE_D))

    # ---- 3 Planning & optimization ----------------------------------------------------
    px, pw = 750, 400
    s.append(container(px, TOP, pw, BOT - TOP, VIOLET_T, VIOLET))
    s.append(header(px, TOP, pw, 3, VIOLET_D, "tune", VIOLET, "Planning & optimization", "PuLP · HiGHS"))
    cx, cw, y = px + 16, pw - 32, TOP + 56
    for title, body, chip_spec in [
        ("Urgency tier", "Acts on the band's lower edge: Review ≤ 10 · Plan ≤ 40 · otherwise Monitor", None),
        ("Need-by time", "Landing time + (lower edge − 5-cycle buffer) ÷ the aircraft's flights per day", None),
        ("Option ranking", "Reserve stock · transfer · standard or expedited order; on-time first, then cost vs kg CO2e",
         None),
        ("Fleet optimizer", "Shares scarce spares across engines so no spare is promised twice", ("MILP", VIOLET_T, VIOLET_D)),
    ]:
        svg, h = card(cx, y, cw, title, body, VIOLET, chip_spec)
        s.append(svg)
        y += h + 8
    s.append(pill(px + pw / 2, BOT - 42, "Output: ranked plan + plain-language rationale", VIOLET_D))

    # ---- 4 Dashboard & alert -----------------------------------------------------------
    dx, dw, dy, dh = 1200, 360, TOP, 262
    s.append(f'<rect x="{dx}" y="{dy}" width="{dw}" height="{dh}" rx="16" fill="{AQUA_T}" stroke="{AQUA}" stroke-opacity="0.5" stroke-width="1.4"/>')
    s.append(header(dx, dy, dw, 4, AQUA_D, "dashboard", AQUA_D, "Dashboard & alert", "Streamlit"))
    s.append(t(dx + 16, dy + 72, "Five views: fleet · engine · landing · parts · impact", 12, 400, INK2))
    mx, my, mw, mh = dx + 16, dy + 86, dw - 32, 120
    s.append(f'<rect x="{mx}" y="{my}" width="{mw}" height="{mh}" rx="10" fill="#ffffff" stroke="{BORDER}"/>')
    s.append(f'<rect x="{mx + 8}" y="{my + 12}" width="3.5" height="{mh - 24}" rx="1.75" fill="{RED}"/>')
    s.append(t(mx + 20, my + 22, "EXAMPLE POST-LANDING ALERT", 10.5, 700, MUTED, extra='letter-spacing="0.6"'))
    s.append(t(mx + 20, my + 43, "AC-01 at Hyderabad · engine FD001-R-074", 13, 600, INK))
    pw_ = text_w("Review before next departure", 11, 600) + 18
    s.append(f'<rect x="{mx + 20}" y="{my + 52}" width="{pw_:.1f}" height="21" rx="10.5" fill="{RED_T}"/>')
    s.append(t(mx + 20 + pw_ / 2, my + 66.5, "Review before next departure", 11, 600, RED_TXT, "middle"))
    s.append(t(mx + 20, my + 92, "Part needed by 08:19 · nothing arrives in time", 12, 400, INK2))
    s.append(t(mx + 20, my + 108, "Fastest: air transfer from Nagpur → escalated", 12, 400, INK2))
    s.append(t(dx + 16, dy + dh - 22, "FastAPI exposes the same core over REST", 12, 400, MUTED))

    # ---- 5 Authorised planner ------------------------------------------------------------
    hx, hy, hw = 1200, 462, 360
    hh = BOT - hy
    s.append(f'<rect x="{hx}" y="{hy}" width="{hw}" height="{hh}" rx="16" fill="{ORANGE_T}" stroke="{ORANGE}" stroke-width="2"/>')
    s.append(icon(hx + 32, hy + 30, "person", ORANGE_D))
    s.append(t(hx + 56, hy + 36, "Authorised maintenance planner", 15.5, 600, INK))
    for i, line in enumerate(wrap("Approves or overrides each recommendation; an override needs a note. Records the maintenance "
                                  "once the part arrives. The system never orders on its own.", 12, hw - 32)):
        s.append(t(hx + 16, hy + 66 + 16 * i, line, 12, 400, INK2))
    s.append(badge(hx, hy, 5, ORANGE_D))

    # ---- Main flow arrows ------------------------------------------------------------------
    s.append(arrow(f"M{ax + aw + 2},{MID} H{bx - 4}"))
    s.append(arrow(f"M{bx + bw + 2},{MID} H{px - 4}"))
    s.append(arrow(f"M{px + pw + 2},{MID} H{dx - 4}"))
    s.append(arrow(f"M{dx + dw / 2},{dy + dh + 2} V{hy - 4}"))
    s.append(halo(dx + dw / 2 + 10, (dy + dh + hy) / 2 + 4, "alert + ranked options", 11.5, MUTED))

    # ---- Data layer -------------------------------------------------------------------------
    DY0, DY1 = 672, 804
    s.append(f'<rect x="40" y="{DY0}" width="{W - 80}" height="{DY1 - DY0}" rx="16" fill="{SLATE_T}" stroke="{BORDER}" stroke-width="1.2"/>')
    s.append(t(56, DY0 + 30, "Data layer", 15, 600, INK))
    for i, line in enumerate(wrap("Simulated engine data, synthetic supply chain, cited emission factors and an audit log", 12, 166)):
        s.append(t(56, DY0 + 52 + 16 * i, line, 12, 400, MUTED))
    cy0, ch = DY0 + 20, 92
    stores = {
        "cmapss": (330, 320, "NASA C-MAPSS FD001", ["Simulated run-to-failure", "sensor histories"]),
        "factors": (766, 170, "Emission factors", ["kg CO2e per tonne-km", "NZ MfE 2026 (cited)"]),
        "supply": (952, 182, "Synthetic supply chain", ["Stock · lanes · suppliers", "fleet schedule"]),
        "log": (1340, 220, "Decision log", ["Approvals, overrides,", "completed maintenance"]),
    }
    for x, w, title, sub in stores.values():
        s.append(cylinder(x, cy0, w, ch, title, sub))

    def mid(k):
        x, w = stores[k][0], stores[k][1]
        return x + w / 2

    label_y = BOT + 36
    s.append(arrow(f"M{mid('cmapss')},{cy0 - 2} V{BOT + 4}", LINE, True, "arrow-grey", 1.6))
    s.append(halo(mid("cmapss") - 10, label_y, "sensor histories", anchor="end"))
    s.append(arrow(f"M{mid('factors')},{cy0 - 2} V{BOT + 4}", LINE, True, "arrow-grey", 1.6))
    s.append(halo(mid("factors") - 10, label_y, "kg CO2e / t-km", anchor="end"))
    s.append(arrow(f"M{mid('supply')},{cy0 - 2} V{BOT + 4}", LINE, True, "arrow-grey", 1.6))
    s.append(halo(mid("supply") - 10, label_y, "stock · lead times", anchor="end"))
    s.append(arrow(f"M{mid('log')},{BOT + 2} V{cy0 - 4}", LINE, True, "arrow-grey", 1.6))
    s.append(halo(mid("log") - 10, label_y, "every decision", anchor="end"))
    # Feedback loop: approved reservations change the stock the next plan sees.
    fx, enter_x, run_y = 1262, 1110, BOT + 52
    s.append(arrow(f"M{fx},{BOT + 2} V{run_y} H{enter_x} V{cy0 - 4}", ORANGE, True, "arrow-orange", 1.8))
    s.append(halo(fx - 10, run_y - 9, "approved reservations update stock", 11, ORANGE_D, "end", 600))

    # ---- Footer: tech stack + responsibility note ----------------------------------------------
    fy = DY1 + 30
    s.append(t(40, fy, "BUILT WITH", 10.5, 700, MUTED, extra='letter-spacing="0.8"'))
    x = 40
    for label in ["Python", "PyTorch", "LightGBM", "scikit-learn", "PuLP + HiGHS", "Streamlit", "Plotly", "FastAPI", "pytest"]:
        w = text_w(label, 12, 500) + 22
        s.append(f'<rect x="{x:.1f}" y="{fy + 12}" width="{w:.1f}" height="26" rx="13" fill="#f1f5f9" stroke="{BORDER}"/>')
        s.append(t(x + w / 2, fy + 29.5, label, 12, 500, INK2, "middle"))
        x += w + 8
    s.append(t(W - 40, fy + 16, "Decision support only", 13, 600, INK, "end"))
    s.append(t(W - 40, fy + 35, "Engine data: NASA C-MAPSS simulation · supply chain: synthetic", 12, 400, MUTED, "end"))

    defs = "".join(
        f'<marker id="{mid_}" viewBox="0 0 10 10" refX="8.5" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">'
        f'<path d="M0,0 L10,5 L0,10 z" fill="{col}"/></marker>'
        for mid_, col in (("arrow-dark", INK2), ("arrow-grey", LINE), ("arrow-orange", ORANGE))
    )
    return f"<defs>{defs}</defs>" + "".join(s)


def svg_doc(body: str, viewbox: tuple[int, int, int, int]) -> str:
    x, y, w, h = viewbox
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="{x} {y} {w} {h}" width="{w}" height="{h}" '
            f'font-family="{FAMILY}" role="img" aria-label="System architecture: landing event, AI wear estimation, '
            f'planning and optimization, dashboard and alert, authorised planner, data layer">{body}</svg>')


def render_png(svg: str, path: Path, w: int, h: int) -> None:
    from playwright.sync_api import sync_playwright

    faces = "".join(
        f"@font-face{{font-family:Inter;src:url('file://{FONT_DIR / f}');font-weight:{wt};}}" for wt, f in FONT_FILES.items()
    )
    page_html = f"<html><head><style>{faces}html,body{{margin:0;background:#fff}}</style></head><body>{svg}</body></html>"
    tmp = path.with_suffix(".render.html")
    tmp.write_text(page_html)
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch()
            page = browser.new_page(viewport={"width": w, "height": h}, device_scale_factor=2)
            page.goto(tmp.as_uri())
            page.wait_for_timeout(400)
            page.locator("svg").screenshot(path=str(path))
            browser.close()
    finally:
        tmp.unlink(missing_ok=True)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    body = build()
    full = svg_doc(body, (0, 0, W, H))
    slide = svg_doc(body, (0, 112, W, 772))
    (OUT / "architecture.svg").write_text(full)
    render_png(full, OUT / "architecture.png", W, H)
    render_png(slide, OUT / "architecture_slide.png", W, 772)
    print("wrote docs/images/architecture.svg, architecture.png, architecture_slide.png")


if __name__ == "__main__":
    main()
