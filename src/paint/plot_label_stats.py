"""Draw the PAINT label statistics as a two-panel grouped bar chart (SVG).

Left panel: WorldMedQA-Pain, computed from data/WorldMedQA-Pain/<Partition>.csv.
Right panel: BRIDGE-Pain, taken from Table 6 of the paper, because the
BRIDGE-Pain labels are not redistributed in this repository.

Every bar is a share of the items where both pain fields are resolved (the
three models agree on pain relevance and pain presence): the pain-presence
rate and the coverage of the four demographic fields.

Writes assets/label_stats.svg (light) and assets/label_stats_dark.svg (dark);
the README embeds both with a <picture> element. Opened on its own, the SVG
shows each bar's value on hover (CSS only, no script). Standard library only.

Run:
    python3 src/paint/plot_label_stats.py
"""

from __future__ import annotations

import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = ROOT / "data" / "WorldMedQA-Pain"
OUT_DIR = ROOT / "assets"

METRICS = ["Pain presence", "Age", "Gender", "Ethnicity", "Occupation"]
ABSTAIN = "abstain"

# (partition, language, csv stem) for the computed panel
WMQA = [("UK PLAB", "en", "UK"), ("Spain MIR", "es", "Spain"), ("Japan NLE", "ja", "Japan"), ("Brazil REVALIDA", "pt", "Brazil")]

# Table 6 of the paper: items, both pain fields resolved, pain-presence-positive
# (within resolved), then coverage of age, gender, ethnicity, occupation (within
# resolved), all in percent.
BRIDGE = [
    ("MEDIQA", "en", 1501, 1402, [20.4, 14.1, 40.2, 3.0, 7.1]),
    ("MIMIC-III", "en", 5000, 3993, [63.7, 87.0, 100.0, 89.6, 41.5]),
    ("Cantemist", "es", 1294, 1091, [67.8, 95.6, 98.6, 1.9, 14.9]),
    ("IFMIR", "ja", 5000, 4744, [8.2, 7.5, 4.2, 0.0, 3.2]),
    ("Brateca", "pt", 5000, 3411, [70.1, 99.8, 99.9, 73.8, 7.9]),
]

THEMES = {
    "light": {"series": ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"], "ink": "#0b0b0b", "ink2": "#52514e", "grid": "#e4e4e1"},
    "dark":  {"series": ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181"], "ink": "#f0efec", "ink2": "#c3c2b7", "grid": "#30363d"},
}


def _value(cell: str) -> bool:
    return cell not in ("", ABSTAIN)


def wmqa_stats(stem: str):
    """(items, resolved, [metric percentages]) for one WorldMedQA-Pain partition."""
    with open(DATA_DIR / f"{stem}.csv", encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    resolved = [r for r in rows if r["ensemble_pain_related"] != ABSTAIN and r["ensemble_pain_presence"] != ABSTAIN]
    pres = sum(1 for r in resolved if r["ensemble_pain_presence"] == "True")
    cov = [sum(1 for r in resolved if _value(r["ensemble_" + f])) for f in ("age", "gender", "ethnicity", "occupation")]
    return len(rows), len(resolved), [100 * pres / len(resolved)] + [100 * c / len(resolved) for c in cov]


def bar_path(x: float, y_top: float, w: float, y_base: float, r: float = 4) -> str:
    """Column with a rounded top and a square baseline."""
    h = y_base - y_top
    if h < r:  # tiny values: a plain sliver, at least 1px so the bar exists
        return f"M{x:.1f},{y_base:.1f}V{y_base - max(h, 1):.1f}H{x + w:.1f}V{y_base:.1f}Z"
    return (f"M{x:.1f},{y_base:.1f}V{y_top + r:.1f}Q{x:.1f},{y_top:.1f} {x + r:.1f},{y_top:.1f}"
            f"H{x + w - r:.1f}Q{x + w:.1f},{y_top:.1f} {x + w:.1f},{y_top + r:.1f}V{y_base:.1f}Z")


def render(panels, theme: dict) -> str:
    """panels: list of (title, [(label, language, [values])])."""
    W, H = 960, 454
    left, right, top, bottom = 44, 16, 114, 52  # top leaves room for hover labels above 100% bars
    panel_gap = 36
    ph = H - top - bottom
    y_base = top + ph
    font = "font-family=\"-apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif\""
    n_groups = sum(len(groups) for _, groups in panels)
    group_w = (W - left - right - panel_gap * (len(panels) - 1)) / n_groups
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" role="img" '
           f'aria-label="PAINT label statistics: pain presence and demographic coverage per partition">']
    out.append('<style>.bar text{opacity:0;pointer-events:none}.bar:hover text{opacity:1}.bar:hover path{opacity:.8}</style>')
    out.append(f'<text x="{left}" y="22" {font} font-size="15" font-weight="600" fill="{theme["ink"]}">PAINT label statistics</text>')
    out.append(f'<text x="{left}" y="41" {font} font-size="12" fill="{theme["ink2"]}">Share of items where both pain fields are resolved. '
               f'WorldMedQA-Pain is computed from the released files; BRIDGE-Pain is Table 6 of the paper.</text>')
    lx = left
    for metric, color in zip(METRICS, theme["series"]):
        out.append(f'<rect x="{lx}" y="53" width="10" height="10" rx="2" fill="{color}"/>')
        out.append(f'<text x="{lx + 15}" y="62" {font} font-size="12" fill="{theme["ink2"]}">{metric}</text>')
        lx += 15 + 7 * len(metric) + 20
    px = left
    bw, gap = 14, 2
    cluster_w = len(METRICS) * bw + (len(METRICS) - 1) * gap
    for title, groups in panels:
        pw = group_w * len(groups)
        out.append(f'<text x="{px + pw / 2:.1f}" y="88" {font} font-size="13" font-weight="600" text-anchor="middle" fill="{theme["ink"]}">{title}</text>')
        for v in (0, 25, 50, 75, 100):
            y = y_base - ph * v / 100
            out.append(f'<line x1="{px:.1f}" y1="{y:.1f}" x2="{px + pw:.1f}" y2="{y:.1f}" stroke="{theme["grid"]}" stroke-width="1"/>')
            if px == left:
                out.append(f'<text x="{left - 8}" y="{y + 4:.1f}" {font} font-size="11" text-anchor="end" fill="{theme["ink2"]}">{v}%</text>')
        for i, (label, lang, values) in enumerate(groups):
            gx = px + i * group_w
            x0 = gx + (group_w - cluster_w) / 2
            for j, (metric, color) in enumerate(zip(METRICS, theme["series"])):
                v = values[j]
                x = x0 + j * (bw + gap)
                y_top = y_base - ph * v / 100
                out.append(f'<g class="bar"><path d="{bar_path(x, y_top, bw, y_base)}" fill="{color}"><title>{label} · {metric}: {v:.1f}%</title></path>'
                           f'<text x="{x + bw / 2:.1f}" y="{y_top - 6:.1f}" {font} font-size="11" font-weight="600" text-anchor="middle" fill="{theme["ink"]}">{v:.1f}%</text></g>')
            cx = gx + group_w / 2
            out.append(f'<text x="{cx:.1f}" y="{y_base + 18}" {font} font-size="12" text-anchor="middle" fill="{theme["ink"]}">{label}</text>')
            out.append(f'<text x="{cx:.1f}" y="{y_base + 33}" {font} font-size="11" text-anchor="middle" fill="{theme["ink2"]}">{lang}</text>')
        px += pw + panel_gap
    out.append("</svg>")
    return "\n".join(out) + "\n"


def main() -> None:
    wmqa = []
    for label, lang, stem in WMQA:
        n, resolved, values = wmqa_stats(stem)
        wmqa.append((label, lang, values))
        print(f"{label:16} n={n:5} resolved={resolved:5}  " + "  ".join(f"{m}={v:.1f}" for m, v in zip(METRICS, values)))
    bridge = [(label, lang, values) for label, lang, _, _, values in BRIDGE]
    panels = [("WorldMedQA-Pain", wmqa), ("BRIDGE-Pain", bridge)]
    OUT_DIR.mkdir(exist_ok=True)
    for name, theme in THEMES.items():
        path = OUT_DIR / ("label_stats.svg" if name == "light" else "label_stats_dark.svg")
        path.write_text(render(panels, theme), encoding="utf-8")
        print("wrote", path.relative_to(ROOT))


if __name__ == "__main__":
    main()
