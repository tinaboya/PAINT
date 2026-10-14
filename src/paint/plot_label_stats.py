"""Draw the WorldMedQA-Pain label statistics as a grouped bar chart (SVG).

Reads data/WorldMedQA-Pain/<Partition>.csv and writes assets/label_stats.svg
(light) and assets/label_stats_dark.svg (dark), which the README embeds with a
<picture> element so GitHub shows the right one. Standard library only.

Run:
    python3 src/paint/plot_label_stats.py
"""

from __future__ import annotations

import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = ROOT / "data" / "WorldMedQA-Pain"
OUT_DIR = ROOT / "assets"

PARTITIONS = [("UK", "UK PLAB"), ("Spain", "Spain MIR"), ("Japan", "Japan NLE"), ("Brazil", "Brazil REVALIDA")]
METRICS = ["Pain presence", "Pain location", "Age", "Gender", "Ethnicity", "Occupation"]
ABSTAIN = "abstain"

THEMES = {
    "light": {"series": ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"], "ink": "#0b0b0b", "ink2": "#52514e", "grid": "#e4e4e1"},
    "dark":  {"series": ["#3987e5", "#d95926", "#199e70", "#c98500"], "ink": "#f0efec", "ink2": "#c3c2b7", "grid": "#30363d"},
}


def _value(cell: str) -> bool:
    return cell not in ("", ABSTAIN)


def stats(partition: str) -> list[float]:
    """Percentages in METRICS order. Pain presence and location: share of all
    items (location counts pain-presence-positive items with a location).
    Demographics: coverage among items where both pain fields are resolved."""
    with open(DATA_DIR / f"{partition}.csv", encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    n = len(rows)
    resolved = [r for r in rows if r["ensemble_pain_related"] != ABSTAIN and r["ensemble_pain_presence"] != ABSTAIN]
    pres = sum(1 for r in rows if r["ensemble_pain_presence"] == "True")
    loc = sum(1 for r in rows if r["ensemble_pain_presence"] == "True" and _value(r["ensemble_location"]))
    cov = [100 * sum(1 for r in resolved if _value(r["ensemble_" + f])) / len(resolved)
           for f in ("age", "gender", "ethnicity", "occupation")]
    return [100 * pres / n, 100 * loc / n] + cov


def bar_path(x: float, y_top: float, w: float, y_base: float, r: float = 4) -> str:
    """Column with a rounded top and a square baseline."""
    h = y_base - y_top
    if h < r:  # tiny values: a plain sliver, at least 1px so the bar exists
        return f"M{x:.1f},{y_base:.1f}V{y_base - max(h, 1):.1f}H{x + w:.1f}V{y_base:.1f}Z"
    return (f"M{x:.1f},{y_base:.1f}V{y_top + r:.1f}Q{x:.1f},{y_top:.1f} {x + r:.1f},{y_top:.1f}"
            f"H{x + w - r:.1f}Q{x + w:.1f},{y_top:.1f} {x + w:.1f},{y_top + r:.1f}V{y_base:.1f}Z")


def render(values: dict[str, list[float]], theme: dict) -> str:
    W, H = 860, 400
    left, right, top, bottom = 44, 16, 78, 40
    pw, ph = W - left - right, H - top - bottom
    y_base = top + ph
    font = "font-family=\"-apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif\""
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" role="img" '
           f'aria-label="WorldMedQA-Pain label statistics by partition">']
    out.append(f'<text x="{left}" y="22" {font} font-size="15" font-weight="600" fill="{theme["ink"]}">WorldMedQA-Pain label statistics</text>')
    out.append(f'<text x="{left}" y="41" {font} font-size="12" fill="{theme["ink2"]}">Pain presence and location: % of items. '
               f'Age to occupation: % of items where both pain fields are resolved.</text>')
    # legend
    lx = left
    for (part, label), color in zip(PARTITIONS, theme["series"]):
        out.append(f'<rect x="{lx}" y="53" width="10" height="10" rx="2" fill="{color}"/>')
        out.append(f'<text x="{lx + 15}" y="62" {font} font-size="12" fill="{theme["ink2"]}">{label}</text>')
        lx += 15 + 7 * len(label) + 20
    # grid and y ticks
    for v in (0, 25, 50, 75, 100):
        y = y_base - ph * v / 100
        out.append(f'<line x1="{left}" y1="{y:.1f}" x2="{left + pw}" y2="{y:.1f}" stroke="{theme["grid"]}" stroke-width="1"/>')
        out.append(f'<text x="{left - 8}" y="{y + 4:.1f}" {font} font-size="11" text-anchor="end" fill="{theme["ink2"]}">{v}%</text>')
    # bars
    bw, gap = 20, 2
    group_w = pw / len(METRICS)
    cluster_w = len(PARTITIONS) * bw + (len(PARTITIONS) - 1) * gap
    for i, metric in enumerate(METRICS):
        gx = left + i * group_w
        x0 = gx + (group_w - cluster_w) / 2
        for j, ((part, _), color) in enumerate(zip(PARTITIONS, theme["series"])):
            v = values[part][i]
            x = x0 + j * (bw + gap)
            y_top = y_base - ph * v / 100
            out.append(f'<path d="{bar_path(x, y_top, bw, y_base)}" fill="{color}"><title>{_label(part)} · {metric}: {v:.1f}%</title></path>')
        out.append(f'<text x="{gx + group_w / 2:.1f}" y="{y_base + 20}" {font} font-size="12" text-anchor="middle" fill="{theme["ink"]}">{metric}</text>')
    out.append("</svg>")
    return "\n".join(out) + "\n"


def _label(part: str) -> str:
    return dict(PARTITIONS)[part]


def main() -> None:
    values = {part: stats(part) for part, _ in PARTITIONS}
    OUT_DIR.mkdir(exist_ok=True)
    for name, theme in THEMES.items():
        path = OUT_DIR / ("label_stats.svg" if name == "light" else "label_stats_dark.svg")
        path.write_text(render(values, theme), encoding="utf-8")
        print("wrote", path.relative_to(ROOT))
    for part, label in PARTITIONS:
        print(f"{label:16}", "  ".join(f"{m}={v:.1f}" for m, v in zip(METRICS, values[part])))


if __name__ == "__main__":
    main()
