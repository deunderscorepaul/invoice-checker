"""Kleine SVG-Bausteine für das Dashboard.

Handgeschriebenes SVG statt matplotlib oder einer JS-Bibliothek: keine
Abhängigkeiten, funktioniert offline, und das Ergebnis ist eine Datei, die man
verschicken oder drucken kann. Jedes Diagramm steht zusätzlich als Tabelle da,
die Grafik ist also nur Lesehilfe.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from html import escape

PALETTE = {
    "precision": "#2563eb",
    "recall": "#f59e0b",
    "f1": "#10b981",
    "tp": "#10b981",
    "fp": "#ef4444",
    "fn": "#f59e0b",
    "tn": "#94a3b8",
    "accent": "#7c3aed",
    "muted": "#64748b",
    "grid": "#e2e8f0",
    "text": "#0f172a",
}

FONT = "font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,Helvetica,Arial,sans-serif"


def fmt(value, digits: int = 3) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _text(x: float, y: float, content: str, *, size: int = 12, anchor: str = "start",
          fill: str = PALETTE["text"], weight: str = "400", opacity: float = 1.0) -> str:
    return (f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" text-anchor="{anchor}" '
            f'fill="{fill}" font-weight="{weight}" opacity="{opacity}" '
            f'style="{FONT}">{escape(str(content))}</text>')


@dataclass
class Chart:
    svg: str
    caption: str = ""
    height: int = 0

    def render(self) -> str:
        caption = f'<figcaption>{escape(self.caption)}</figcaption>' if self.caption else ""
        return f'<figure class="chart">{self.svg}{caption}</figure>'


# --- KPI donut ----------------------------------------------------------------------------------

def gauge(value: float | None, label: str, sublabel: str = "", *,
          size: int = 120, colour: str = PALETTE["f1"]) -> Chart:
    """Ein Wert zwischen 0 und 1 als Ring."""
    value = 0.0 if value is None else max(0.0, min(1.0, float(value)))
    radius = size / 2 - 10
    centre = size / 2
    circumference = 2 * 3.141592653589793 * radius
    filled = circumference * value
    return Chart(
        svg=(
            f'<svg viewBox="0 0 {size} {size}" width="{size}" height="{size}" role="img" '
            f'aria-label="{escape(label)} {value:.3f}">'
            f'<circle cx="{centre}" cy="{centre}" r="{radius:.1f}" fill="none" '
            f'stroke="{PALETTE["grid"]}" stroke-width="10"/>'
            f'<circle cx="{centre}" cy="{centre}" r="{radius:.1f}" fill="none" stroke="{colour}" '
            f'stroke-width="10" stroke-linecap="round" '
            f'stroke-dasharray="{filled:.1f} {circumference - filled:.1f}" '
            f'transform="rotate(-90 {centre} {centre})"/>'
            f'{_text(centre, centre + 2, f"{value:.3f}", size=20, anchor="middle", weight="600")}'
            f'{_text(centre, centre + 18, label, size=10, anchor="middle", fill=PALETTE["muted"])}'
            f'{_text(centre, centre + 30, sublabel, size=9, anchor="middle", fill=PALETTE["muted"])}'
            f'</svg>'
        ),
        height=size,
    )


# --- grouped horizontal bars --------------------------------------------------------------------

def grouped_bars(rows: list[tuple[str, dict[str, float]]], series: list[str], *,
                 colours: dict[str, str] | None = None, label_width: int = 260,
                 row_height: int = 26, value_width: int = 46, title: str = "") -> Chart:
    """Eine Zeile pro Eintrag, ein Balken pro Serie. Scrollbar statt seitenweise."""
    colours = colours or PALETTE
    width = 980
    plot_left = label_width
    plot_right = width - value_width - 10
    plot_width = plot_right - plot_left
    header = 26
    height = header + max(1, len(rows)) * row_height + 12

    parts = [f'<svg viewBox="0 0 {width} {height}" width="100%" height="{height}" '
             f'preserveAspectRatio="xMinYMin meet" role="img" aria-label="{escape(title or "metrics")}">']
    # Hilfslinien bei 0, 0.25, 0.5, 0.75, 1
    for tick in (0.0, 0.25, 0.5, 0.75, 1.0):
        x = plot_left + plot_width * tick
        parts.append(f'<line x1="{x:.1f}" y1="{header - 6}" x2="{x:.1f}" y2="{height - 8}" '
                     f'stroke="{PALETTE["grid"]}" stroke-width="1"/>')
        parts.append(_text(x, header - 12, f"{tick:.2f}", size=9, anchor="middle",
                           fill=PALETTE["muted"]))

    bar_height = max(3.0, (row_height - 8) / max(1, len(series)))
    for index, (label, values) in enumerate(rows):
        y0 = header + index * row_height
        parts.append(_text(plot_left - 8, y0 + row_height / 2 + 3, label[:44], size=11,
                           anchor="end", fill=PALETTE["text"]))
        for series_index, key in enumerate(series):
            value = values.get(key)
            if value is None:
                continue
            value = max(0.0, min(1.0, float(value)))
            y = y0 + 4 + series_index * bar_height
            bar_width = plot_width * value
            parts.append(
                f'<rect x="{plot_left}" y="{y:.1f}" width="{bar_width:.1f}" '
                f'height="{bar_height - 0.5:.1f}" rx="1.5" fill="{colours.get(key, "#888")}">'
                f'<title>{escape(label)} — {escape(key)} {value:.3f}</title></rect>'
            )
            if series_index == 0:
                parts.append(_text(plot_right + 6, y0 + row_height / 2 + 3,
                                   fmt(values.get("f1"), 2), size=11,
                                   fill=PALETTE["muted"]))
    parts.append("</svg>")
    return Chart(svg="".join(parts), caption=title, height=height)


# --- stacked single bar -------------------------------------------------------------------------

@dataclass
class StackSegment:
    label: str
    value: float
    colour: str


def stacked_bar(segments: list[StackSegment], *, width: int = 980, height: int = 54,
                title: str = "") -> Chart:
    total = sum(max(0.0, seg.value) for seg in segments) or 1.0
    parts = [f'<svg viewBox="0 0 {width} {height}" width="100%" height="{height}" '
             f'preserveAspectRatio="xMinYMin meet" role="img" '
             f'aria-label="{escape(title or "composition")}">']
    x = 0.0
    for seg in segments:
        fraction = max(0.0, seg.value) / total
        seg_width = width * fraction
        if seg_width <= 0:
            continue
        parts.append(
            f'<rect x="{x:.1f}" y="0" width="{seg_width:.1f}" height="26" fill="{seg.colour}">'
            f'<title>{escape(seg.label)}: {seg.value:g} ({fraction:.1%})</title></rect>'
        )
        if seg_width > 54:
            parts.append(_text(x + seg_width / 2, 18, f"{seg.value:g}", size=11, anchor="middle",
                               fill="#ffffff", weight="600"))
        x += seg_width

    # Legende unter dem Balken.
    legend_x = 0.0
    for seg in segments:
        parts.append(f'<rect x="{legend_x:.1f}" y="36" width="10" height="10" rx="2" '
                     f'fill="{seg.colour}"/>')
        parts.append(_text(legend_x + 14, 45, f"{seg.label} ({seg.value:g})", size=11,
                           fill=PALETTE["muted"]))
        legend_x += 14 + len(f"{seg.label} ({seg.value:g})") * 6.6 + 18
    parts.append("</svg>")
    return Chart(svg="".join(parts), caption=title, height=height)


# --- histogram ----------------------------------------------------------------------------------

def histogram(values: list[float], *, bins: int = 24, width: int = 980, height: int = 180,
              title: str = "", unit: str = "ms") -> Chart:
    if not values:
        return Chart(svg=f'<svg viewBox="0 0 {width} 60" width="100%" height="60"></svg>',
                     caption=f"{title} (no data)", height=60)

    low, high = min(values), max(values)
    span = (high - low) or 1.0
    counts = [0] * bins
    for value in values:
        index = min(bins - 1, int((value - low) / span * bins))
        counts[index] += 1
    peak = max(counts) or 1

    pad_left, pad_bottom, pad_top = 52, 30, 14
    plot_width = width - pad_left - 16
    plot_height = height - pad_bottom - pad_top
    bar_width = plot_width / bins

    parts = [f'<svg viewBox="0 0 {width} {height}" width="100%" height="{height}" role="img" '
             f'aria-label="{escape(title)}">']
    # y-Hilfslinien
    for fraction in (0.0, 0.5, 1.0):
        y = pad_top + plot_height * (1 - fraction)
        parts.append(f'<line x1="{pad_left}" y1="{y:.1f}" x2="{width - 16}" y2="{y:.1f}" '
                     f'stroke="{PALETTE["grid"]}"/>')
        parts.append(_text(pad_left - 6, y + 4, f"{int(peak * fraction)}", size=10, anchor="end",
                           fill=PALETTE["muted"]))
    for index, count in enumerate(counts):
        bar_height = plot_height * (count / peak)
        x = pad_left + index * bar_width
        y = pad_top + plot_height - bar_height
        parts.append(
            f'<rect x="{x:.1f}" y="{y:.1f}" width="{max(1.0, bar_width - 1):.1f}" '
            f'height="{bar_height:.1f}" fill="{PALETTE["precision"]}" opacity="0.85">'
            f'<title>{"%.1f" % (low + span * index / bins)}–'
            f'{"%.1f" % (low + span * (index + 1) / bins)} {unit}: {count}</title></rect>'
        )
    parts.append(_text(pad_left, height - 8, f"{low:.1f} {unit}", size=10,
                       fill=PALETTE["muted"]))
    parts.append(_text(width - 16, height - 8, f"{high:.1f} {unit}", size=10, anchor="end",
                       fill=PALETTE["muted"]))
    parts.append("</svg>")
    return Chart(svg="".join(parts), caption=title, height=height)


# --- scatter ------------------------------------------------------------------------------------

def scatter(points: list[tuple[str, float, float]], *, x_label: str, y_label: str,
            width: int = 980, height: int = 260, title: str = "",
            y_range: tuple[float, float] = (0.0, 1.0)) -> Chart:
    pad_left, pad_bottom, pad_top, pad_right = 60, 38, 16, 24
    plot_width = width - pad_left - pad_right
    plot_height = height - pad_bottom - pad_top
    xs = [point[1] for point in points] or [0.0]
    x_min, x_max = min(xs), max(xs)
    if x_max == x_min:
        x_max = x_min + 1.0

    def sx(value: float) -> float:
        return pad_left + plot_width * (value - x_min) / (x_max - x_min)

    def sy(value: float) -> float:
        low, high = y_range
        clamped = max(low, min(high, value))
        return pad_top + plot_height * (1 - (clamped - low) / (high - low))

    parts = [f'<svg viewBox="0 0 {width} {height}" width="100%" height="{height}" role="img" '
             f'aria-label="{escape(title)}">']
    for fraction in (0.0, 0.25, 0.5, 0.75, 1.0):
        y = sy(y_range[0] + (y_range[1] - y_range[0]) * fraction)
        parts.append(f'<line x1="{pad_left}" y1="{y:.1f}" x2="{width - pad_right}" y2="{y:.1f}" '
                     f'stroke="{PALETTE["grid"]}"/>')
        parts.append(_text(pad_left - 6, y + 4, f"{fraction:.2f}", size=10, anchor="end",
                           fill=PALETTE["muted"]))
    for name, x_value, y_value in points:
        cx, cy = sx(x_value), sy(y_value)
        parts.append(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="6" fill="{PALETTE["accent"]}" '
                     f'opacity="0.8"><title>{escape(name)}: {x_label} {x_value:.0f}, '
                     f'{y_label} {y_value:.3f}</title></circle>')
        parts.append(_text(cx + 9, cy + 4, name, size=10, fill=PALETTE["text"]))
    parts.append(_text(pad_left + plot_width / 2, height - 6, x_label, size=11, anchor="middle",
                       fill=PALETTE["muted"]))
    parts.append("</svg>")
    return Chart(svg="".join(parts), caption=title, height=height)


# --- table --------------------------------------------------------------------------------------

def table(headers: list[str], rows: list[list[str]], *, numeric_from: int = 0,
          highlight: callable = None) -> str:
    head = "".join(f"<th>{escape(str(h))}</th>" for h in headers)
    body = []
    for row in rows:
        cells = []
        for index, cell in enumerate(row):
            css = "num" if index >= numeric_from else ""
            if highlight is not None:
                extra = highlight(row, index)
                if extra:
                    css = f"{css} {extra}".strip()
            cells.append(f'<td class="{css}">{escape(str(cell))}</td>')
        body.append("<tr>" + "".join(cells) + "</tr>")
    return f'<table><thead><tr>{head}</tr></thead><tbody>{"".join(body)}</tbody></table>'


def bar_cell(value: float | None, colour: str = PALETTE["f1"]) -> str:
    """Mini-Balken für eine Tabellenzelle."""
    if value is None:
        return "n/a"
    value = max(0.0, min(1.0, float(value)))
    return (f'<span class="minibar"><span class="minibar-fill" '
            f'style="width:{value * 100:.1f}%;background:{colour}"></span></span>'
            f'<span class="minibar-label">{value:.3f}</span>')
