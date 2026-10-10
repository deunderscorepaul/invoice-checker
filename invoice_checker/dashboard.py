"""Erzeugt das HTML-Dashboard aus den Ergebnisdateien.

Quellen:
  data/synth/manifest.json      Korpus
  data/out/predictions_*.jsonl  Vorhersagen + Ground Truth

Die Kennzahlen werden mit demselben Code wie ``eval`` neu berechnet, das
Dashboard kann der CLI also nicht widersprechen. Ergebnis ist eine HTML-Datei mit
Inline-SVG, ohne Server und ohne Netz.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from html import escape
from pathlib import Path

from . import charts
from .config import OUT_DIR, SYNTH_DIR
from .evaluate import EvalReport, evaluate
from .taxonomy import FAULT_BY_CODE, INJECTABLE_CODES

SERIES_COLOURS = {
    "precision": charts.PALETTE["precision"],
    "recall": charts.PALETTE["recall"],
    "f1": charts.PALETTE["f1"],
}


@dataclass
class Variant:
    """Ein bewerteter Lauf: Vorhersagedatei plus neu berechnete Kennzahlen."""
    tag: str
    report: EvalReport
    path: Path
    source: str = ""
    extraction: str = ""
    latencies: list[float] = field(default_factory=list)
    predictions_per_doc: dict[str, int] = field(default_factory=dict)
    truth_per_doc: dict[str, int] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    @property
    def is_llm(self) -> bool:
        return "llm" in self.extraction or self.report.latency_mean_ms > 100


def load_variant(path: Path) -> Variant:
    report = evaluate(path, tag=path.stem.replace("predictions_", ""))
    latencies: list[float] = []
    predictions: dict[str, int] = {}
    truth: dict[str, int] = {}
    warnings: list[str] = []
    source = extraction = ""

    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            doc_id = record.get("doc_id", "?")
            latency = record.get("latency_ms")
            if isinstance(latency, (int, float)):
                latencies.append(float(latency))
            predictions[doc_id] = len([c for c in record.get("codes", [])
                                       if c in INJECTABLE_CODES])
            truth[doc_id] = len([c for c in record.get("truth_codes", [])
                                 if c in INJECTABLE_CODES])
            source = record.get("source", source)
            extraction = record.get("extraction", extraction)
            for note in record.get("notes", []):
                if note not in warnings:
                    warnings.append(note)
    return Variant(tag=report.tag, report=report, path=path, source=source,
                   extraction=extraction, latencies=latencies,
                   predictions_per_doc=predictions, truth_per_doc=truth, warnings=warnings)


def discover_variants(out_dir: Path = OUT_DIR) -> list[Variant]:
    variants: list[Variant] = []
    for path in sorted(Path(out_dir).glob("predictions_*.jsonl")):
        try:
            variants.append(load_variant(path))
        except (OSError, json.JSONDecodeError):
            continue

    # Oracle zuerst, dann deterministisch, dann Modell -- feste Lesereihenfolge.
    def sort_key(variant: Variant) -> tuple[int, str]:
        if variant.source == "oracle":
            return (0, variant.tag)
        if variant.extraction == "regex":
            return (1, variant.tag)
        return (2, variant.tag)

    return sorted(variants, key=sort_key)


def load_manifest(synth_dir: Path = SYNTH_DIR) -> dict:
    path = Path(synth_dir) / "manifest.json"
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


# --- page assembly ------------------------------------------------------------------------------

CSS = """
*{box-sizing:border-box}
body{margin:0;background:#f1f5f9;color:#0f172a;
 font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,Helvetica,Arial,sans-serif;
 line-height:1.5;font-size:15px}
header{background:#0f172a;color:#f8fafc;padding:28px 32px}
header h1{margin:0 0 6px;font-size:22px;font-weight:600}
header p{margin:0;color:#94a3b8;font-size:13px}
.wrap{max-width:1180px;margin:0 auto;padding:24px 32px 64px}
section{background:#fff;border:1px solid #e2e8f0;border-radius:10px;padding:22px 24px;margin:0 0 22px}
section>h2{margin:0 0 4px;font-size:17px;font-weight:600}
section>p.hint{margin:0 0 18px;color:#64748b;font-size:13px}
.kpis{display:flex;flex-wrap:wrap;gap:14px}
.kpi{flex:1 1 150px;background:#f8fafc;border:1px solid #e2e8f0;border-radius:9px;
 padding:14px;text-align:center}
.kpi .v{font-size:24px;font-weight:600}
.kpi .l{color:#64748b;font-size:12px;text-transform:uppercase;letter-spacing:.04em}
.twocol{display:grid;grid-template-columns:1fr 1fr;gap:22px}
@media(max-width:820px){.twocol{grid-template-columns:1fr}}
figure.chart{margin:0 0 8px}
figure.chart figcaption{color:#64748b;font-size:12px;margin-top:6px}
table{border-collapse:collapse;width:100%;font-size:13px}
th,td{text-align:left;padding:7px 10px;border-bottom:1px solid #e2e8f0;white-space:nowrap}
th{background:#f8fafc;font-weight:600;color:#475569;font-size:12px;text-transform:uppercase;
 letter-spacing:.03em}
td.num{text-align:right;font-variant-numeric:tabular-nums}
tbody tr:hover{background:#f8fafc}
.scroll{max-height:520px;overflow:auto;border:1px solid #e2e8f0;border-radius:8px}
.scroll table th{position:sticky;top:0;z-index:1}
.scroll table{border-collapse:separate;border-spacing:0}
.minibar{display:inline-block;width:56px;height:8px;background:#e2e8f0;border-radius:4px;
 vertical-align:middle;margin-right:6px;overflow:hidden}
.minibar-fill{display:block;height:100%}
.minibar-label{font-variant-numeric:tabular-nums;color:#334155}
.tag{display:inline-block;padding:2px 8px;border-radius:99px;font-size:11px;font-weight:600}
.tag-oracle{background:#ede9fe;color:#5b21b6}
.tag-regex{background:#fef3c7;color:#92400e}
.tag-llm{background:#dbeafe;color:#1e40af}
.perfect{background:#ecfdf5}
.weak{background:#fef2f2}
.note{background:#fffbeb;border:1px solid #fde68a;border-radius:8px;padding:12px 14px;
 font-size:13px;margin:0 0 10px}
.note b{color:#92400e}
details{border:1px solid #e2e8f0;border-radius:8px;padding:10px 14px;margin-top:14px}
summary{cursor:pointer;font-weight:600;font-size:14px}
footer{color:#64748b;font-size:12px;text-align:center;padding:0 32px 40px}
code{background:#f1f5f9;padding:1px 5px;border-radius:4px;font-size:12px}
"""


def _tag_class(variant: Variant) -> str:
    if variant.source == "oracle":
        return "tag tag-oracle"
    if variant.extraction == "regex":
        return "tag tag-regex"
    return "tag tag-llm"


def _kpi(label: str, value: str) -> str:
    return f'<div class="kpi"><div class="v">{escape(value)}</div><div class="l">{escape(label)}</div></div>'


def _overview(manifest: dict, variants: list[Variant]) -> str:
    documents = 0
    faulty = 0
    clean = 0
    if variants:
        documents = variants[0].report.documents
        faulty = variants[0].report.faulty_documents
        clean = variants[0].report.clean_documents
    elif manifest:
        documents = manifest.get("count", 0)
        faulty = manifest.get("faulty", 0)
        clean = documents - faulty

    cards = [
        _kpi("documents", f"{documents}"),
        _kpi("faulty", f"{faulty}"),
        _kpi("clean", f"{clean}"),
        _kpi("fault rate", f"{faulty / documents:.0%}" if documents else "n/a"),
        _kpi("fault codes", f"{len(FAULT_BY_CODE)}"),
        _kpi("variants scored", f"{len(variants)}"),
    ]
    return f'<section><h2>Corpus and run overview</h2>' \
           f'<p class="hint">Ground truth counts as reported by the evaluation harness. Faults are ' \
           f'injected programmatically, so every label is exact.</p>' \
           f'<div class="kpis">{"".join(cards)}</div></section>'


def _headline(variants: list[Variant]) -> str:
    """Headline metric cards: one row of gauges per variant."""
    if not variants:
        return ""
    blocks = []
    for variant in variants:
        report = variant.report
        micro = report.micro
        blocks.append(
            f'<div style="flex:1 1 320px">'
            f'<h3 style="margin:0 0 10px;font-size:14px">'
            f'<span class="{_tag_class(variant)}">{escape(variant.tag)}</span>'
            f'<span style="color:#64748b;font-weight:400;font-size:12px;margin-left:8px">'
            f'{escape(variant.extraction or variant.source or "")}</span></h3>'
            f'<div style="display:flex;gap:10px;flex-wrap:wrap">'
            + charts.gauge(micro["precision"], "precision").svg
            + charts.gauge(micro["recall"], "recall", colour=charts.PALETTE["recall"]).svg
            + charts.gauge(micro["f1"], "micro F1").svg
            + charts.gauge(report.document_level["f1"], "doc F1",
                           colour=charts.PALETTE["accent"]).svg
            + '</div></div>'
        )
    return ('<section><h2>Headline performance</h2>'
            '<p class="hint">Fault-level micro precision/recall/F1 plus document-level F1 '
            '(did the checker flag the right invoices at all). Micro-F1 across all fault codes is '
            'the headline number.</p>'
            f'<div style="display:flex;gap:26px;flex-wrap:wrap">{"".join(blocks)}</div></section>')


def _comparison(variants: list[Variant]) -> str:
    if not variants:
        return ""
    headers = ["variant", "field source", "micro P", "micro R", "micro F1", "macro F1",
               "doc F1", "tp", "fp", "fn", "latency/invoice"]
    rows = []
    for variant in variants:
        macro = variant.report.macro
        doc = variant.report.document_level
        micro = variant.report.micro
        source = variant.source or "-"
        if variant.extraction:
            source += f" + {variant.extraction}"
        latency = variant.report.latency_mean_ms
        rows.append([
            variant.tag, source,
            f"{micro['precision']:.3f}", f"{micro['recall']:.3f}", f"{micro['f1']:.3f}",
            f"{macro['f1']:.3f}", f"{doc['f1']:.3f}",
            str(micro["tp"]), str(micro["fp"]), str(micro["fn"]),
            f"{latency:,.0f} ms" if latency < 1000 else f"{latency / 1000:.1f} s",
        ])
    return ('<section><h2>Variant comparison</h2>'
            '<p class="hint">The oracle row uses ground-truth fields, so it is a <b>coverage '
            'reference</b> rather than an accuracy claim: it shows what the rule engine could achieve '
            'with perfect extraction. The gap to the other rows is the cost of imperfect extraction.'
            '</p>'
            + charts.table(headers, rows, numeric_from=2) + '</section>')


def _composition(variants: list[Variant]) -> str:
    if not variants:
        return ""
    variant = variants[-1]
    report = variant.report
    doc = report.document_level
    # Document-level decomposition of the confusion matrix.
    detected_correctly = doc["tp"] - doc["partial_matches"]
    segments = [
        charts.StackSegment("fully correct documents", detected_correctly, charts.PALETTE["tp"]),
        charts.StackSegment("partially flagged", doc["partial_matches"], charts.PALETTE["fn"]),
        charts.StackSegment("missed faults", doc["fn"], charts.PALETTE["fp"]),
        charts.StackSegment("false alarms", doc["fp"], "#f97316"),
        charts.StackSegment("correctly clean", doc["tn"], charts.PALETTE["tn"]),
    ]
    missing_vs_wrong = [
        (code, metrics)
        for code, metrics in report.codes.items() if metrics.injected or metrics.fp
    ]
    # Die zwei Fehlerarten sind nicht gleich teuer: ein Treffer zu wenig versteckt einen
    # Fehler, ein Fehlalarm kostet nur Zeit. Deshalb getrennt dargestellt.
    segments_missing = [
        charts.StackSegment("true positives", report.micro["tp"], charts.PALETTE["tp"]),
        charts.StackSegment("missed (false negatives)", report.micro["fn"], charts.PALETTE["fn"]),
        charts.StackSegment("false alarms (false positives)", report.micro["fp"],
                            charts.PALETTE["fp"]),
    ]
    return (
        '<section><h2>Where the findings came from</h2>'
        f'<p class="hint">Variant <b>{escape(variant.tag)}</b>. A missed fault hides a defect; a '
        'false alarm costs a human a review. The two are shown separately because they are not '
        'equivalent costs.</p>'
        '<div class="twocol">'
        f'<div><h3 style="font-size:14px;margin:0 0 8px">Documents ({report.documents})</h3>'
        + charts.stacked_bar(segments, title="").render() + '</div>'
        f'<div><h3 style="font-size:14px;margin:0 0 8px">Fault findings</h3>'
        + charts.stacked_bar(segments_missing, title="").render() + '</div>'
        '</div>'
        f'<p class="hint" style="margin-top:16px">{len(missing_vs_wrong)} distinct fault codes '
        'were exercised by this variant.</p>'
        '</section>'
    )


def _per_code(variants: list[Variant]) -> str:
    if not variants:
        return ""
    variant = variants[-1]

    table_rows = []
    for code, metrics in variant.report.codes.items():
        if not (metrics.injected or metrics.fp or metrics.fn):
            continue
        spec = FAULT_BY_CODE.get(code)
        table_rows.append({
            "code": code,
            "severity": spec.severity.value if spec else "info",
            "detector": spec.detector.value if spec else "rules",
            "injected": metrics.injected, "tp": metrics.tp, "fp": metrics.fp, "fn": metrics.fn,
            "precision": metrics.precision, "recall": metrics.recall, "f1": metrics.f1,
        })
    # Schlechteste zuerst: die Grafik soll zeigen was kaputt ist, nicht was läuft.
    table_rows.sort(key=lambda row: (row["f1"], -row["injected"], row["code"]))

    chart = charts.grouped_bars(
        [(row["code"], {"precision": row["precision"], "recall": row["recall"], "f1": row["f1"]})
         for row in table_rows],
        ["precision", "recall", "f1"], colours=SERIES_COLOURS,
        label_width=250, row_height=27,
        title="Precision / recall / F1 per fault code (worst first; right-hand number is F1)",
    )

    headers = ["fault code", "severity", "detector", "injected", "tp", "fp", "fn",
               "precision", "recall", "F1"]
    body = []
    for row in table_rows:
        body.append([
            row["code"], row["severity"], row["detector"], str(row["injected"]),
            str(row["tp"]), str(row["fp"]), str(row["fn"]),
            charts.bar_cell(row["precision"], charts.PALETTE["precision"]),
            charts.bar_cell(row["recall"], charts.PALETTE["recall"]),
            charts.bar_cell(row["f1"]),
        ])
    head = "".join(f"<th>{escape(h)}</th>" for h in headers)
    html_table = (f'<table><thead><tr>{head}</tr></thead><tbody>'
                  + "".join("<tr>" + "".join(
                      f'<td class="{"num" if i >= 3 else ""}">{cell}</td>'
                      for i, cell in enumerate(r)) + "</tr>" for r in body)
                  + '</tbody></table>')

    return (
        '<section><h2>Per-fault-code performance</h2>'
        f'<p class="hint">Variant <b>{escape(variant.tag)}</b>. Rows are sorted worst-F1-first. '
        'Codes with no injected instances and no false positives are omitted.</p>'
        + chart.render()
        + f'<div class="scroll" style="margin-top:14px">{html_table}</div>'
        + '</section>'
    )


def _latency(variants: list[Variant]) -> str:
    with_latency = [v for v in variants if v.latencies and v.report.latency_mean_ms > 1]
    if not with_latency:
        return ('<section><h2>Latency</h2><p class="hint">No variant recorded a measurable '
                'per-invoice latency, so there is nothing to plot.</p></section>')

    blocks = []
    for variant in with_latency:
        blocks.append(
            f'<div><h3 style="font-size:14px;margin:0 0 8px">'
            f'<span class="{_tag_class(variant)}">{escape(variant.tag)}</span> '
            f'<span style="color:#64748b;font-weight:400;font-size:12px">'
            f'mean {variant.report.latency_mean_ms:,.0f} ms · '
            f'p95 {variant.report.latency_p95_ms:,.0f} ms</span></h3>'
            + charts.histogram(variant.latencies, title="",
                               unit="ms").render() + '</div>'
        )

    points = [(v.tag, v.report.latency_mean_ms, v.report.micro["f1"]) for v in with_latency]
    return ('<section><h2>Latency vs quality</h2>'
            '<p class="hint">Per-invoice latency distribution, and the tradeoff against fault-level '
            'F1. The deterministic runs are sub-millisecond, which is why the model stage dominates '
            'total runtime by orders of magnitude.</p>'
            f'{"".join(blocks)}'
            + charts.scatter(points, x_label="mean latency (ms)", y_label="micro F1",
                             title="Quality against cost — further up and further left is better")
            .render() + '</section>')


def _detector_split(variants: list[Variant]) -> str:
    if not variants:
        return ""
    totals: dict[str, dict[str, int]] = {}
    for code in FAULT_BY_CODE.values():
        bucket = totals.setdefault(code.detector.value, {"codes": 0, "detected": 0})
        bucket["codes"] += 1

    variant = variants[-1]
    for code, metrics in variant.report.codes.items():
        spec = FAULT_BY_CODE.get(code)
        if spec is None:
            continue
        totals[spec.detector.value]["detected"] += metrics.tp

    rows = [[key, str(value["codes"]), str(value["detected"])]
            for key, value in sorted(totals.items())]
    return ('<section><h2>Detector attribution</h2>'
            '<p class="hint">How many fault codes exist per detector type, and how many findings '
            'that detector produced in the selected variant. This is the evidence for the central '
            'design claim: nearly all validation is deterministic.</p>'
            + charts.table(["detector", "fault codes defined", "findings in this variant"], rows,
                            numeric_from=1)
            + '</section>')


def _diagnostics(variants: list[Variant]) -> str:
    notes: list[str] = []
    for variant in variants:
        if variant.report.non_injectable:
            note = ", ".join(f"{code} × {count}"
                             for code, count in sorted(variant.report.non_injectable.items()))
            notes.append(f"<b>{escape(variant.tag)}</b>: non-injectable codes reported ({note}). "
                         f"These are excluded from scoring and shown here instead.")
        for warning in variant.warnings[:3]:
            notes.append(f"<b>{escape(variant.tag)}</b>: {escape(warning)}")
        if variant.report.non_injectable.get("EXTRACTION_FAILED"):
            notes.append(f"<b>{escape(variant.tag)}</b>: some documents produced no fields at all.")

    if not notes:
        notes.append("No extraction warnings, fallbacks or non-injectable findings were recorded "
                     "for these variants.")
    body = "".join(f'<div class="note">{note}</div>' for note in notes)
    return ('<section><h2>Diagnostics</h2>'
            '<p class="hint">Extraction health, fallbacks and findings that cannot be scored '
            'against injected ground truth. Reported rather than hidden inside accuracy.</p>'
            f'{body}</section>')


def build_dashboard(out_dir: Path = OUT_DIR, synth_dir: Path = SYNTH_DIR) -> tuple[str, Path]:
    variants = discover_variants(out_dir)
    manifest = load_manifest(synth_dir)
    generated = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    if not variants:
        body = ('<section><h2>No predictions found</h2>'
                '<p class="hint">Run one of the following, then regenerate the dashboard:</p>'
                '<p><code>run --baseline</code><br><code>run --source text --extraction regex '
                '--tag regex</code><br><code>run --source text --extraction llm --tag llm</code></p>'
                '</section>')
    else:
        body = (
            _overview(manifest, variants)
            + _headline(variants)
            + _comparison(variants)
            + _composition(variants)
            + _per_code(variants)
            + _latency(variants)
            + _detector_split(variants)
            + _diagnostics(variants)
        )

    corpus_note = ""
    if manifest:
        corpus_note = (f"corpus: {manifest.get('count', 0)} documents, "
                       f"{manifest.get('faulty', 0)} faulty, "
                       f"{manifest.get('pdfs', 0)} PDFs, {manifest.get('scans', 0)} scans")

    html = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Invoice checker — results dashboard</title>
<style>{CSS}</style></head><body>
<header>
  <h1>Local AI invoice checker — evaluation dashboard</h1>
  <p>{escape(corpus_note) if corpus_note else 'no corpus manifest found'} &nbsp;·&nbsp;
     generated {escape(generated)} &nbsp;·&nbsp; {len(variants)} variant(s) scored</p>
</header>
<div class="wrap">
{body}
<details>
  <summary>How these numbers are produced</summary>
  <p class="hint" style="margin-top:10px">
  Ground truth is derived by running the deterministic rule engine over the stored document, not
  taken from the injection list alone: a single injected mutation cascades (corrupting the VAT amount
  breaks both <code>net × rate = VAT</code> and <code>net + VAT = gross</code>), so labelling only the
  injected code would score the second, legitimate detection as a false positive.
  </p>
  <p class="hint">
  Metrics are per fault type, macro-averaged, and restricted to injectable codes; non-injectable
  outcomes appear under Diagnostics rather than inside the accuracy figure. Document-level F1 treats
  a document as detected when at least one of its faults was found, with partial matches counted
  separately.
  </p>
  <p class="hint">
  An oracle variant reports what the rule engine achieves with perfect field extraction and is a
  coverage reference, not an accuracy claim. The informative comparison is between the extraction
  strategies.
  </p>
</details>
</div>
<footer>Generated by <code>invoice_checker dashboard</code> · self-contained file, no network required</footer>
</body></html>"""

    target = Path(out_dir) / "dashboard.html"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(html, encoding="utf-8")
    return html, target


# --- terminal summary ---------------------------------------------------------------------------

def terminal_summary(out_dir: Path = OUT_DIR) -> str:
    """Kurzfassung fürs Terminal, wenn kein Browser zur Hand ist."""
    variants = discover_variants(out_dir)
    if not variants:
        return "no predictions found in " + str(out_dir)

    lines: list[str] = []
    width = max(len(v.tag) for v in variants) + 2
    lines.append(f"{'variant':<{width}}{'microF1':>9}{'macroF1':>9}{'docF1':>8}"
                 f"{'P':>7}{'R':>7}{'tp':>6}{'fp':>6}{'fn':>6}{'latency':>11}")
    lines.append("-" * (width + 69))
    for variant in variants:
        report = variant.report
        latency = report.latency_mean_ms
        latency_text = f"{latency:,.0f} ms" if latency < 1000 else f"{latency / 1000:.1f} s"
        lines.append(
            f"{variant.tag:<{width}}{report.micro['f1']:>9.3f}{report.macro['f1']:>9.3f}"
            f"{report.document_level['f1']:>8.3f}{report.micro['precision']:>7.3f}"
            f"{report.micro['recall']:>7.3f}{report.micro['tp']:>6}{report.micro['fp']:>6}"
            f"{report.micro['fn']:>6}{latency_text:>11}"
        )

    # Schwächste Codes des interessantesten Laufs (der letzte, also modellnächste).
    variant = variants[-1]
    weak = sorted(
        ((code, m) for code, m in variant.report.codes.items() if m.injected and m.f1 < 0.9),
        key=lambda item: item[1].f1,
    )[:12]
    if weak:
        lines.append("")
        lines.append(f"weakest codes in '{variant.tag}' (injected count in brackets):")
        for code, metrics in weak:
            bar = "#" * int(round(metrics.f1 * 20))
            lines.append(f"  {code:<38}{metrics.f1:>6.2f}  {bar:<20} "
                         f"(inj {metrics.injected}, fp {metrics.fp}, fn {metrics.fn})")
    else:
        lines.append("")
        lines.append(f"no code below 0.90 F1 in '{variant.tag}'")
    return "\n".join(lines)
