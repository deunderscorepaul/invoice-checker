"""CSV-Export und lesbare Fehlerberichte.

Damit die Auswertung direkt in Bericht oder Tabelle geht und niemand Zahlen von
Hand abtippt.
"""
from __future__ import annotations

import csv
from pathlib import Path

from .evaluate import EvalReport
from .models import CheckResult
from .taxonomy import FAULT_BY_CODE

CSV_COLUMNS = ["code", "severity", "detector", "injected", "tp", "fp", "fn",
               "precision", "recall", "f1"]


def write_code_table_csv(report: EvalReport, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for code, metrics in sorted(report.codes.items(),
                                    key=lambda item: (-item[1].injected, item[0])):
            writer.writerow({column: value for column, value in metrics.to_dict().items()
                             if column in CSV_COLUMNS})
    return path


def render_invoice_report(result: CheckResult) -> str:
    """Feststellungen einer Rechnung, nach Schwere sortiert -- so würde ein UI das zeigen."""
    order = {"critical": 0, "major": 1, "minor": 2, "info": 3}
    lines = [f"Invoice {result.doc_id}: "
             f"{'no faults found' if not result.faults else f'{len(result.faults)} finding(s)'}"]
    if result.notes:
        for note in result.notes:
            lines.append(f"  note: {note}")
    for fault in sorted(result.faults, key=lambda f: (order.get(f.severity, 9), f.code)):
        spec = FAULT_BY_CODE.get(fault.code)
        reference = f"  [{spec.legal_reference}]" if spec and spec.legal_reference else ""
        lines.append(f"  [{fault.severity.upper():8}] {fault.code}: {fault.message}{reference}")
        for key, value in fault.evidence.items():
            lines.append(f"             {key} = {value}")
    return "\n".join(lines)


def summarise_fault_frequency(results: list[CheckResult]) -> list[tuple[str, int]]:
    counts: dict[str, int] = {}
    for result in results:
        for code in result.codes:
            counts[code] = counts.get(code, 0) + 1
    return sorted(counts.items(), key=lambda item: (-item[1], item[0]))


def render_summary_table(results: list[CheckResult]) -> str:
    frequencies = summarise_fault_frequency(results)
    if not frequencies:
        return "no faults detected"
    width = max(len(code) for code, _ in frequencies)
    lines = [f"{'fault':<{width}}  {'count':>6}  severity"]
    lines.append("-" * (width + 18))
    for code, count in frequencies:
        spec = FAULT_BY_CODE.get(code)
        lines.append(f"{code:<{width}}  {count:>6}  {spec.severity.value if spec else 'info'}")
    return "\n".join(lines)
