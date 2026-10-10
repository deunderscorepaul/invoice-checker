"""Der deutsche Prüfbericht zu einer abgelehnten Rechnung.

Reihenfolge im Text: Ergebnis, dann jede Feststellung mit den beanstandeten Werten,
dann was zu tun ist, dann die Rechtsgrundlage. Reiner Text, damit er überall
aufgeht -- Notepad, Terminal, Ausdruck.
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path

from .i18n_de import (SEVERITY_MEANING_DE, SEVERITY_ORDER_DE, de_date, explain)
from .models import CheckResult, Invoice, Fault
from .taxonomy import FAULT_BY_CODE

LINE = "=" * 78
THIN = "-" * 78


def _wrap(text: str, width: int = 74, indent: str = "  ") -> list[str]:
    """Einfacher Wortumbruch, damit der Bericht auch im schmalen Fenster lesbar bleibt."""
    words = text.split()
    lines: list[str] = []
    current = ""
    for word in words:
        candidate = f"{current} {word}".strip()
        if len(candidate) > width and current:
            lines.append(indent + current)
            current = word
        else:
            current = candidate
    if current:
        lines.append(indent + current)
    return lines or [indent + text]


def _field(label: str, value: str, width: int = 26) -> str:
    return f"  {label + ':':<{width}}{value}"


def render_report(result: CheckResult, invoice: Invoice, *, source_name: str,
                  folder_label: str = "", extraction: str = "",
                  processing_seconds: float | None = None) -> str:
    """Berichtstext für einen Beleg."""
    ordered = sorted(result.faults,
                     key=lambda f: (SEVERITY_ORDER_DE.get(f.severity, 9), f.code))
    worst = ordered[0].severity if ordered else "info"

    lines: list[str] = [LINE, "PRÜFBERICHT AUSGANGSRECHNUNG", LINE, ""]
    lines.append(f"  Beleg:            {source_name}")
    if folder_label:
        lines.append(f"  Eingangsordner:   {folder_label}")
    lines.append(f"  Geprüft am:       {datetime.now().strftime('%d.%m.%Y um %H:%M:%S')}")
    if extraction:
        lines.append(f"  Leseverfahren:    {extraction}")
    if processing_seconds is not None:
        lines.append(f"  Prüfdauer:        {processing_seconds:.1f} Sekunden")
    lines.append("")

    if not ordered:
        lines.extend([
            THIN,
            "ERGEBNIS: KEINE FEHLER GEFUNDEN",
            THIN,
            "",
            "  Die Rechnung wurde geprüft und entspricht den hinterlegten Regeln.",
            "  Es sind keine Maßnahmen erforderlich.",
            "",
        ])
        return "\n".join(lines) + "\n" + LINE + "\n"

    lines.extend([
        THIN,
        f"ERGEBNIS: FEHLERHAFT – {len(ordered)} Feststellung(en)",
        THIN,
        "",
        f"  Schweregrad: {_severity_label(worst)}",
        f"  {SEVERITY_MEANING_DE.get(worst, '')}",
        "",
    ])

    # Übersicht zuerst, damit man das Ganze sieht, bevor die Details kommen.
    lines.append("  Übersicht der Feststellungen:")
    lines.append("")
    for index, fault in enumerate(ordered, start=1):
        label = _severity_label(fault.severity)
        lines.append(f"    {index:>2}. [{label:<14}] {fault.code}")
    lines.append("")

    for index, fault in enumerate(ordered, start=1):
        lines.extend(_render_fault(index, fault))
    return "\n".join(lines) + "\n" + LINE + "\n"


def _severity_label(severity: str) -> str:
    from .i18n_de import SEVERITY_DE
    return SEVERITY_DE.get(severity, severity)


def _render_fault(index: int, fault: Fault) -> list[str]:
    spec = FAULT_BY_CODE.get(fault.code)
    out = [THIN, f"  {index}. {fault.code}  ({_severity_label(fault.severity)})", ""]
    out.extend(_wrap(explain(fault.code, fault.evidence, severity=fault.severity,
                             legal_reference=fault.legal_reference or "")))
    out.append("")

    evidence_lines = _evidence_lines(fault, code=fault.code)
    if evidence_lines:
        out.append("  Festgestellte Werte:")
        out.extend(evidence_lines)
        out.append("")

    if spec and spec.legal_reference:
        out.append(f"  Rechtsgrundlage: {_legal_de(spec.legal_reference)}")
        out.append("")
    return out


# Fields that are internal bookkeeping rather than invoice values, and German labels for the rest.
_EVIDENCE_LABELS = {
    "net": "Nettobetrag", "vat": "Umsatzsteuer", "expected_gross": "erwarteter Bruttobetrag",
    "stated_gross": "angegebener Bruttobetrag", "difference": "Differenz",
    "line_items_total": "Summe der Positionen", "stated_net": "angegebener Nettobetrag",
    "expected_vat": "erwartete Umsatzsteuer", "stated_vat": "angegebene Umsatzsteuer",
    "rate": "Steuersatz", "line": "Position", "description": "Bezeichnung",
    "quantity": "Menge", "unit_price": "Einzelpreis", "line_net": "Positionsbetrag",
    "expected_net": "erwarteter Positionsbetrag", "historical_median": "üblicher Einzelpreis",
    "ratio": "Verhältnis zum üblichen Satz", "vat_id": "USt-IdNr.", "iban": "IBAN",
    "expected": "erwartete Länge", "length": "angegebene Länge", "value": "Wert",
    "pattern": "erwartetes Format", "invoice_date": "Rechnungsdatum",
    "reference_date": "Verarbeitungsdatum", "service_date": "Leistungsdatum",
    "due_date": "Fälligkeitsdatum", "term_days": "Zahlungsziel (Tage)",
    "max_days": "zulässiges Maximum (Tage)", "customer_country": "Empfängerland",
    "permitted": "zulässige Sätze", "gross": "Bruttobetrag", "limit": "Kleinbetragsgrenze",
    "present_fields": "vorhandene Angaben", "supplier": "Rechnungssteller",
    "invoice_currency": "Rechnungswährung", "contract_currency": "vereinbarte Währung",
    "currency": "Währung", "threshold": "Genehmigungsgrenze", "cost_centre": "Kostenstelle",
    "invoice_number": "Rechnungsnummer", "invoice_iban": "angegebene IBAN",
    "master_iban": "hinterlegte IBAN", "zscore": "Standardabweichungen",
    "mean": "Durchschnitt des Lieferanten", "method": "Leseverfahren",
    "matched_document": "bereits erfasster Beleg",
}

# Interne Schlüssel, die im Bericht nur stören. 'currency' fehlt bewusst: die
# Währung steht schon an jedem Betrag.
_HIDDEN_EVIDENCE = {"mod97", "reason", "detector", "currency"}


def _evidence_lines(fault: Fault, code: str = "") -> list[str]:
    rows: list[str] = []
    # Beträge mit Währung, sonst muss man raten ob "1.915,94" EUR sind.
    currency = str(fault.evidence.get("currency") or "").strip()
    if not currency:
        currency = str(fault.evidence.get("invoice_currency") or "").strip()
    currency = currency or None
    for key, value in fault.evidence.items():
        if key in _HIDDEN_EVIDENCE or value in (None, "", []):
            continue
        label = _evidence_label(key, code)
        rows.append(_field(label, _format_value(key, value, currency), width=30))
    return rows


# Derselbe Schlüssel bedeutet je Regel etwas anderes: 'stated_net' ist bei
# ARITHMETIC_LINE_ITEMS_MISMATCH der Rechnungs-Netto, bei
# LINE_QUANTITY_PRICE_INCONSISTENT der Positionsbetrag. Beides "Nettobetrag" zu
# nennen schickt den Leser zur falschen Zahl.
_EVIDENCE_LABEL_OVERRIDES = {
    ("LINE_QUANTITY_PRICE_INCONSISTENT", "stated_net"): "angegebener Positionsbetrag",
    ("LINE_QUANTITY_PRICE_INCONSISTENT", "expected_net"): "erwarteter Positionsbetrag",
    ("LINE_AMOUNT_ZERO", "quantity"): "Menge",
    ("LINE_UNIT_PRICE_ZERO", "quantity"): "Menge",
}


def _evidence_label(key: str, code: str) -> str:
    override = _EVIDENCE_LABEL_OVERRIDES.get((code, key))
    return override or _EVIDENCE_LABELS.get(key, key)


def _format_value(key: str, value, currency: str | None) -> str:
    if key in {"ratio"}:
        try:
            return f"{float(value):.0%}"
        except (TypeError, ValueError):
            return str(value)
    if key == "quantity":
        from .i18n_de import de_quantity
        return de_quantity(value)
    if key == "permitted" and isinstance(value, (list, tuple)):
        from .i18n_de import de_percent
        return ", ".join(de_percent(item) for item in value)
    if key == "present_fields" and isinstance(value, (list, tuple)):
        labels = {"net_amount": "Nettobetrag", "vat_rate": "Steuersatz",
                  "vat_amount": "Steuerbetrag"}
        return ", ".join(labels.get(str(item), str(item)) for item in value)
    if key == "rate":
        from .i18n_de import de_percent
        return de_percent(value)
    if key.endswith("_date") or key in {"reference_date"}:
        return de_date(value)
    if key in {"net", "vat", "expected_gross", "stated_gross", "difference",
               "line_items_total", "stated_net", "expected_vat", "stated_vat",
               "unit_price", "line_net", "expected_net", "historical_median",
               "gross", "limit", "threshold", "mean"}:
        from .i18n_de import de_amount
        return de_amount(value, currency)
    return str(value)


def _legal_de(reference: str) -> str:
    """'UStG s14(4) Nr.4' -> '§ 14 Abs. 4 Nr. 4 UStG'."""
    import re

    match = re.match(r"^(UStG|UStDV)\s+(?:s|§)\s*([0-9]+[a-z]?)(?:\s*\((\d+)\))?\s*(.*)$", reference)
    if not match:
        return reference
    law, section, subsection, rest = match.groups()
    text = f"§ {section}"
    if subsection:
        text += f" Abs. {subsection}"
    if rest.strip():
        # Die Regeltabelle schreibt "Nr.4", deutsch ist "Nr. 4".
        cleaned = re.sub(r"\b(Nr|Abs|Satz)\.\s*", r"\1. ", rest.strip())
        text += f" {cleaned}"
    return f"{text} {law}"


def report_filename(original_name: str) -> str:
    """'rechnung-42.pdf' -> 'rechnung-42.fehlerbericht.txt'."""
    stem = Path(original_name).stem
    return f"{stem}.fehlerbericht.txt"
