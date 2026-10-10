"""Deutsche Texte für jeden Fehlercode.

Die Regel-Engine liefert englische Meldungen mit Rohwerten. Hier wird daraus ein
Satz, den die Buchhaltung versteht: was falsch ist, welcher Wert gefunden wurde,
und was zu tun ist.
"""
from __future__ import annotations

from datetime import date, datetime
from decimal import Decimal, InvalidOperation
from typing import Any

from .taxonomy import FAULT_BY_CODE


# --- deutsche Formatierung ----------------------------------------------------------------------

def de_amount(value: Any, currency: str | None = None) -> str:
    """1234.5 -> '1.234,50', optional mit Währung."""
    if value in (None, ""):
        return "–"
    try:
        number = Decimal(str(value))
    except (InvalidOperation, TypeError):
        return str(value)
    text = f"{number:,.2f}".replace(",", "\u00a0").replace(".", ",").replace("\u00a0", ".")
    return f"{text} {currency}" if currency else text


def de_date(value: Any) -> str:
    """Datum oder ISO-String -> DD.MM.YYYY."""
    if not value:
        return "–"
    if isinstance(value, datetime):
        return value.strftime("%d.%m.%Y")
    if isinstance(value, date):
        return value.strftime("%d.%m.%Y")
    text = str(value)
    for fmt in ("%Y-%m-%d", "%d.%m.%Y"):
        try:
            return datetime.strptime(text, fmt).strftime("%d.%m.%Y")
        except ValueError:
            continue
    return text


def de_percent(value: Any) -> str:
    try:
        number = Decimal(str(value)) * 100
    except (InvalidOperation, TypeError):
        return str(value)
    text = f"{number.normalize():f}".rstrip("0").rstrip(".")
    return f"{text} %"


def de_quantity(value: Any) -> str:
    """4.0 -> '4', 7.5 -> '7,5'. Mengen sind keine Beträge."""
    if value in (None, ""):
        return "–"
    try:
        number = Decimal(str(value)).normalize()
    except (InvalidOperation, TypeError):
        return str(value)
    text = f"{number:f}"
    # Kein Tausenderpunkt: "1.200 Stunden" wäre missverständlich.
    return text.replace(".", ",") if "." in text else text


# --- Meldungen ----------------------------------------------------------------------------------

def explain(code: str, evidence: dict, *, severity: str = "", legal_reference: str = "") -> str:
    """Deutscher Satz zu einer Feststellung."""
    builder = _BUILDERS.get(code)
    if builder is not None:
        try:
            return builder(evidence or {})
        except Exception:
            # Formatierungsfehler dürfen die Feststellung nicht verschlucken.
            pass
    spec = FAULT_BY_CODE.get(code)
    generic = spec.description if spec else code
    return f"{generic} (Regelcode {code})."


def _money(evidence: dict, key: str, currency: str | None = None) -> str:
    return de_amount(evidence.get(key), currency or evidence.get("currency"))


# --- individual explanations --------------------------------------------------------------------

def _gross_mismatch(ev: dict) -> str:
    return (
        f"Nettobetrag ({_money(ev, 'net')}) zuzüglich Umsatzsteuer "
        f"({_money(ev, 'vat')}) ergeben {_money(ev, 'expected_gross')}, "
        f"angegeben ist jedoch {_money(ev, 'stated_gross')}. "
        f"Abweichung: {_money(ev, 'difference')}. "
        f"Bitte die Beträge abstimmen und die Rechnung korrigiert erneut senden."
    )


def _line_items_mismatch(ev: dict) -> str:
    return (
        f"Die Summe der Einzelpositionen ergibt {de_amount(ev.get('line_items_total'))}, "
        f"als Nettobetrag sind jedoch {de_amount(ev.get('stated_net'))} angegeben. "
        f"Bitte die Positionen oder den Nettobetrag korrigieren."
    )


def _vat_amount_wrong(ev: dict) -> str:
    return (
        f"Aus dem Nettobetrag {de_amount(ev.get('net'))} und dem Steuersatz "
        f"{de_percent(ev.get('rate'))} ergibt sich eine Umsatzsteuer von "
        f"{de_amount(ev.get('expected_vat'))}; angegeben ist {de_amount(ev.get('stated_vat'))}. "
        f"Bitte den Steuerbetrag korrigieren."
    )


def _line_amount_zero(ev: dict) -> str:
    return (
        f"Position {ev.get('line', '?')} „{ev.get('description') or 'ohne Bezeichnung'}“: "
        f"Es werden {de_quantity(ev.get('quantity'))} Einheiten berechnet, der Positionsbetrag ist "
        f"jedoch {de_amount(ev.get('line_net'))}. Eine erbrachte Leistung darf nicht mit 0,00 "
        f"berechnet werden. Bitte den Einzelpreis oder den Positionsbetrag ergänzen."
    )


def _line_unit_price_zero(ev: dict) -> str:
    return (
        f"Position {ev.get('line', '?')} „{ev.get('description') or 'ohne Bezeichnung'}“: "
        f"Bei {de_quantity(ev.get('quantity'))} Einheiten ist der Einzelpreis "
        f"{de_amount(ev.get('unit_price'))}. Bitte den Einzelpreis ergänzen."
    )


def _line_quantity_price(ev: dict) -> str:
    return (
        f"Position {ev.get('line', '?')} „{ev.get('description') or 'ohne Bezeichnung'}“: "
        f"{de_quantity(ev.get('quantity'))} × {de_amount(ev.get('unit_price'))} ergibt "
        f"{de_amount(ev.get('expected_net'))}, angegeben ist {de_amount(ev.get('stated_net'))}. "
        f"Bitte Menge, Einzelpreis oder Positionsbetrag korrigieren."
    )


def _line_rate_outlier(ev: dict) -> str:
    return (
        f"Position {ev.get('line', '?')} „{ev.get('description') or 'ohne Bezeichnung'}“ ist mit "
        f"{de_amount(ev.get('unit_price'))} je Einheit berechnet. Der übliche Satz für diese Leistung "
        f"liegt bei etwa {de_amount(ev.get('historical_median'))} "
        f"({float(ev.get('ratio') or 0):.0%} des üblichen Satzes). "
        f"Bitte prüfen, ob ein Tippfehler vorliegt."
    )


def _missing(field_de: str, hint: str = ""):
    def builder(_ev: dict) -> str:
        text = f"Die Rechnung enthält {field_de} nicht."
        return f"{text} {hint}".strip()
    return builder


def _invalid_vat_id(ev: dict) -> str:
    return (
        f"Die Umsatzsteuer-Identifikationsnummer {ev.get('vat_id')} ist rechnerisch ungültig "
        f"(Prüfziffer stimmt nicht). Bitte die Nummer korrigieren und die Rechnung erneut senden."
    )


def _invalid_iban_checksum(ev: dict) -> str:
    return (
        f"Die IBAN {ev.get('iban')} ist ungültig (Prüfsumme nach MOD-97 stimmt nicht). "
        f"Bitte die Bankverbindung prüfen."
    )


def _invalid_iban_format(ev: dict) -> str:
    expected = ev.get("expected")
    extra = f" Erwartet werden {expected} Zeichen, angegeben sind {ev.get('length')}." if expected else ""
    return f"Die IBAN {ev.get('iban')} hat eine unzulässige Länge oder Zeichenfolge.{extra}"


def _invoice_number_format(ev: dict) -> str:
    return (
        f"Die Rechnungsnummer „{ev.get('value')}“ entspricht nicht dem vereinbarten Aufbau "
        f"({ev.get('pattern')}). Bitte das Nummernformat prüfen."
    )


def _date_future(ev: dict) -> str:
    return (
        f"Das Rechnungsdatum {de_date(ev.get('invoice_date'))} liegt nach dem Verarbeitungsdatum "
        f"{de_date(ev.get('reference_date'))}. Ein Rechnungsdatum in der Zukunft ist unzulässig."
    )


def _date_delivery_after(ev: dict) -> str:
    return (
        f"Das Leistungsdatum {de_date(ev.get('service_date'))} liegt nach dem Rechnungsdatum "
        f"{de_date(ev.get('invoice_date'))}. Bitte die Datumsangaben korrigieren."
    )


def _date_due_before(ev: dict) -> str:
    return (
        f"Das Fälligkeitsdatum {de_date(ev.get('due_date'))} liegt vor dem Rechnungsdatum "
        f"{de_date(ev.get('invoice_date'))}. Bitte das Zahlungsziel korrigieren."
    )


def _date_term_unusual(ev: dict) -> str:
    return (
        f"Das Zahlungsziel von {ev.get('term_days')} Tagen ist ungewöhnlich "
        f"(zulässig sind bis zu {ev.get('max_days')} Tagen). Bitte das Zahlungsziel prüfen."
    )


def _reverse_charge(ev: dict) -> str:
    return (
        f"Bei einem steuerfreien innergemeinschaftlichen Umsatz (Steuersatz "
        f"{de_percent(ev.get('vat_rate'))}, Empfängerland {ev.get('customer_country')}) fehlt der "
        f"erforderliche Hinweis auf die Steuerschuldnerschaft des Leistungsempfängers "
        f"(Reverse-Charge-Verfahren, § 14a Abs. 5 UStG)."
    )


def _vat_rate_unexpected(ev: dict) -> str:
    permitted = ev.get("permitted") or []
    allowed = ", ".join(de_percent(rate) for rate in permitted)
    return (
        f"Der Steuersatz {de_percent(ev.get('rate'))} ist nicht zulässig. "
        f"Zulässig sind: {allowed}."
    )


def _small_amount(ev: dict) -> str:
    present = ev.get("present_fields") or []
    labels = {"net_amount": "Nettobetrag", "vat_rate": "Steuersatz", "vat_amount": "Steuerbetrag"}
    vorhanden = ", ".join(labels.get(name, name) for name in present)
    return (
        f"Bei einem Kleinbetrag von {de_amount(ev.get('gross'))} (Grenze "
        f"{de_amount(ev.get('limit'))}) dürfen die Steuerangaben entfallen. Es sind jedoch nur "
        f"folgende Angaben vorhanden: {vorhanden}. Bitte die Angaben vervollständigen oder streichen."
    )


def _unknown_supplier(ev: dict) -> str:
    return (
        f"Der Rechnungssteller „{ev.get('supplier')}“ ist im Kreditorenstamm nicht vorhanden. "
        f"Bitte den Lieferanten anlegen oder die Schreibweise prüfen."
    )


def _currency_mismatch(ev: dict) -> str:
    return (
        f"Die Rechnung ist in {ev.get('invoice_currency')} ausgestellt, vereinbart ist jedoch "
        f"{ev.get('contract_currency')}. Bitte die Währung abstimmen."
    )


def _unknown_currency(ev: dict) -> str:
    return (
        f"Die Währungsangabe „{ev.get('currency')}“ ist kein gültiger ISO-4217-Code. "
        f"Bitte die Währung korrigieren."
    )


def _negative_amount(ev: dict) -> str:
    return (
        f"Der Rechnungsbetrag {de_amount(ev.get('gross'))} ist negativ. "
        f"Für eine Ausgangsrechnung ist ein negativer Betrag unzulässig."
    )


def _above_threshold(ev: dict) -> str:
    centre = ev.get("cost_centre")
    extra = f" (Kostenstelle {centre})" if centre else ""
    return (
        f"Der Rechnungsbetrag {de_amount(ev.get('gross'))} überschreitet die "
        f"Genehmigungsgrenze von {de_amount(ev.get('threshold'))}{extra}. "
        f"Bitte vor dem Versand die Freigabe einholen."
    )


def _duplicate(ev: dict) -> str:
    return (
        f"Die Rechnungsnummer {ev.get('invoice_number')} wurde bereits für einen anderen Beleg "
        f"verwendet. Bitte prüfen, ob eine Doppelerfassung vorliegt."
    )


def _iban_changed(ev: dict) -> str:
    return (
        f"Die angegebene IBAN {ev.get('invoice_iban')} weicht von der im Stammdatensatz "
        f"hinterlegten IBAN {ev.get('master_iban')} ab. Bitte die Bankverbindung bestätigen lassen."
    )


def _amount_outlier(ev: dict) -> str:
    return (
        f"Der Rechnungsbetrag ist für diesen Lieferanten ungewöhnlich hoch "
        f"(Abweichung {ev.get('zscore', '?')} Standardabweichungen, das "
        f"{ev.get('ratio', '?')}-fache des Durchschnitts). Bitte auf Tippfehler prüfen."
    )


def _extraction_failed(ev: dict) -> str:
    return (
        "Aus diesem Beleg konnten keine Rechnungsdaten gelesen werden"
        + (f" (Verfahren: {ev.get('method')})" if ev.get("method") else "")
        + ". Der Beleg ist möglicherweise beschädigt, leer oder ein nicht unterstütztes Format. "
          "Bitte den Beleg manuell prüfen."
    )


def _llm_judgement(ev: dict) -> str:
    return (
        "Das Sprachmodell hat eine unklare oder ungewöhnliche Passage im Beleg gemeldet"
        + (f": {ev.get('note')}" if ev.get("note") else "")
        + ". Bitte den Beleg manuell prüfen; dies ist kein formaler Fehler."
    )


_BUILDERS = {
    "ARITHMETIC_GROSS_MISMATCH": _gross_mismatch,
    "ARITHMETIC_LINE_ITEMS_MISMATCH": _line_items_mismatch,
    "ARITHMETIC_VAT_AMOUNT_WRONG": _vat_amount_wrong,
    "LINE_AMOUNT_ZERO": _line_amount_zero,
    "LINE_UNIT_PRICE_ZERO": _line_unit_price_zero,
    "LINE_QUANTITY_PRICE_INCONSISTENT": _line_quantity_price,
    "LINE_RATE_OUTLIER": _line_rate_outlier,
    "MISSING_INVOICE_NUMBER":
        _missing("keine Rechnungsnummer",
                 "Nach § 14 Abs. 4 Nr. 4 UStG ist eine fortlaufende Rechnungsnummer Pflicht."),
    "MISSING_INVOICE_DATE":
        _missing("kein Rechnungsdatum",
                 "Nach § 14 Abs. 4 Nr. 3 UStG ist das Ausstellungsdatum Pflicht."),
    "MISSING_SUPPLIER_NAME":
        _missing("keine vollständige Absenderangabe (Name und Anschrift des Rechnungsstellers)",
                 "Nach § 14 Abs. 4 Nr. 1 UStG sind Name und Anschrift des leistenden "
                 "Unternehmers Pflicht."),
    "MISSING_SUPPLIER_VAT_ID":
        _missing("keine Umsatzsteuer-Identifikationsnummer und keine Steuernummer",
                 "Nach § 14 Abs. 4 Nr. 2 UStG ist die USt-IdNr. oder Steuernummer des "
                 "Rechnungsstellers Pflicht."),
    "MISSING_CUSTOMER_NAME":
        _missing("keine Empfängerangabe",
                 "Nach § 14 Abs. 4 Nr. 1 UStG ist der Name des Leistungsempfängers Pflicht."),
    "MISSING_NET_AMOUNT":
        _missing("keinen Nettobetrag",
                 "Bei Rechnungen über 250 EUR ist der Nettobetrag gesondert anzugeben."),
    "MISSING_VAT_RATE":
        _missing("keinen Steuersatz",
                 "Der anzuwendende Steuersatz ist gesondert anzugeben."),
    "MISSING_VAT_AMOUNT":
        _missing("keinen Steuerbetrag",
                 "Der auf den Nettobetrag entfallende Steuerbetrag ist gesondert anzugeben."),
    "MISSING_GROSS_AMOUNT":
        _missing("keinen Rechnungsbetrag (brutto)",
                 "Der Gesamtbetrag der Rechnung ist anzugeben."),
    "MISSING_PAYMENT_TERMS":
        _missing("kein Zahlungsziel und kein Fälligkeitsdatum",
                 "Nach § 14 Abs. 4 Nr. 5 UStG ist der Zeitpunkt der Zahlung anzugeben."),
    "MISSING_SERVICE_DATE":
        _missing("kein Leistungsdatum und keinen Leistungszeitraum",
                 "Nach § 14 Abs. 4 Nr. 6 UStG ist der Leistungszeitpunkt oder -zeitraum Pflicht."),
    "INVALID_VAT_ID_CHECKSUM": _invalid_vat_id,
    "INVALID_IBAN_CHECKSUM": _invalid_iban_checksum,
    "INVALID_IBAN_FORMAT": _invalid_iban_format,
    "INVALID_INVOICE_NUMBER_FORMAT": _invoice_number_format,
    "DATE_IN_FUTURE": _date_future,
    "DATE_DELIVERY_AFTER_INVOICE": _date_delivery_after,
    "DATE_DUE_BEFORE_INVOICE": _date_due_before,
    "DATE_DUE_TERM_UNUSUAL": _date_term_unusual,
    "REVERSE_CHARGE_NOTE_MISSING": _reverse_charge,
    "VAT_RATE_UNEXPECTED": _vat_rate_unexpected,
    "SMALL_AMOUNT_FIELDS_INCONSISTENT": _small_amount,
    "UNKNOWN_SUPPLIER": _unknown_supplier,
    "CURRENCY_MISMATCH": _currency_mismatch,
    "UNKNOWN_CURRENCY": _unknown_currency,
    "NEGATIVE_AMOUNT": _negative_amount,
    "AMOUNT_ABOVE_APPROVAL_THRESHOLD": _above_threshold,
    "DUPLICATE_INVOICE": _duplicate,
    "IBAN_CHANGED_RECENTLY": _iban_changed,
    "AMOUNT_OUTLIER": _amount_outlier,
    "EXTRACTION_FAILED": _extraction_failed,
    "LLM_JUDGEMENT_FLAG": _llm_judgement,
}

SEVERITY_DE = {
    "critical": "Kritisch",
    "major": "Schwerwiegend",
    "minor": "Geringfügig",
    "info": "Hinweis",
}

SEVERITY_ORDER_DE = {"critical": 0, "major": 1, "minor": 2, "info": 3}

SEVERITY_MEANING_DE = {
    "critical": "Die Rechnung ist in dieser Form nicht buchungsfähig und darf nicht versendet werden.",
    "major": "Die Rechnung muss vor dem Versand korrigiert werden.",
    "minor": "Formeller Mangel – bitte prüfen und ggf. korrigieren.",
    "info": "Auffälligkeit, die eine manuelle Prüfung empfiehlt.",
}
