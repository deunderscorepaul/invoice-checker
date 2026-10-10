"""Feldextraktion: LLM, Regex-Baseline und Umwandlung ins Schema.

Drei Wege, absichtlich:
  RegexExtractor  kein Modell, schnell, scheitert an ungewohnten Layouts
  LLMExtractor    lokales Modell mit JSON-Zwang
  extract_with_fallback  LLM, sonst Regex
Die Regex-Variante ist der ehrliche Vergleichsmaßstab für das Modell.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from decimal import Decimal

from .llm import LLMUnavailable, OllamaClient, SchemaError
from .models import Invoice, parse_date, parse_decimal

INVOICE_JSON_SCHEMA: dict = {
    "type": "object",
    "additionalProperties": False,
    # Alles, was eine Regel auf Vorhandensein prüft, steht hier als Pflichtfeld --
    # der JSON-Zwang zwingt das Modell dann, jedes Feld anzusehen. Leerer String
    # heißt weiterhin "steht nicht im Beleg" und wird zu None.
    "required": ["invoice_number", "invoice_date", "service_date", "supplier_name", "supplier_vat_id",
                 "customer_name", "customer_vat_id", "net_amount", "vat_rate", "vat_amount",
                 "gross_amount", "currency", "due_date", "iban"],
    # Typen bewusst einfach. Ein Union wie ["string", "null"] wäre die Lehrbuchlösung
    # für "fehlt", aber Ollamas Grammatik kam damit nicht klar: alle Felder kamen als
    # null zurück und die Laufzeit verdoppelte sich. "Fehlt" ist deshalb der leere
    # String (siehe models._text).
    "properties": {
        "invoice_number": {"type": "string"},
        "invoice_date": {"type": "string", "format": "date"},
        "service_date": {"type": "string", "format": "date"},
        "supplier_name": {"type": "string"},
        "supplier_address": {"type": "string"},
        "supplier_vat_id": {"type": "string"},
        "customer_name": {"type": "string"},
        "customer_vat_id": {"type": "string"},
        "net_amount": {"type": "number"},
        "vat_rate": {"type": "number"},
        "vat_amount": {"type": "number"},
        "gross_amount": {"type": "number"},
        "currency": {"type": "string"},
        "payment_terms_days": {"type": "integer"},
        "due_date": {"type": "string", "format": "date"},
        "iban": {"type": "string"},
        "purchase_order": {"type": "string"},
        "cost_centre": {"type": "string"},
        "reverse_charge_note": {"type": "string"},
        "line_items": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["description", "quantity", "unit_price", "line_net"],
                "properties": {
                    "description": {"type": "string"},
                    "quantity": {"type": "number"},
                    "unit_price": {"type": "number"},
                    "line_net": {"type": "number"},
                },
            },
        },
    },
}

SYSTEM_PROMPT = (
    "You extract structured fields from outgoing invoices. "
    "Copy values exactly as they appear in the document. Do not compute, infer or correct any "
    "number: if the document states a value, report that value even when it looks wrong. "
    "Amounts are decimal numbers without currency symbols. Dates use YYYY-MM-DD. "
    "Work through every field of the schema before answering, and prefer extracting a value over "
    "leaving it out: most invoices state the recipient, the service date and both VAT numbers. "
    "Use an empty string only when the document genuinely does not contain that field. Never guess, "
    "invent or compute a value, and never carry a number over from a different field: reporting a "
    "plausible amount the document does not state is worse than an empty string, because it hides a "
    "real defect on the invoice."
)

# Achtung, das Beispiel ist eine Vorgabe: was hier leer bleibt, lässt das Modell
# auch in echten Belegen weg. Eine frühere Version hatte "supplier_name": "" und
# kein service_date/customer_vat_id -- das Modell hat das brav übernommen und die
# Regeln meldeten auf allen Testrechnungen fehlende Kundenanschrift.
FEW_SHOT = """Example document:
Rechnung Nr. RE-2024-0011
Rechnungsdatum: 14.03.2024
Leistungsdatum: 11.03.2024
Umsatzsteuer-Identifikationsnummer: DE123456788
Rechnungsempfaenger: Beispiel Handel GmbH
Anschrift: Musterstrasse 5, 10115 Berlin
USt-IdNr. Empfaenger: DE811907980
Gesamtbetrag netto / Net total:            1.000,00
Umsatzsteuer 19% / VAT:              190,00
Rechnungsbetrag brutto / Gross total:      1.190,00 EUR
Zahlungsbedingungen: zahlbar innerhalb von 30 Tagen bis zum 13.04.2024
Bankverbindung / IBAN: DE02120300000000202051
Bestellnummer / PO: PO-445566
Kostenstelle: CC-100

Example answer:
{"invoice_number": "RE-2024-0011", "invoice_date": "2024-03-14", "service_date": "2024-03-11",
 "supplier_name": "Rechnungssteller GmbH", "supplier_address": "Beispielweg 1, 20095 Hamburg",
 "supplier_vat_id": "DE123456788", "customer_name": "Beispiel Handel GmbH",
 "customer_address": "Musterstrasse 5, 10115 Berlin", "customer_vat_id": "DE811907980",
 "net_amount": 1000.00, "vat_rate": 0.19, "vat_amount": 190.00, "gross_amount": 1190.00,
 "currency": "EUR", "payment_terms_days": 30, "due_date": "2024-04-13",
 "iban": "DE02120300000000202051", "purchase_order": "PO-445566", "cost_centre": "CC-100",
 "reverse_charge_note": "", "line_items": []}

Note: a field the document does not contain is answered as an empty string, never as a guessed value.

Now extract the real document. Answer with JSON only."""


@dataclass
class ExtractionOutcome:
    invoice: Invoice
    method: str
    ok: bool
    warnings: list[str] = field(default_factory=list)
    raw_response: str = ""


# --- label vocabularies --------------------------------------------------------------------------

LABELS = {
    "invoice_number": [r"Rechnung\s*(?:s-)?\s*(?:Nr|Nummer|nummer)\.?", r"Rechnungsnummer",
                       r"Invoice\s*(?:No|Number|Nr)\.?", r"Beleg(?:nummer)?"],
    "invoice_date": [r"Rechnungsdatum", r"Datum\s*der\s*Rechnung", r"Invoice\s*date", r"Rechnungsstellung"],
    "service_date": [r"Leistungsdatum", r"Lieferdatum", r"Date\s*of\s*supply", r"Delivery\s*date",
                     r"Leistungstag"],
    "supplier_vat_id": [r"Umsatzsteuer-?Identifikationsnummer", r"USt-?IdNr\.?", r"USt-IdNr",
                        r"VAT\s*(?:identification\s*)?(?:number|no|id)", r"UID"],
    "customer_name": [r"Rechnungsempf(?:ae|ä)nger", r"Empf(?:ae|ä)nger", r"Bill\s*to", r"Kunde",
                      r"Customer"],
    "customer_vat_id": [r"USt-?IdNr\.?\s*Empf(?:ae|ä)nger", r"Customer\s*VAT"],
    "net_amount": [r"Gesamtbetrag\s*netto", r"Netto(?:summe|betrag)", r"Net\s*total",
                   r"Total\s*net", r"Zwischensumme"],
    "vat_amount": [r"Umsatzsteuer(?!-?Ident)", r"MwSt\.?", r"VAT\s*amount", r"Mehrwertsteuer"],
    "gross_amount": [r"Rechnungsbetrag\s*brutto", r"Brutto(?:betrag|summe)", r"Gross\s*total",
                     r"Total\s*due", r"Gesamtbetrag\s*brutto", r"Rechnungsbetrag", r"Endbetrag"],
    "vat_rate": [r"Umsatzsteuer\s*(\d{1,2}(?:[.,]\d)?)\s*%"],
    "payment_terms": [r"zahlbar\s*innerhalb\s*von\s*(\d{1,3})\s*Tagen",
                      r"Zahlungsziel\s*:?\s*(\d{1,3})\s*Tage",
                      r"payable\s*within\s*(\d{1,3})\s*days",
                      r"payment\s*terms?\s*:?\s*(\d{1,3})\s*days"],
    "due_date": [r"bis\s*zum\s*", r"f(?:ae|ä)llig\s*am", r"due\s*(?:on|date)\s*:?\s*"],
    "iban": [r"IBAN", r"Bankverbindung"],
    "purchase_order": [r"Bestellnummer", r"Bestell-?Nr\.?", r"PO\s*(?:number|no)?", r"Auftragsnummer"],
    "cost_centre": [r"Kostenstelle", r"Cost\s*cent(?:re|er)"],
    "reverse_charge": [r"Reverse\s*Charge", r"Steuerschuldnerschaft",
                       r"Steuerschuld\s*des\s*Leistungsempf"],
}

DATE_PATTERNS = [
    re.compile(r"\b(\d{2}\.\d{2}\.\d{4})\b"),
    re.compile(r"\b(\d{4}-\d{2}-\d{2})\b"),
    re.compile(r"\b(\d{2}/\d{2}/\d{4})\b"),
]

# Reihenfolge ist wichtig: die vollständige Variante muss zuerst stehen. Sonst
# matcht die kurze Alternative nur den ganzzahligen Teil von "6,456.69" und die
# Nachkommastellen fallen weg.
AMOUNT_PATTERN = re.compile(
    r"(-?\d{1,3}(?:[.,\u00a0 ]\d{3})+[.,]\d{2})"   # grouped: 1.234,56 | 1,234.56
    r"|(-?\d+[.,]\d{2})"                            # plain decimal: 1234.56
    r"|(-?\d{1,3}(?:[.,\u00a0 ]\d{3})+)?"           # optional captured integer group
)
PERCENT_AFTER = re.compile(r"^\s*%")
IBAN_PATTERN = re.compile(r"\b([A-Z]{2}\d{2}[A-Z0-9]{10,30})\b")
# Reihenfolge der Alternativen: deutsche ID zuerst, sonst greift der allgemeine
# Zweig und schneidet Ziffern ab.
VAT_ID_PATTERN = re.compile(r"\b(DE\s?\d{9}|[A-Z]{2}\s?[A-Z0-9]{8,12})\b")


def _label_regex(label_key: str) -> re.Pattern:
    alternatives = "|".join(f"(?:{pattern})" for pattern in LABELS[label_key])
    return re.compile(rf"(?:{alternatives})\s*[:\-]?\s*(.+)$", re.IGNORECASE | re.MULTILINE)


def find_amount_after(text: str, position: int, window: int = 160) -> Decimal | None:
    """Return the most plausible amount following a label.

    Two heuristics matter here and both come from real invoice layouts:

    * take the *last* number on the line, because "Netto 1.000,00" style layouts put the value
      after the label and before any trailing currency;
    * never take a number immediately followed by ``%`` -- on "Umsatzsteuer 19% / VAT: 190,00" the
      rate precedes the amount, and treating the rate as the amount silently corrupts the document.
    """
    segment = text[position:position + window]
    line = segment.split("\n")[0]
    candidates = list(_iter_amounts(line))
    if not candidates:
        candidates = list(_iter_amounts(segment))
    for value in reversed(candidates):
        return value
    return None


def _iter_amounts(text: str):
    for match in AMOUNT_PATTERN.finditer(text):
        raw = next((group for group in match.groups() if group), None)
        if not raw:
            continue
        if PERCENT_AFTER.match(text[match.end():match.end() + 2]):
            continue
        value = parse_decimal(raw)
        if value is not None:
            yield value


# One table row: leading position number, description, quantity, unit price, line total.
#
# Von hinten aufzäumen: die letzten drei Felder sind eindeutig numerisch, damit
# muss man nicht raten wo die Bezeichnung endet.
#
# MONEY: die Tausender-Variante zuerst. Sonst zerlegt die Regex "3,130.40" in
# "130.40" und hängt das "3," an die Bezeichnung -- die erste Tabellenzeile fehlt.
MONEY = r"-?\d{1,3}(?:[.,\u00a0 ]\d{3})+[.,]\d{2}|-?\d+[.,]\d{2}"
QUANTITY = r"-?\d+(?:[.,]\d+)?"
LINE_ITEM_ROW = re.compile(
    rf"^\s*(\d{{1,3}})\s+"                                # position
    rf"([^\s,].*?)\s+"                                    # description: never ends in a separator
    rf"({QUANTITY})\s+"                                   # quantity
    rf"({MONEY})\s+"                                      # unit price
    rf"({MONEY})\s*$"                                     # line net
)


def extract_line_items(text: str) -> list[dict]:
    """Parse positional invoice table rows into line items.

    A regex parser has no notion of columns, so it has to guess the row shape from whitespace. It
    gets the common case right and fails whenever the template changes, which is the point of having
    it as a baseline for comparison against a model.
    """
    items: list[dict] = []
    for line in text.splitlines():
        match = LINE_ITEM_ROW.match(line)
        if not match:
            continue
        position, description, quantity, unit_price, line_net = match.groups()
        parsed_quantity = parse_decimal(quantity)
        parsed_price = parse_decimal(unit_price)
        parsed_net = parse_decimal(line_net)
        if None in (parsed_quantity, parsed_price, parsed_net):
            continue
        items.append({
            "description": description.strip(),
            "quantity": float(parsed_quantity),
            "unit_price": float(parsed_price),
            "line_net": float(parsed_net),
        })
    return items


class RegexExtractor:
    """Hand-written baseline. Fast, explainable, and defeated by layout noise -- which is the point."""

    name = "regex"

    def extract(self, text: str) -> ExtractionOutcome:
        warnings: list[str] = []
        data: dict = {}

        for key in ("invoice_number", "customer_name", "purchase_order", "cost_centre"):
            match = _label_regex(key).search(text)
            if match:
                data[key] = match.group(1).strip(" :;\t").split("  ")[0].strip()

        for key in ("invoice_date", "service_date"):
            match = _label_regex(key).search(text)
            if match:
                for pattern in DATE_PATTERNS:
                    found = pattern.search(match.group(1))
                    if found:
                        data[key] = found.group(1)
                        break

        match = _label_regex("supplier_vat_id").search(text)
        if match:
            found = VAT_ID_PATTERN.search(match.group(1))
            data["supplier_vat_id"] = found.group(1) if found else match.group(1).strip()
        else:
            found = VAT_ID_PATTERN.search(text)
            if found:
                data["supplier_vat_id"] = found.group(1)

        for key in ("net_amount", "vat_amount", "gross_amount"):
            match = _label_regex(key).search(text)
            if match:
                amount = find_amount_after(text, match.start())
                if amount is not None:
                    data[key] = float(amount)

        rate_match = _label_regex("vat_rate").search(text)
        if rate_match:
            rate = parse_decimal(rate_match.group(1))
            if rate is not None:
                data["vat_rate"] = float(rate / 100)

        terms_match = _label_regex("payment_terms").search(text)
        if terms_match:
            digits = re.search(r"\d{1,3}", terms_match.group(1))
            if digits:
                data["payment_terms_days"] = int(digits.group(0))

        due_match = _label_regex("due_date").search(text)
        if due_match:
            for pattern in DATE_PATTERNS:
                found = pattern.search(due_match.group(1))
                if found:
                    data["due_date"] = found.group(1)
                    break

        iban_match = _label_regex("iban").search(text)
        if iban_match:
            found = IBAN_PATTERN.search(iban_match.group(1))
            data["iban"] = found.group(1) if found else ""
        else:
            found = IBAN_PATTERN.search(text)
            if found:
                data["iban"] = found.group(1)

        if re.search("|".join(LABELS["reverse_charge"]), text, re.IGNORECASE):
            data["reverse_charge_note"] = "Steuerschuldnerschaft des Leistungsempfaengers"

        currency_match = re.search(r"\b(EUR|USD|GBP|CHF|SEK|NOK|DKK|PLN|CZK|HUF)\b", text)
        if currency_match:
            data["currency"] = currency_match.group(1)

        # The first non-empty line of a commercial letter is the issuer block.
        for line in text.splitlines():
            stripped = line.strip()
            if stripped:
                data.setdefault("supplier_name", stripped)
                break

        data["line_items"] = extract_line_items(text) or None
        if not data["line_items"]:
            data.pop("line_items", None)

        invoice = Invoice.from_dict(data)
        missing = [name for name in ("invoice_number", "invoice_date", "gross_amount")
                   if getattr(invoice, name) in (None, "")]
        if missing:
            warnings.append(f"regex extractor could not find: {', '.join(missing)}")
        return ExtractionOutcome(invoice=invoice, method=self.name, ok=not missing,
                                 warnings=warnings)


class LLMExtractor:
    name = "llm"

    def __init__(self, client: OllamaClient, *, use_few_shot: bool = True, max_chars: int = 6000):
        self.client = client
        self.use_few_shot = use_few_shot
        self.max_chars = max_chars

    def build_prompt(self, text: str) -> str:
        document = text.strip()
        if len(document) > self.max_chars:
            document = document[:self.max_chars]
        prefix = FEW_SHOT + "\n\n" if self.use_few_shot else ""
        return f"{prefix}Document:\n{document}\n\nJSON:"

    def extract(self, text: str, *, on_progress=None) -> ExtractionOutcome:
        try:
            payload, raw = self.client.generate_json(
                self.build_prompt(text), SYSTEM_PROMPT, INVOICE_JSON_SCHEMA,
                on_progress=on_progress,
            )
        except SchemaError as exc:
            return ExtractionOutcome(
                invoice=Invoice(), method=self.name, ok=False,
                warnings=[f"schema validation failed after retry: {exc}"],
            )
        invoice = Invoice.from_dict(payload)
        warnings: list[str] = []
        if not invoice.gross_amount and not invoice.net_amount:
            warnings.append("model returned no monetary amounts")
        return ExtractionOutcome(invoice=invoice, method=self.name, ok=not warnings,
                                 warnings=warnings, raw_response=raw)


def extract_with_fallback(text: str, client: OllamaClient | None, *,
                          strategy: str = "llm", on_progress=None) -> ExtractionOutcome:
    """``strategy`` is one of ``llm``, ``regex``, ``llm-then-regex``."""
    regex = RegexExtractor()
    if strategy == "regex" or client is None:
        return regex.extract(text)
    if strategy == "llm":
        return LLMExtractor(client).extract(text, on_progress=on_progress)
    if strategy == "llm-then-regex":
        try:
            outcome = LLMExtractor(client).extract(text, on_progress=on_progress)
        except LLMUnavailable:
            outcome = regex.extract(text)
            outcome.warnings.append("Ollama unavailable; fell back to the regex extractor")
            outcome.method = "regex(fallback)"
            return outcome
        if outcome.ok:
            return outcome
        fallback = regex.extract(text)
        merged = Invoice.from_dict({**fallback.invoice.to_dict(), **{
            key: value for key, value in outcome.invoice.to_dict().items() if value not in (None, "", [])
        }})
        return ExtractionOutcome(
            invoice=merged, method="llm+regex", ok=True,
            warnings=outcome.warnings + ["merged regex values for fields the model missed"],
        )
    raise ValueError(f"unknown extraction strategy: {strategy}")
