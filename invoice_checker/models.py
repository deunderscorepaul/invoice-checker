"""Canonical invoice schema plus JSON coercion.

Deliberately dependency-free: no pydantic. ``Invoice.from_dict`` is tolerant (a vision model or an
OCR layer will hand us strings where we want numbers) but never silently invents a value -- anything
it cannot parse becomes ``None`` and is surfaced as an extraction failure by the caller.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field, asdict
from datetime import date, datetime
from decimal import Decimal, InvalidOperation
from typing import Any, Iterable

ISO_DATE_FORMATS = (
    "%Y-%m-%d", "%d.%m.%Y", "%d/%m/%Y", "%Y/%m/%d", "%d-%m-%Y", "%Y%m%d", "%d.%m.%y",
)

CURRENCY_RE = re.compile(r"^[A-Z]{3}$")

FIELD_NAMES = (
    "invoice_number", "invoice_date", "service_date", "service_period_start", "service_period_end",
    "supplier_name", "supplier_address", "supplier_vat_id", "supplier_tax_number",
    "customer_name", "customer_address", "customer_vat_id",
    "net_amount", "vat_rate", "vat_amount", "gross_amount", "currency",
    "payment_terms_days", "due_date", "iban", "bic", "purchase_order", "cost_centre",
    "reverse_charge_note", "line_items",
)


def parse_date(value: Any) -> date | None:
    if value is None or value == "":
        return None
    if isinstance(value, date) and not isinstance(value, datetime):
        return value
    if isinstance(value, datetime):
        return value.date()
    text = str(value).strip()
    if not text:
        return None
    for fmt in ISO_DATE_FORMATS:
        try:
            return datetime.strptime(text, fmt).date()
        except ValueError:
            continue
    return None


def parse_decimal(value: Any) -> Decimal | None:
    """Parse a money-ish value. Handles '1.234,56', '1,234.56', '€ 1234.56', '(123.45)'."""
    if value is None or value == "":
        return None
    if isinstance(value, Decimal):
        return value
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        try:
            return Decimal(str(value))
        except InvalidOperation:
            return None
    text = str(value).strip()
    if not text:
        return None
    negative = text.startswith("(") and text.endswith(")") or text.startswith("-")
    cleaned = re.sub(r"[^0-9,.\-]", "", text)
    if not cleaned or cleaned in {"-", ".", ","}:
        return None
    if "," in cleaned and "." in cleaned:
        # Whichever separator appears last is the decimal separator.
        if cleaned.rfind(",") > cleaned.rfind("."):
            cleaned = cleaned.replace(".", "").replace(",", ".")
        else:
            cleaned = cleaned.replace(",", "")
    elif "," in cleaned:
        head, _, tail = cleaned.rpartition(",")
        # '1,234' is ambiguous; treat 3 trailing digits with no other comma as a thousands group.
        if len(tail) == 3 and head and "," not in head and len(head) <= 3:
            cleaned = cleaned.replace(",", "")
        else:
            cleaned = cleaned.replace(",", ".")
    try:
        result = Decimal(cleaned)
    except InvalidOperation:
        return None
    return -result if negative and result > 0 else result


def money(value: Decimal | None) -> Decimal | None:
    if value is None:
        return None
    return value.quantize(Decimal("0.01"))


@dataclass
class LineItem:
    description: str = ""
    quantity: Decimal | None = None
    unit_price: Decimal | None = None
    line_net: Decimal | None = None

    @classmethod
    def from_dict(cls, raw: dict) -> "LineItem":
        return cls(
            description=str(raw.get("description") or "").strip(),
            quantity=parse_decimal(raw.get("quantity")),
            unit_price=parse_decimal(raw.get("unit_price")),
            line_net=parse_decimal(raw.get("line_net")),
        )

    def to_dict(self) -> dict:
        return {
            "description": self.description,
            "quantity": _num(self.quantity),
            "unit_price": _num(self.unit_price),
            "line_net": _num(self.line_net),
        }


def _num(value: Decimal | None) -> float | None:
    return None if value is None else float(value)


@dataclass
class Invoice:
    invoice_number: str | None = None
    invoice_date: date | None = None
    service_date: date | None = None
    service_period_start: date | None = None
    service_period_end: date | None = None
    supplier_name: str | None = None
    supplier_address: str | None = None
    supplier_vat_id: str | None = None
    supplier_tax_number: str | None = None
    customer_name: str | None = None
    customer_address: str | None = None
    customer_vat_id: str | None = None
    net_amount: Decimal | None = None
    vat_rate: Decimal | None = None
    vat_amount: Decimal | None = None
    gross_amount: Decimal | None = None
    currency: str | None = None
    payment_terms_days: int | None = None
    due_date: date | None = None
    iban: str | None = None
    bic: str | None = None
    purchase_order: str | None = None
    cost_centre: str | None = None
    reverse_charge_note: str | None = None
    line_items: list[LineItem] = field(default_factory=list)

    # --- construction ---------------------------------------------------------------------------
    @classmethod
    def from_dict(cls, raw: dict) -> "Invoice":
        data = {k: v for k, v in (raw or {}).items() if k in FIELD_NAMES}
        items_raw = data.pop("line_items", None) or []
        invoice = cls(
            invoice_number=_text(data.get("invoice_number")),
            invoice_date=parse_date(data.get("invoice_date")),
            service_date=parse_date(data.get("service_date")),
            service_period_start=parse_date(data.get("service_period_start")),
            service_period_end=parse_date(data.get("service_period_end")),
            supplier_name=_text(data.get("supplier_name")),
            supplier_address=_text(data.get("supplier_address")),
            supplier_vat_id=_text(data.get("supplier_vat_id")),
            supplier_tax_number=_text(data.get("supplier_tax_number")),
            customer_name=_text(data.get("customer_name")),
            customer_address=_text(data.get("customer_address")),
            customer_vat_id=_text(data.get("customer_vat_id")),
            net_amount=parse_decimal(data.get("net_amount")),
            vat_rate=parse_decimal(data.get("vat_rate")),
            vat_amount=parse_decimal(data.get("vat_amount")),
            gross_amount=parse_decimal(data.get("gross_amount")),
            currency=_text(data.get("currency")),
            payment_terms_days=_int(data.get("payment_terms_days")),
            due_date=parse_date(data.get("due_date")),
            iban=_norm_iban(data.get("iban")),
            bic=_text(data.get("bic")),
            purchase_order=_text(data.get("purchase_order")),
            cost_centre=_text(data.get("cost_centre")),
            reverse_charge_note=_text(data.get("reverse_charge_note")),
            line_items=[LineItem.from_dict(i) for i in items_raw if isinstance(i, dict)],
        )
        if invoice.currency:
            invoice.currency = invoice.currency.strip().upper()
        if invoice.vat_rate is not None and invoice.vat_rate > 1:
            # A model may answer "19" for 19%; normalise to a fraction.
            invoice.vat_rate = invoice.vat_rate / Decimal(100)
        return invoice

    def to_dict(self) -> dict:
        out: dict[str, Any] = {}
        for name in FIELD_NAMES:
            value = getattr(self, name)
            if isinstance(value, date):
                out[name] = value.isoformat()
            elif isinstance(value, Decimal):
                out[name] = float(value)
            elif name == "line_items":
                out[name] = [item.to_dict() for item in value]
            else:
                out[name] = value
        return out

    # --- convenience ----------------------------------------------------------------------------
    def present_fields(self) -> dict[str, Any]:
        """Field -> value for every non-empty field. Used for the mandatory-field rules."""
        out: dict[str, Any] = {}
        for name in FIELD_NAMES:
            value = getattr(self, name)
            if value is None:
                continue
            if isinstance(value, str) and not value.strip():
                continue
            if isinstance(value, list) and not value:
                continue
            out[name] = value
        return out

    def tax_split_required(self, small_amount_limit: float = 250.0) -> bool:
        """Small-amount invoices (UStDV s33) may state gross only."""
        if self.reverse_charge_note:
            return True
        if self.gross_amount is not None and abs(float(self.gross_amount)) <= small_amount_limit:
            return False
        return True

    def line_items_net(self) -> Decimal | None:
        totals = [item.line_net for item in self.line_items if item.line_net is not None]
        if not totals or len(totals) != len(self.line_items):
            return None
        return sum(totals, Decimal("0"))

    def days_until_due(self) -> int | None:
        if self.invoice_date and self.due_date:
            return (self.due_date - self.invoice_date).days
        return self.payment_terms_days


def _text(value: Any) -> str | None:
    """Coerce to a non-empty string, or None.

    The literal strings "null"/"none"/"n/a" are treated as absent. Models asked for JSON null
    sometimes emit the *word* null as a string; without this guard the field would be "present" with
    the value "null" and every presence check would pass while the data was garbage.
    """
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() in {"null", "none", "n/a", "na", "-"}:
        return None
    return text


def _int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(str(value).strip().rstrip("d").split()[0])
    except (ValueError, IndexError):
        return None


def _norm_iban(value: Any) -> str | None:
    text = _text(value)
    if text is None:
        return None
    return re.sub(r"\s+", "", text).upper()


@dataclass
class Fault:
    """One finding. ``evidence`` is what makes a report auditable and is shown to the user."""
    code: str
    severity: str
    message: str
    evidence: dict[str, Any] = field(default_factory=dict)
    detector: str = "rules"
    legal_reference: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class CheckResult:
    doc_id: str
    faults: list[Fault] = field(default_factory=list)
    extracted: bool = True
    llm_used: bool = False
    latency_ms: float = 0.0
    notes: list[str] = field(default_factory=list)

    @property
    def codes(self) -> list[str]:
        return sorted({fault.code for fault in self.faults})

    def to_dict(self) -> dict:
        return {
            "doc_id": self.doc_id,
            "codes": self.codes,
            "faults": [fault.to_dict() for fault in self.faults],
            "extracted": self.extracted,
            "llm_used": self.llm_used,
            "latency_ms": round(self.latency_ms, 2),
            "notes": self.notes,
        }


def unique_codes(codes: Iterable[str]) -> list[str]:
    return sorted(set(codes))
