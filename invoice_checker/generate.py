"""Testkorpus mit eingebauten, bekannten Fehlern.

Zwei Dinge machen das brauchbar: der Korpus ist aus einem Seed reproduzierbar,
und die Fehler werden programmatisch gesetzt und protokolliert -- die Auswertung
ist damit exakt und nicht Ansichtssache.

Pro Rechnung entstehen clean.txt (fehlerfrei), doc.txt (was der Prüfer sieht) und
optional doc.pdf / doc.png, damit auch der echte Einleseweg getestet wird.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from datetime import date, timedelta
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path

from . import checksums
from .config import THRESHOLDS, dump_json
from .models import Invoice, LineItem
from .pdfgen import render_scan_style, write_pdf

CENT = Decimal("0.01")

SERVICE_CATALOGUE = [
    ("Softwareentwicklung", "Software development", "Stunden", "hours", Decimal("95.00")),
    ("Projektmanagement", "Project management", "Stunden", "hours", Decimal("110.00")),
    ("Serverhosting", "Server hosting", "Monate", "months", Decimal("240.00")),
    ("Wartung und Support", "Maintenance and support", "Stunden", "hours", Decimal("78.50")),
    ("Lizenzgebuehr", "Licence fee", "Lizenzen", "licences", Decimal("450.00")),
    ("Schulung", "Training", "Tage", "days", Decimal("1200.00")),
    ("Beratung", "Consulting", "Stunden", "hours", Decimal("145.00")),
    ("Material und Verbrauchsmaterial", "Materials and consumables", "Stueck", "units", Decimal("32.40")),
    ("Reisekosten", "Travel expenses", "Positionen", "items", Decimal("187.30")),
    ("Datenmigration", "Data migration", "Pauschale", "flat rate", Decimal("2600.00")),
]

SUPPLIERS = [
    ("Nordwind IT-Systeme GmbH", "Hafenstrasse 14, 20359 Hamburg"),
    ("Bergmann Logistik AG", "Industrieweg 7, 70565 Stuttgart"),
    ("Kleinschmidt Buchhaltung e.K.", "Lindenallee 3, 04109 Leipzig"),
    ("Rheintal Edelstahl GmbH", "Werkstrasse 22, 47119 Duisburg"),
    ("Suedlicht Energie GmbH", "Sonnenweg 5, 80339 Muenchen"),
    ("Adler & Partner Rechtsanwaelte", "Kanzleistrasse 9, 60311 Frankfurt"),
    ("Vogtland Maschinenbau GmbH", "Gewerbering 18, 08527 Plauen"),
    ("Hanseatic Cloud Services AB", "Skeppsbron 12, 111 30 Stockholm"),
    ("Alpenland Verpackungen AG", "Talstrasse 44, 8001 Zuerich"),
    ("Meridian Consulting S.L.", "Calle Mayor 88, 28013 Madrid"),
]

OUR_COMPANY = "Kernwerk Digital GmbH"
OUR_ADDRESS = "Friedrichstrasse 100, 10117 Berlin"
OUR_VAT_ID = "DE811907980"
OUR_IBAN = "DE02120300000000202051"


@dataclass
class SupplierProfile:
    name: str
    address: str
    vat_id: str
    iban: str
    country: str
    contract_currency: str
    payment_terms_days: int
    approval_threshold: Decimal
    number_prefix: str

    def to_dict(self) -> dict:
        payload = dict(self.__dict__)
        payload["approval_threshold"] = float(self.approval_threshold)
        return payload


def build_master_data(seed: int = 20240517) -> dict:
    """Deterministic supplier master data. Persisted so rules and generator share one source."""
    rng = random.Random(seed)
    profiles: list[SupplierProfile] = []
    for index, (name, address) in enumerate(SUPPLIERS):
        country = "DE"
        if "Stockholm" in address:
            country = "SE"
        elif "Zuerich" in address:
            country = "CH"
        elif "Madrid" in address:
            country = "ES"
        if country == "DE":
            body = f"{rng.randint(10000000, 99999999)}"
            vat_id = checksums.make_vat_id_de(body)
        else:
            vat_id = f"{country}{rng.randint(100000000, 999999999)}"
        iban = checksums.make_iban_de(f"{rng.randint(10000000, 99999999)}", f"{rng.randint(0, 9999999999):010d}")
        profiles.append(
            SupplierProfile(
                name=name,
                address=address,
                vat_id=vat_id,
                iban=iban,
                country=country,
                contract_currency="EUR" if country != "SE" else "SEK",
                payment_terms_days=rng.choice([14, 30, 30, 45, 60]),
                approval_threshold=Decimal(rng.choice(["5000.00", "10000.00", "15000.00"])),
                number_prefix=f"{name.split()[0][:3].upper()}-{2024}",
            )
        )
    return {
        "our_company": {
            "name": OUR_COMPANY,
            "address": OUR_ADDRESS,
            "vat_id": OUR_VAT_ID,
            "iban": OUR_IBAN,
        },
        "currency": "EUR",
        "issuer_number_pattern": r"^[A-Z]{3}-\d{4}-\d{4}$",
        "suppliers": [profile.to_dict() for profile in profiles],
    }


def _q(value: Decimal) -> Decimal:
    return value.quantize(CENT, rounding=ROUND_HALF_UP)


@dataclass
class GeneratedDocument:
    doc_id: str
    clean: Invoice
    document: Invoice
    injected_codes: list[str]
    truth_codes: list[str]
    missing_fields: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    clean_text: str = ""
    doc_text: str = ""
    duplicate_of: str | None = None


class CorpusGenerator:
    def __init__(self, master: dict, seed: int = 20240517, start_date: date | None = None):
        self.master = master
        self.rng = random.Random(seed)
        self.profiles = [SupplierProfile(**{
            **profile, "approval_threshold": Decimal(str(profile["approval_threshold"]))
        }) for profile in master["suppliers"]]
        self.today = start_date or date(2024, 6, 30)

    # --- clean invoice --------------------------------------------------------------------------
    def make_clean_invoice(self, doc_id: str) -> tuple[Invoice, SupplierProfile]:
        rng = self.rng
        profile = rng.choice(self.profiles)
        sequence = rng.randint(1, 9999)
        invoice_number = f"{profile.number_prefix}-{sequence:04d}"

        invoice_date = self.today - timedelta(days=rng.randint(0, 300))
        service_date = invoice_date - timedelta(days=rng.randint(0, 25))
        terms = profile.payment_terms_days
        due_date = invoice_date + timedelta(days=terms)

        # Steuerfall: meist 19%, manchmal 7%, manchmal innergemeinschaftlich mit Reverse Charge.
        treatment = rng.choices(["standard", "reduced", "reverse"], weights=[70, 15, 15])[0]
        if treatment == "reverse" and profile.country == "DE":
            treatment = "standard"

        line_items: list[LineItem] = []
        net_total = Decimal("0")
        for _ in range(rng.randint(1, 5)):
            de, en, unit_de, unit_en, unit_price = rng.choice(SERVICE_CATALOGUE)
            quantity = Decimal(str(rng.choice([1, 1, 2, 3, 4, 5, 7.5, 10, 12, 16, 20, 40])))
            # The unit price is rounded FIRST and the line net is derived from that rounded price.
            # Deriving the line net from an unrounded price is the classic invoicing bug: the
            # printed price and the printed line total then disagree by a cent or two, so the
            # invoice is internally inconsistent even though its totals happen to be fine.
            # Real systems bill from the rounded unit price; so does this generator.
            price = _q(unit_price * Decimal(str(rng.uniform(0.85, 1.15))))
            line_net = _q(quantity * price)
            line_items.append(LineItem(
                description=de if treatment != "reverse" or profile.country in {"DE", "AT"} else f"{en} ({de})",
                quantity=quantity,
                unit_price=price,
                line_net=line_net,
            ))
            net_total += line_net
        net_total = _q(net_total)

        # Saubere Rechnungen bleiben unter der Genehmigungsgrenze, sonst meldet
        # AMOUNT_ABOVE_APPROVAL_THRESHOLD einen Beleg, der laut Ground Truth sauber ist.
        # Skaliert wird der Einzelpreis, damit Menge x Preis stimmig bleibt.
        ceiling = profile.approval_threshold / Decimal("1.19") * Decimal("0.8")
        if net_total > ceiling:
            scale = ceiling / net_total
            for item in line_items:
                if item.unit_price is not None and item.quantity is not None:
                    item.unit_price = _q(item.unit_price * scale)
                    item.line_net = _q(item.quantity * item.unit_price)
            net_total = _q(sum((item.line_net for item in line_items), Decimal("0")))

        if treatment == "standard":
            rate = Decimal("0.19")
            vat_amount = _q(net_total * rate)
            reverse_note = None
        elif treatment == "reduced":
            rate = Decimal("0.07")
            vat_amount = _q(net_total * rate)
            reverse_note = None
        else:
            rate = Decimal("0.00")
            vat_amount = Decimal("0.00")
            reverse_note = "Steuerschuldnerschaft des Leistungsempfaengers (Reverse Charge)"

        gross = _q(net_total + vat_amount)

        return Invoice(
            invoice_number=invoice_number,
            invoice_date=invoice_date,
            service_date=service_date,
            supplier_name=profile.name,
            supplier_address=profile.address,
            supplier_vat_id=profile.vat_id,
            customer_name=OUR_COMPANY,
            customer_address=OUR_ADDRESS,
            customer_vat_id=OUR_VAT_ID,
            net_amount=net_total,
            vat_rate=rate,
            vat_amount=vat_amount,
            gross_amount=gross,
            currency=profile.contract_currency,
            payment_terms_days=terms,
            due_date=due_date,
            iban=profile.iban,
            purchase_order=f"PO-{rng.randint(100000, 999999)}",
            cost_centre=rng.choice(["CC-100", "CC-200", "CC-300", "CC-400"]),
            reverse_charge_note=reverse_note,
            line_items=line_items,
        ), profile

    # --- fault injection ------------------------------------------------------------------------
    def inject(self, invoice: Invoice, profile: SupplierProfile, available: list[str],
               duplicate_source: str | None = None) -> tuple[Invoice, list[str]]:
        """Apply 1-2 randomly chosen faults. Returns the mutated copy and the fault codes."""
        rng = self.rng
        codes: list[str] = []
        count = rng.choices([1, 1, 1, 2], weights=[70, 10, 10, 10])[0]
        for _ in range(count):
            pool = [code for code in available if code not in codes]
            if not pool:
                break
            code = rng.choice(pool)
            if self._apply(code, invoice, profile, duplicate_source):
                codes.append(code)
        return invoice, codes

    def _apply(self, code: str, inv: Invoice, profile: SupplierProfile,
               duplicate_source: str | None) -> bool:
        rng = self.rng
        if code == "MISSING_INVOICE_NUMBER":
            inv.invoice_number = None
        elif code == "MISSING_INVOICE_DATE":
            inv.invoice_date = None
            inv.due_date = None
            inv.payment_terms_days = None
        elif code == "MISSING_SUPPLIER_NAME":
            inv.supplier_name = None
            inv.supplier_address = None
        elif code == "MISSING_SUPPLIER_VAT_ID":
            inv.supplier_vat_id = None
        elif code == "MISSING_CUSTOMER_NAME":
            inv.customer_name = None
        elif code == "MISSING_NET_AMOUNT":
            if not inv.tax_split_required():
                return False
            inv.net_amount = None
        elif code == "MISSING_VAT_RATE":
            if not inv.tax_split_required():
                return False
            inv.vat_rate = None
        elif code == "MISSING_VAT_AMOUNT":
            if not inv.tax_split_required() or (inv.vat_amount or Decimal(0)) == 0:
                return False
            inv.vat_amount = None
        elif code == "MISSING_GROSS_AMOUNT":
            inv.gross_amount = None
        elif code == "MISSING_PAYMENT_TERMS":
            inv.payment_terms_days = None
            inv.due_date = None
        elif code == "MISSING_SERVICE_DATE":
            inv.service_date = None
            inv.service_period_start = None
            inv.service_period_end = None
        elif code == "ARITHMETIC_GROSS_MISMATCH":
            if inv.gross_amount is None:
                return False
            inv.gross_amount = _q(inv.gross_amount + Decimal(str(rng.uniform(0.5, 40.0))))
        elif code == "ARITHMETIC_LINE_ITEMS_MISMATCH":
            if not inv.line_items or inv.net_amount is None:
                return False
            inv.net_amount = _q(inv.net_amount + Decimal(str(rng.uniform(5.0, 120.0))))
        elif code == "ARITHMETIC_VAT_AMOUNT_WRONG":
            if inv.vat_amount is None or (inv.vat_amount == 0):
                return False
            inv.vat_amount = _q(inv.vat_amount + Decimal(str(rng.uniform(1.0, 25.0))))
        elif code == "INVALID_VAT_ID_CHECKSUM":
            if not inv.supplier_vat_id or not inv.supplier_vat_id.startswith("DE"):
                return False
            digits = list(inv.supplier_vat_id[2:])
            digits[8] = str((int(digits[8]) + rng.randint(1, 9)) % 10)
            inv.supplier_vat_id = "DE" + "".join(digits)
        elif code == "INVALID_IBAN_CHECKSUM":
            if not inv.iban:
                return False
            digits = list(inv.iban)
            digits[2] = str((int(digits[2]) + rng.randint(1, 9)) % 10)
            inv.iban = "".join(digits)
        elif code == "INVALID_IBAN_FORMAT":
            if not inv.iban:
                return False
            inv.iban = inv.iban[:-1]
        elif code == "INVALID_INVOICE_NUMBER_FORMAT":
            if not inv.invoice_number:
                return False
            inv.invoice_number = inv.invoice_number.replace("-", "_")
        elif code == "DATE_IN_FUTURE":
            inv.invoice_date = self.today + timedelta(days=rng.randint(1, 30))
            inv.due_date = inv.invoice_date + timedelta(days=inv.payment_terms_days or 30)
        elif code == "DATE_DELIVERY_AFTER_INVOICE":
            if not (inv.invoice_date and inv.service_date):
                return False
            inv.service_date = inv.invoice_date + timedelta(days=rng.randint(1, 20))
        elif code == "DATE_DUE_BEFORE_INVOICE":
            if not (inv.invoice_date and inv.due_date):
                return False
            inv.due_date = inv.invoice_date - timedelta(days=rng.randint(1, 15))
        elif code == "DATE_DUE_TERM_UNUSUAL":
            if not inv.invoice_date:
                return False
            inv.payment_terms_days = 180
            inv.due_date = inv.invoice_date + timedelta(days=180)
        elif code == "REVERSE_CHARGE_NOTE_MISSING":
            if not inv.reverse_charge_note:
                return False
            inv.reverse_charge_note = None
        elif code == "VAT_RATE_UNEXPECTED":
            if inv.vat_rate is None or inv.vat_rate == 0:
                return False
            inv.vat_rate = Decimal("0.16")
            inv.vat_amount = _q((inv.net_amount or Decimal(0)) * Decimal("0.16"))
            if inv.gross_amount is not None and inv.net_amount is not None:
                inv.gross_amount = _q(inv.net_amount + inv.vat_amount)
        elif code == "SMALL_AMOUNT_FIELDS_INCONSISTENT":
            if inv.gross_amount is None or inv.gross_amount > Decimal("250"):
                return False
            inv.vat_rate = None
            inv.vat_amount = None
            inv.net_amount = None
        elif code == "UNKNOWN_SUPPLIER":
            inv.supplier_name = "Unbekannte Fremdfirma UG (haftungsbeschraenkt)"
            inv.supplier_address = "Nirgendwo 1, 00000 Musterstadt"
        elif code == "CURRENCY_MISMATCH":
            inv.currency = "USD" if profile.contract_currency == "EUR" else "EUR"
        elif code == "UNKNOWN_CURRENCY":
            inv.currency = "XYZ"
        elif code == "NEGATIVE_AMOUNT":
            if inv.gross_amount is None or inv.gross_amount <= 0:
                return False
            inv.gross_amount = -inv.gross_amount
        elif code == "AMOUNT_ABOVE_APPROVAL_THRESHOLD":
            # This fault changes the amount, which means an invoice generated as "clean" could
            # accidentally exceed its supplier's threshold and be flagged. The corpus generator
            # guarantees clean invoices stay under the threshold (see _clamp_to_threshold), leaving
            # this as the only way above it -- so the rule's precision stays honest.
            if inv.gross_amount is None:
                return False
            inv.gross_amount = _q(profile.approval_threshold + Decimal("500.00"))
            if inv.net_amount is not None and inv.vat_amount is not None:
                inv.net_amount = _q(inv.gross_amount / (Decimal(1) + (inv.vat_rate or Decimal(0))))
                inv.vat_amount = _q(inv.gross_amount - inv.net_amount)
        elif code == "AMOUNT_OUTLIER":
            if inv.gross_amount is None:
                return False
            inv.gross_amount = _q(inv.gross_amount * Decimal("9.5"))
            if inv.net_amount is not None:
                inv.net_amount = _q(inv.net_amount * Decimal("9.5"))
            if inv.vat_amount is not None:
                inv.vat_amount = _q(inv.vat_amount * Decimal("9.5"))
        elif code == "DUPLICATE_INVOICE":
            if not duplicate_source:
                return False
            inv.invoice_number = duplicate_source
        elif code == "IBAN_CHANGED_RECENTLY":
            other = checksums.make_iban_de(f"{rng.randint(10000000, 99999999)}", f"{rng.randint(0, 9999999999):010d}")
            if other == inv.iban:
                return False
            inv.iban = other
        else:
            return False
        return True


def maybe_price_line(inv: Invoice, index: int) -> str | None:
    """Deterministically give every Nth invoice a line-pricing defect.

    Rotating rather than randomising guarantees all three pricing faults appear in a corpus of a few
    hundred documents, so the per-fault metrics have a non-zero denominator instead of silently
    skipping the rule. Returns the injected code, or None when nothing was changed.
    """
    if not inv.line_items:
        return None
    pivot = index % 9
    if pivot not in (2, 5, 8):
        return None

    item = inv.line_items[index % len(inv.line_items)]
    if pivot == 2:
        # "10 h of labour, nothing charged." Line net zeroed; quantity left intact so the defect is
        # visible, and the invoice totals left alone so this does not cascade into the arithmetic
        # rules -- keeping the cause of each finding attributable to one rule.
        if item.quantity is None or item.quantity <= 0 or item.line_net is None:
            return None
        item.line_net = Decimal("0.00")
        return "LINE_AMOUNT_ZERO"
    if pivot == 5:
        if item.quantity is None or item.quantity <= 0 or item.unit_price is None:
            return None
        item.line_net = Decimal("0.00")
        item.unit_price = Decimal("0.00")
        return "LINE_UNIT_PRICE_ZERO"
    if item.unit_price is None or item.quantity is None or item.line_net is None:
        return None
    # Tippfehler um eine Stelle: der Satz ist ein Zehntel vom Üblichen.
    item.unit_price = _q(item.unit_price / Decimal(10))
    item.line_net = _q(item.quantity * item.unit_price)
    return "LINE_RATE_OUTLIER"


def rule_fault_codes(inv: Invoice, master: dict, thresholds=None,
                     rate_index: dict | None = None) -> set[str]:
    """Alle Fehler, die in diesem Dokumentzustand tatsächlich vorliegen.

    Das ist die Ground Truth für die Auswertung. Warum nicht einfach die injizierten
    Codes? Weil eine Änderung mehrere Regeln kippt: ein falscher Steuerbetrag
    verletzt "Netto x Satz = Steuer" und "Netto + Steuer = Brutto". Die zweite,
    korrekte Meldung als Fehlalarm zu zählen würde die Precision künstlich drücken.

    Folge für die Auswertung: bei perfekter Extraktion trifft die Engine diese Menge
    exakt, der Regel-Baseline-Recall ist also per Konstruktion 1,0. Interessant ist
    der Vergleich mit einer unsauberen Extraktion.
    """
    from .rules import RulesEngine

    thresholds = thresholds or THRESHOLDS
    engine = RulesEngine(master, thresholds)
    return {fault.code for fault in engine.check_invoice(inv, rate_index=rate_index)}


def build_corpus_rate_index(generator: "CorpusGenerator", count: int,
                            thresholds=None) -> dict:
    """Historical service rates, learned from a *separate* set of fault-free invoices.

    The rate index must not be learned from the documents being checked: if it were, a mispriced
    line would help define the very baseline it is compared against and the rate rule would be
    self-confirming. Deriving it from an independent clean sample is the honest construction and
    mirrors how a real system would learn rates from historical invoices.
    """
    from .rules import build_rate_index

    probe = CorpusGenerator(generator.master, seed=generator.rng.randrange(1, 10 ** 6))
    samples: list[Invoice] = []
    for index in range(max(count, 40)):
        clean, _ = probe.make_clean_invoice(f"rate-probe-{index:04d}")
        samples.append(clean)
    return build_rate_index(samples, thresholds)


def _legacy_missing_field_codes(inv: Invoice, thresholds) -> list[str]:
    """Kept for the self-test's targeted assertions; :func:`rule_fault_codes` supersedes it."""
    codes: list[str] = []
    split_required = inv.tax_split_required(thresholds.small_amount_limit)
    if not inv.invoice_number:
        codes.append("MISSING_INVOICE_NUMBER")
    if not inv.invoice_date:
        codes.append("MISSING_INVOICE_DATE")
    if not inv.supplier_name and not inv.supplier_address:
        codes.append("MISSING_SUPPLIER_NAME")
    if not inv.supplier_vat_id and not inv.supplier_tax_number:
        codes.append("MISSING_SUPPLIER_VAT_ID")
    if not inv.customer_name:
        codes.append("MISSING_CUSTOMER_NAME")
    if split_required:
        if inv.net_amount is None:
            codes.append("MISSING_NET_AMOUNT")
        if inv.vat_rate is None:
            codes.append("MISSING_VAT_RATE")
        if inv.vat_amount is None:
            codes.append("MISSING_VAT_AMOUNT")
    if inv.gross_amount is None:
        codes.append("MISSING_GROSS_AMOUNT")
    if inv.payment_terms_days is None and inv.due_date is None:
        codes.append("MISSING_PAYMENT_TERMS")
    if not inv.service_date and not inv.service_period_start and not inv.service_period_end:
        codes.append("MISSING_SERVICE_DATE")
    return codes


def render_document_text(inv: Invoice, profile: SupplierProfile | None, *, variant: int = 0) -> str:
    """Render an invoice as document text. ``variant`` selects between label/date conventions."""
    de = variant % 2 == 0
    date_fmt = "%d.%m.%Y" if variant % 3 != 2 else "%Y-%m-%d"

    def fmt_date(value):
        return value.strftime(date_fmt) if value else ""

    def money(value, currency=None):
        if value is None:
            return ""
        text = f"{value:,.2f}"
        return f"{text} {currency}" if currency else text

    lines: list[str] = []
    lines.append(profile.name if profile else (inv.supplier_name or ""))
    lines.append(profile.address if profile else (inv.supplier_address or ""))
    lines.append("")
    if de:
        lines.append(f"Rechnung Nr. {inv.invoice_number or ''}")
        lines.append(f"Rechnungsdatum: {fmt_date(inv.invoice_date)}")
        if inv.service_date:
            lines.append(f"Leistungsdatum: {fmt_date(inv.service_date)}")
        if inv.service_period_start:
            lines.append(
                f"Leistungszeitraum: {fmt_date(inv.service_period_start)} - {fmt_date(inv.service_period_end)}"
            )
        lines.append(f"Umsatzsteuer-Identifikationsnummer: {inv.supplier_vat_id or ''}")
        lines.append("")
        lines.append(f"Rechnungsempfaenger: {inv.customer_name or ''}")
        lines.append(f"Anschrift: {inv.customer_address or ''}")
        if inv.customer_vat_id:
            lines.append(f"USt-IdNr. Empfaenger: {inv.customer_vat_id}")
    else:
        lines.append(f"Invoice No. {inv.invoice_number or ''}")
        lines.append(f"Invoice date: {fmt_date(inv.invoice_date)}")
        if inv.service_date:
            lines.append(f"Date of supply: {fmt_date(inv.service_date)}")
        lines.append(f"VAT identification number: {inv.supplier_vat_id or ''}")
        lines.append("")
        lines.append(f"Bill to: {inv.customer_name or ''}")
        lines.append(f"Address: {inv.customer_address or ''}")

    lines.append("")
    lines.append("Pos  Beschreibung                          Menge      Einzelpreis        Betrag")
    lines.append("-" * 88)
    for index, item in enumerate(inv.line_items, start=1):
        description = (item.description or "")[:34]
        quantity = "" if item.quantity is None else f"{item.quantity:g}"
        lines.append(
            f"{index:>3}  {description:<34}{quantity:>8}  "
            f"{money(item.unit_price):>14}  {money(item.line_net):>14}"
        )
    lines.append("-" * 88)
    lines.append("")
    lines.append(f"Gesamtbetrag netto / Net total:            {money(inv.net_amount)}")
    if inv.vat_rate is not None:
        rate_text = f"{(inv.vat_rate * 100).normalize():f}".rstrip("0").rstrip(".")
        lines.append(f"Umsatzsteuer {rate_text}% / VAT:              {money(inv.vat_amount)}")
    else:
        lines.append(f"Umsatzsteuer / VAT:                        {money(inv.vat_amount)}")
    lines.append(f"Rechnungsbetrag brutto / Gross total:      {money(inv.gross_amount)} {inv.currency or ''}".rstrip())
    lines.append("")
    if inv.reverse_charge_note:
        lines.append(inv.reverse_charge_note)
    if inv.payment_terms_days is not None:
        lines.append(
            f"Zahlungsbedingungen: zahlbar innerhalb von {inv.payment_terms_days} Tagen "
            f"bis zum {fmt_date(inv.due_date)}"
        )
    else:
        lines.append(f"Zahlungsbedingungen: faellig am {fmt_date(inv.due_date)}")
    lines.append(f"Bankverbindung / IBAN: {inv.iban or ''}")
    if inv.purchase_order:
        lines.append(f"Bestellnummer / PO: {inv.purchase_order}")
    if inv.cost_centre:
        lines.append(f"Kostenstelle: {inv.cost_centre}")
    lines.append("")
    lines.append("Vielen Dank fuer Ihren Auftrag.")
    return "\n".join(lines)


# Verteilung der eingebauten Fehler.
FAULT_MIX = [
    "MISSING_INVOICE_NUMBER", "MISSING_INVOICE_DATE", "MISSING_SUPPLIER_NAME",
    "MISSING_SUPPLIER_VAT_ID", "MISSING_CUSTOMER_NAME", "MISSING_NET_AMOUNT",
    "MISSING_VAT_RATE", "MISSING_VAT_AMOUNT", "MISSING_GROSS_AMOUNT",
    "MISSING_PAYMENT_TERMS", "MISSING_SERVICE_DATE",
    "ARITHMETIC_GROSS_MISMATCH", "ARITHMETIC_LINE_ITEMS_MISMATCH", "ARITHMETIC_VAT_AMOUNT_WRONG",
    "INVALID_VAT_ID_CHECKSUM", "INVALID_IBAN_CHECKSUM", "INVALID_IBAN_FORMAT",
    "INVALID_INVOICE_NUMBER_FORMAT",
    "DATE_IN_FUTURE", "DATE_DELIVERY_AFTER_INVOICE", "DATE_DUE_BEFORE_INVOICE",
    "DATE_DUE_TERM_UNUSUAL",
    "REVERSE_CHARGE_NOTE_MISSING", "VAT_RATE_UNEXPECTED", "SMALL_AMOUNT_FIELDS_INCONSISTENT",
    "UNKNOWN_SUPPLIER", "CURRENCY_MISMATCH", "UNKNOWN_CURRENCY", "NEGATIVE_AMOUNT",
    "AMOUNT_ABOVE_APPROVAL_THRESHOLD", "AMOUNT_OUTLIER", "IBAN_CHANGED_RECENTLY",
]


def generate_corpus(count: int, *, seed: int = 20240517, fault_rate: float = 0.6,
                    include_duplicates: bool = True, master: dict | None = None,
                    thresholds=None) -> list[GeneratedDocument]:
    """Build the corpus in memory. Deterministic for a given seed, count and fault rate."""
    from .config import THRESHOLDS

    thresholds = thresholds or THRESHOLDS
    master = master or build_master_data(seed)
    generator = CorpusGenerator(master, seed=seed)
    rng = generator.rng

    # Aus einer unabhängigen sauberen Stichprobe, nicht aus den Prüflingen selbst.
    rate_index = build_corpus_rate_index(generator, count, thresholds)

    documents: list[GeneratedDocument] = []
    first_by_amount: dict[tuple[str, str], str] = {}

    for index in range(count):
        doc_id = f"inv-{index + 1:04d}"
        clean, profile = generator.make_clean_invoice(doc_id)
        clean_text = render_document_text(clean, profile, variant=index)

        document = Invoice.from_dict(clean.to_dict())
        injected: list[str] = []
        duplicate_of = None

        if rng.random() < fault_rate:
            document, injected = generator.inject(document, profile, list(FAULT_MIX))
        else:
            # Auch ein sonst fehlerfreier Beleg kann eine falsch bepreiste Position haben.
            # Rotierend statt zufällig, damit alle drei Preisregeln sicher vorkommen.
            pricing_fault = maybe_price_line(document, index)
            if pricing_fault:
                injected = [pricing_fault]

        # Duplikat aus der sauberen Vorlage, damit die Rechnung in sich stimmig bleibt und
        # nur die Nummer das Problem ist. Gleicher Lieferant + Betrag + Datum ist genau der
        # Fingerabdruck, den der Detektor verwendet.
        if include_duplicates and clean.invoice_number and clean.gross_amount is not None:
            key = (profile.name, str(clean.gross_amount), clean.invoice_date.isoformat())
            if key in first_by_amount and rng.random() < 0.6:
                clone = Invoice.from_dict(clean.to_dict())
                clone.invoice_number = first_by_amount[key]
                document = clone
                injected = ["DUPLICATE_INVOICE"]
                duplicate_of = first_by_amount[key]
            else:
                first_by_amount.setdefault(key, clean.invoice_number)

        # Ground Truth aus dem fertigen Dokument, nicht aus der Inject-Liste: eine
        # Änderung kann mehrere Regeln gleichzeitig verletzen.
        truth = rule_fault_codes(document, master, thresholds, rate_index=rate_index)
        if duplicate_of:
            truth.add("DUPLICATE_INVOICE")

        documents.append(GeneratedDocument(
            doc_id=doc_id,
            clean=clean,
            document=document,
            injected_codes=sorted(set(injected)),
            truth_codes=sorted(truth),
            clean_text=clean_text,
            doc_text=render_document_text(document, profile, variant=index),
            duplicate_of=duplicate_of,
        ))

    return documents


def write_corpus(documents: list[GeneratedDocument], directory: Path, *, render_pdf: bool = True,
                 scan_ratio: float = 0.0) -> dict:
    """Write text/PDF/scan artefacts plus per-invoice ground truth. Returns a manifest."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    manifest = {"count": len(documents), "faulty": 0, "scans": 0, "pdfs": 0, "documents": []}

    for index, generated in enumerate(documents):
        base = directory / generated.doc_id
        (base.with_suffix(".clean.txt")).write_text(generated.clean_text, encoding="utf-8")
        (base.with_suffix(".doc.txt")).write_text(generated.doc_text, encoding="utf-8")

        artefacts = {"pdf": None, "scan": None}
        if render_pdf:
            pdf_path = base.with_suffix(".pdf")
            write_pdf(pdf_path, generated.doc_text.splitlines())
            artefacts["pdf"] = pdf_path.name
            manifest["pdfs"] += 1
        if scan_ratio > 0 and (index % max(1, int(round(1 / scan_ratio))) == 0):
            scan_path = base.with_suffix(".png")
            if render_scan_style(generated.doc_text.splitlines(), scan_path, seed=index):
                artefacts["scan"] = scan_path.name
                manifest["scans"] += 1

        truth = {
            "doc_id": generated.doc_id,
            "truth_codes": generated.truth_codes,
            "injected_codes": generated.injected_codes,
            "duplicate_of": generated.duplicate_of,
            "artefacts": artefacts,
            "clean_invoice": generated.clean.to_dict(),
            "document_invoice": generated.document.to_dict(),
        }
        dump_json(base.with_suffix(".truth.json"), truth)

        manifest["documents"].append({
            "doc_id": generated.doc_id,
            "truth_codes": generated.truth_codes,
            "artefacts": artefacts,
        })
        if generated.truth_codes:
            manifest["faulty"] += 1

    dump_json(directory / "manifest.json", manifest)
    return manifest
