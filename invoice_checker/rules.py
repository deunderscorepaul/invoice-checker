"""Die Regel-Engine -- hier werden die meisten Fehler gefunden.

Jede Regel ist eine kleine Funktion über Invoice + Stammdaten, keine ruft ein
Modell. Das LLM befüllt nur die Felder; verbockt es einen Wert, schlagen die
Regeln zu statt der Meinung des Modells zu glauben.

Beträge laufen über Decimal mit zwei Nachkommastellen. Mit Floats gäbe es
Fehlalarme wie 1190.00 vs. 1189.9999999999998.
"""
from __future__ import annotations

from datetime import date
from decimal import Decimal, ROUND_HALF_UP
from typing import Iterable

from . import checksums
from .config import THRESHOLDS, Thresholds
from .models import Fault, Invoice
from .taxonomy import FAULT_BY_CODE, Severity

CENT = Decimal("0.01")
EU_COUNTRIES = {
    "AT", "BE", "BG", "HR", "CY", "CZ", "DK", "EE", "FI", "FR", "GR", "HU", "IE", "IT",
    "LV", "LT", "LU", "MT", "NL", "PL", "PT", "RO", "SK", "SI", "ES", "SE",
}


def q(value: Decimal) -> Decimal:
    return value.quantize(CENT, rounding=ROUND_HALF_UP)


class RulesEngine:
    def __init__(self, master: dict, thresholds: Thresholds | None = None,
                 today: date | None = None):
        self.master = master
        self.thresholds = thresholds or THRESHOLDS
        self.today = today or date(2024, 6, 30)
        self.suppliers = {entry["name"]: entry for entry in master.get("suppliers", [])}
        self.our_company = master.get("our_company", {})
        self.number_pattern = master.get("issuer_number_pattern")

    # --- helper ---------------------------------------------------------------------------------
    def fault(self, code: str, message: str, **evidence) -> Fault:
        spec = FAULT_BY_CODE.get(code)
        return Fault(
            code=code,
            severity=(spec.severity.value if spec else Severity.INFO.value),
            message=message,
            evidence=evidence,
            detector=(spec.detector.value if spec else "rules"),
            legal_reference=(spec.legal_reference if spec else ""),
        )

    def supplier_profile(self, invoice: Invoice) -> dict | None:
        if not invoice.supplier_name:
            return None
        if invoice.supplier_name in self.suppliers:
            return self.suppliers[invoice.supplier_name]
        # Tolerate OCR spacing/casing noise when matching master data.
        needle = _normalise(invoice.supplier_name)
        for name, profile in self.suppliers.items():
            if _normalise(name) == needle:
                return profile
        return None

    # --- individual rule groups -----------------------------------------------------------------
    def check_required_fields(self, inv: Invoice) -> list[Fault]:
        faults: list[Fault] = []
        split_required = inv.tax_split_required(self.thresholds.small_amount_limit)

        if not inv.invoice_number:
            faults.append(self.fault("MISSING_INVOICE_NUMBER", "No invoice number found"))
        if not inv.invoice_date:
            faults.append(self.fault("MISSING_INVOICE_DATE", "No invoice date found"))
        if not inv.supplier_name and not inv.supplier_address:
            faults.append(self.fault("MISSING_SUPPLIER_NAME", "Issuer name and address are absent"))
        if not inv.supplier_vat_id and not inv.supplier_tax_number:
            faults.append(self.fault("MISSING_SUPPLIER_VAT_ID",
                                     "No VAT identification number or tax number for the issuer"))
        if not inv.customer_name:
            faults.append(self.fault("MISSING_CUSTOMER_NAME", "Recipient name is absent"))
        if inv.gross_amount is None:
            faults.append(self.fault("MISSING_GROSS_AMOUNT", "Gross total is absent"))

        if split_required:
            if inv.net_amount is None:
                faults.append(self.fault("MISSING_NET_AMOUNT",
                                         "Net amount absent but a tax split is required"))
            if inv.vat_rate is None:
                faults.append(self.fault("MISSING_VAT_RATE",
                                         "Tax rate absent but a tax split is required"))
            if inv.vat_amount is None:
                faults.append(self.fault("MISSING_VAT_AMOUNT",
                                         "Tax amount absent but a tax split is required"))
        else:
            # Small-amount relief applies; if some split fields are present the document is
            # internally contradictory rather than merely simplified.
            present = [name for name, value in (
                ("net_amount", inv.net_amount), ("vat_rate", inv.vat_rate),
                ("vat_amount", inv.vat_amount)) if value is not None]
            if present and len(present) < 3:
                faults.append(self.fault(
                    "SMALL_AMOUNT_FIELDS_INCONSISTENT",
                    "Small-amount invoice states some but not all of the tax split fields",
                    gross=str(inv.gross_amount), present_fields=present,
                    limit=str(self.thresholds.small_amount_limit),
                ))

        if inv.payment_terms_days is None and inv.due_date is None:
            faults.append(self.fault("MISSING_PAYMENT_TERMS", "No due date or payment term stated"))
        if not inv.service_date and not inv.service_period_start and not inv.service_period_end:
            faults.append(self.fault("MISSING_SERVICE_DATE",
                                     "Neither service date nor service period stated"))
        return faults

    def check_arithmetic(self, inv: Invoice) -> list[Fault]:
        faults: list[Fault] = []
        tolerance = Decimal(str(self.thresholds.amount_tolerance))

        if inv.net_amount is not None and inv.vat_amount is not None and inv.gross_amount is not None:
            expected = q(inv.net_amount + inv.vat_amount)
            if abs(expected - q(inv.gross_amount)) > tolerance:
                faults.append(self.fault(
                    "ARITHMETIC_GROSS_MISMATCH",
                    f"Net {inv.net_amount} + VAT {inv.vat_amount} = {expected}, "
                    f"but the invoice states gross {inv.gross_amount}",
                    net=str(inv.net_amount), vat=str(inv.vat_amount),
                    expected_gross=str(expected), stated_gross=str(inv.gross_amount),
                    difference=str(q(inv.gross_amount - expected)),
                    # Carried so a localised report can format the amounts with their currency.
                    currency=inv.currency,
                ))

        line_total = inv.line_items_net()
        if line_total is not None and inv.net_amount is not None:
            if abs(q(line_total) - q(inv.net_amount)) > tolerance:
                faults.append(self.fault(
                    "ARITHMETIC_LINE_ITEMS_MISMATCH",
                    f"Line items sum to {q(line_total)} but the stated net is {inv.net_amount}",
                    line_items_total=str(q(line_total)), stated_net=str(inv.net_amount),
                ))

        if inv.net_amount is not None and inv.vat_rate is not None and inv.vat_amount is not None:
            expected_vat = q(inv.net_amount * inv.vat_rate)
            if abs(expected_vat - q(inv.vat_amount)) > tolerance:
                faults.append(self.fault(
                    "ARITHMETIC_VAT_AMOUNT_WRONG",
                    f"{inv.net_amount} at {_pct(inv.vat_rate)} implies VAT {expected_vat}, "
                    f"but the invoice states {inv.vat_amount}",
                    net=str(inv.net_amount), rate=str(inv.vat_rate),
                    expected_vat=str(expected_vat), stated_vat=str(inv.vat_amount),
                ))
        return faults

    def check_line_item_pricing(self, inv: Invoice, *,
                                rate_index: dict | None = None) -> list[Fault]:
        """Leistung erbracht, aber nicht berechnet.

        Kernfall: "10 Stunden" mit Einzelpreis oder Betrag 0. Die positive Menge ist
        das Signal -- es wurde etwas geleistet, 0 EUR ist also ein Tippfehler und
        keine Gefälligkeit.

        Bewusst zurückhaltend: Betrag 0 wird nur gemeldet, wenn der Betrag wirklich
        auf 0 rundet, und ein nur niedriger Satz bleibt ein Hinweis. Zu viele
        Fehlalarme auf echte Gratispositionen machen das Tool unbrauchbar.
        """
        faults: list[Fault] = []
        zero_net = Decimal(str(self.thresholds.line_net_zero_threshold))
        tolerance = Decimal(str(self.thresholds.amount_tolerance))

        for index, item in enumerate(inv.line_items, start=1):
            label = item.description or f"line {index}"
            quantity = item.quantity
            positive_quantity = quantity is not None and quantity > 0

            # 1. Menge da, kein Geld. Der eigentliche "10 h x 0 EUR"-Fall.
            if positive_quantity and item.line_net is not None and abs(item.line_net) <= zero_net:
                faults.append(self.fault(
                    "LINE_AMOUNT_ZERO",
                    f"Line {index} ({label}) bills {quantity} units but a net amount of "
                    f"{item.line_net} is charged",
                    line=index, description=item.description, quantity=str(quantity),
                    unit_price=_opt(item.unit_price), line_net=str(item.line_net),
                ))

            # 2. Menge da, Einzelpreis 0. Auch wenn der Betrag fehlt, denn der Preis ist
            #    das Feld, das jemand korrigieren würde.
            if positive_quantity and item.unit_price is not None and abs(item.unit_price) <= zero_net:
                faults.append(self.fault(
                    "LINE_UNIT_PRICE_ZERO",
                    f"Line {index} ({label}) bills {quantity} units at a unit price of "
                    f"{item.unit_price}",
                    line=index, description=item.description, quantity=str(quantity),
                    unit_price=str(item.unit_price),
                ))

            # 3. Menge x Einzelpreis muss der Positionsbetrag sein. Spielraum ist nur die
            #    halbe Cent Rundung, die der Generator selbst macht.
            if quantity is not None and item.unit_price is not None and item.line_net is not None:
                expected = q(quantity * item.unit_price)
                if abs(expected - q(item.line_net)) > tolerance:
                    faults.append(self.fault(
                        "LINE_QUANTITY_PRICE_INCONSISTENT",
                        f"Line {index} ({label}): {quantity} x {item.unit_price} = {expected}, "
                        f"but the line states {item.line_net}",
                        line=index, description=item.description, quantity=str(quantity),
                        unit_price=str(item.unit_price), expected_net=str(expected),
                        stated_net=str(item.line_net),
                    ))

            # 4. Satz deutlich unter dem üblichen Satz für diese Leistung. Nur bei Preis > 0,
            #    sonst hätte Regel 2 schon gemeldet.
            if rate_index and item.unit_price is not None and item.unit_price > 0:
                key = normalise_service(item.description)
                entry = rate_index.get(key)
                if entry and entry["count"] >= self.thresholds.line_rate_min_samples:
                    median = entry["median"]
                    if median > 0:
                        ratio = float(item.unit_price) / median
                        if ratio < self.thresholds.line_rate_ratio:
                            faults.append(self.fault(
                                "LINE_RATE_OUTLIER",
                                f"Line {index} ({label}) is priced at {item.unit_price} per unit, "
                                f"{ratio:.0%} of the historical median {median:.2f} for this service",
                                line=index, description=item.description,
                                unit_price=str(item.unit_price), historical_median=round(median, 2),
                                ratio=round(ratio, 3),
                            ))
        return faults

    def check_identifiers(self, inv: Invoice) -> list[Fault]:
        faults: list[Fault] = []
        vat_id = (inv.supplier_vat_id or "").strip().upper()

        if vat_id:
            if vat_id.startswith("DE"):
                if not checksums.vat_id_checksum_de(vat_id):
                    faults.append(self.fault(
                        "INVALID_VAT_ID_CHECKSUM",
                        f"German VAT ID {vat_id} fails its check digit",
                        vat_id=vat_id, country="DE",
                    ))
            # Scope note: only German VAT IDs are validated numerically here. The equivalent
            # ISO 7064 variants for SE/CH/ES/FR differ per member state, and guessing at them
            # produces false positives on entirely valid foreign IDs -- which the clean-corpus
            # control in the self-test caught. A production system would query VIES instead; that
            # is deliberately out of scope for a local-only checker.

        if inv.iban:
            if not checksums.iban_is_valid(inv.iban):
                if checksums.iban_country_length_ok(inv.iban):
                    faults.append(self.fault(
                        "INVALID_IBAN_CHECKSUM",
                        f"IBAN {inv.iban} fails the MOD-97 check",
                        iban=inv.iban, mod97=checksums.iban_mod97(inv.iban),
                    ))
                else:
                    faults.append(self.fault(
                        "INVALID_IBAN_FORMAT",
                        f"IBAN {inv.iban} has an impossible length for its country code",
                        iban=inv.iban, length=len(inv.iban),
                        expected=checksums.IBAN_LENGTHS.get(inv.iban[:2]),
                    ))

        if inv.invoice_number and self.number_pattern:
            import re

            if not re.match(self.number_pattern, inv.invoice_number):
                faults.append(self.fault(
                    "INVALID_INVOICE_NUMBER_FORMAT",
                    f"Invoice number '{inv.invoice_number}' does not match the issuing scheme",
                    pattern=self.number_pattern, value=inv.invoice_number,
                ))
        return faults

    def check_dates(self, inv: Invoice) -> list[Fault]:
        faults: list[Fault] = []
        if inv.invoice_date and inv.invoice_date > self.today:
            faults.append(self.fault(
                "DATE_IN_FUTURE",
                f"Invoice date {inv.invoice_date} is after the processing date {self.today}",
                invoice_date=inv.invoice_date.isoformat(), reference_date=self.today.isoformat(),
            ))
        if inv.invoice_date and inv.service_date and inv.service_date > inv.invoice_date:
            faults.append(self.fault(
                "DATE_DELIVERY_AFTER_INVOICE",
                f"Service date {inv.service_date} is later than invoice date {inv.invoice_date}",
                service_date=inv.service_date.isoformat(),
                invoice_date=inv.invoice_date.isoformat(),
            ))
        if inv.invoice_date and inv.due_date and inv.due_date < inv.invoice_date:
            faults.append(self.fault(
                "DATE_DUE_BEFORE_INVOICE",
                f"Due date {inv.due_date} precedes invoice date {inv.invoice_date}",
                due_date=inv.due_date.isoformat(), invoice_date=inv.invoice_date.isoformat(),
            ))
        term = inv.days_until_due()
        if term is not None and not (-15 <= term <= self.thresholds.max_payment_term_days):
            faults.append(self.fault(
                "DATE_DUE_TERM_UNUSUAL",
                f"Payment term of {term} days is outside the plausible range",
                term_days=term, max_days=self.thresholds.max_payment_term_days,
            ))
        return faults

    def check_tax_treatment(self, inv: Invoice) -> list[Fault]:
        faults: list[Fault] = []
        rate = float(inv.vat_rate) if inv.vat_rate is not None else None
        if not checksums.vat_rate_is_permitted(rate):
            faults.append(self.fault(
                "VAT_RATE_UNEXPECTED",
                f"Tax rate {_pct(inv.vat_rate)} is not a permitted rate",
                rate=str(inv.vat_rate), permitted=sorted(checksums.PERMITTED_VAT_RATES),
            ))

        # Reverse Charge: steuerfreie innergemeinschaftliche Lieferung braucht den Hinweis
        # nach § 14a Abs. 5 UStG.
        if (inv.vat_rate is not None and float(inv.vat_rate) == 0.0
                and not inv.reverse_charge_note):
            customer_country = (inv.customer_vat_id or "")[:2].upper()
            if customer_country in EU_COUNTRIES:
                faults.append(self.fault(
                    "REVERSE_CHARGE_NOTE_MISSING",
                    "Zero-rated intra-EU supply without the required reverse-charge note",
                    vat_rate=str(inv.vat_rate), customer_country=customer_country,
                ))
        return faults

    def check_commercial(self, inv: Invoice, *, stats: dict | None = None) -> list[Fault]:
        faults: list[Fault] = []
        profile = self.supplier_profile(inv)

        if inv.supplier_name and profile is None:
            faults.append(self.fault(
                "UNKNOWN_SUPPLIER",
                f"Supplier '{inv.supplier_name}' is not in master data",
                supplier=inv.supplier_name,
            ))

        if inv.currency:
            if not checksums.currency_is_iso(inv.currency):
                faults.append(self.fault(
                    "UNKNOWN_CURRENCY", f"'{inv.currency}' is not a recognised ISO 4217 code",
                    currency=inv.currency,
                ))
            elif profile and profile.get("contract_currency") and \
                    inv.currency != profile["contract_currency"]:
                faults.append(self.fault(
                    "CURRENCY_MISMATCH",
                    f"Invoice is in {inv.currency} but the contract currency is "
                    f"{profile['contract_currency']}",
                    invoice_currency=inv.currency,
                    contract_currency=profile["contract_currency"],
                ))

        if inv.gross_amount is not None and inv.gross_amount < 0:
            faults.append(self.fault(
                "NEGATIVE_AMOUNT", f"Gross amount {inv.gross_amount} is negative",
                gross=str(inv.gross_amount),
            ))

        if profile and inv.gross_amount is not None:
            threshold = Decimal(str(profile["approval_threshold"]))
            if inv.gross_amount > threshold:
                faults.append(self.fault(
                    "AMOUNT_ABOVE_APPROVAL_THRESHOLD",
                    f"Gross {inv.gross_amount} exceeds the approval threshold {threshold} "
                    f"for this supplier",
                    gross=str(inv.gross_amount), threshold=str(threshold),
                    cost_centre=inv.cost_centre,
                ))
            if inv.iban and profile.get("iban") and inv.iban != profile["iban"]:
                faults.append(self.fault(
                    "IBAN_CHANGED_RECENTLY",
                    "Bank details differ from the IBAN held in master data",
                    invoice_iban=inv.iban, master_iban=profile["iban"],
                ))
        # Der IBAN-Vergleich steht bewusst innerhalb von `if profile`: bei unbekanntem
        # Lieferanten gibt es keine Stammdaten-IBAN, ein Vergleich würde also jeden
        # solchen Beleg doppelt melden.

        outlier, evidence = outlier_reason(inv, stats, self.thresholds) if stats else (False, {})
        if outlier:
            faults.append(self.fault(
                "AMOUNT_OUTLIER",
                evidence.pop("message"),
                **evidence,
            ))
        return faults

    # --- corpus-level ---------------------------------------------------------------------------
    def supplier_amount_stats(self, invoices: Iterable[Invoice]) -> dict[str, dict]:
        """supplier -> {mean, stdev, count} of gross amounts, for outlier detection.

        Only positive amounts are counted: a negative gross is already reported as NEGATIVE_AMOUNT
        and would otherwise drag the mean around and mask genuine outliers.
        """
        buckets: dict[str, list[float]] = {}
        for inv in invoices:
            if inv.supplier_name and inv.gross_amount is not None and inv.gross_amount > 0:
                buckets.setdefault(inv.supplier_name, []).append(float(inv.gross_amount))
        stats: dict[str, dict] = {}
        for name, values in buckets.items():
            if len(values) < 2:
                continue
            mean = sum(values) / len(values)
            variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
            stats[name] = {"mean": mean, "stdev": variance ** 0.5, "count": len(values)}
        return stats

    def check_invoice(self, inv: Invoice, *, stats: dict | None = None,
                      rate_index: dict | None = None, duplicate: bool = False) -> list[Fault]:
        faults: list[Fault] = []
        faults.extend(self.check_required_fields(inv))
        faults.extend(self.check_arithmetic(inv))
        faults.extend(self.check_line_item_pricing(inv, rate_index=rate_index))
        faults.extend(self.check_identifiers(inv))
        faults.extend(self.check_dates(inv))
        faults.extend(self.check_tax_treatment(inv))
        faults.extend(self.check_commercial(inv, stats=stats))

        if duplicate:
            faults.append(self.fault(
                "DUPLICATE_INVOICE",
                f"Invoice number {inv.invoice_number} appears on more than one document",
                invoice_number=inv.invoice_number,
            ))
        return faults


def _normalise(text: str) -> str:
    return " ".join(text.lower().replace("&", "und").split())


def normalise_service(description: str | None) -> str:
    """Key for the historical rate index. Aggressive on purpose so layout noise folds together."""
    if not description:
        return ""
    cleaned = _normalise(description)
    for filler in ("(de)", "(en)", "stunden", "hours", "pauschal", "flat rate"):
        cleaned = cleaned.replace(filler, " ")
    return " ".join(cleaned.split())


def _opt(value: Decimal | None) -> str | None:
    return None if value is None else str(value)


def build_rate_index(invoices: Iterable[Invoice], thresholds: Thresholds | None = None) -> dict:
    """service -> {median, count} of observed unit prices.

    This is the "what does this service normally cost" table. Median rather than mean because a
    handful of mispriced lines would drag a mean downwards and hide exactly the faults we are
    looking for -- the estimator has to be robust to the outliers being detected.
    """
    buckets: dict[str, list[float]] = {}
    for inv in invoices:
        for item in inv.line_items:
            if item.unit_price is None or item.unit_price <= 0:
                continue
            key = normalise_service(item.description)
            if key:
                buckets.setdefault(key, []).append(float(item.unit_price))

    index: dict[str, dict] = {}
    for key, values in buckets.items():
        values.sort()
        middle = len(values) // 2
        median = (values[middle] if len(values) % 2
                  else (values[middle - 1] + values[middle]) / 2)
        index[key] = {"median": median, "count": len(values)}
    return index


def _pct(rate: Decimal | None) -> str:
    if rate is None:
        return "n/a"
    value = (rate * 100).normalize()
    return f"{value:f}%"


def outlier_reason(inv: Invoice, stats: dict, thresholds: Thresholds) -> tuple[bool, dict]:
    """Ist der Bruttobetrag für diesen Lieferanten ungewöhnlich?

    Zwei Tests, weil einer bei kleiner Datenmenge nicht reicht:
    z-Score ab genug Rechnungen, sonst ein Verhältnis zum Mittelwert -- bei wenigen
    Belegen wird die Standardabweichung vom Ausreißer selbst aufgebläht.
    """
    if not stats or not inv.supplier_name or inv.gross_amount is None:
        return False, {}
    entry = stats.get(inv.supplier_name)
    if not entry or entry.get("count", 0) < thresholds.outlier_min_samples:
        return False, {}

    amount = float(inv.gross_amount)
    mean = entry["mean"]
    stdev = entry["stdev"]
    if mean <= 0:
        return False, {}

    zscore = (amount - mean) / stdev if stdev > 0 else 0.0
    ratio = amount / mean
    if zscore >= thresholds.outlier_zscore:
        return True, {
            "message": f"Gross {amount:.2f} is {zscore:.1f} standard deviations above the "
                       f"{inv.supplier_name} mean of {mean:.2f}",
            "zscore": round(zscore, 2), "ratio": round(ratio, 2), "mean": round(mean, 2),
        }
    if ratio >= thresholds.outlier_ratio:
        return True, {
            "message": f"Gross {amount:.2f} is {ratio:.1f}x the {inv.supplier_name} mean of "
                       f"{mean:.2f}",
            "zscore": round(zscore, 2), "ratio": round(ratio, 2), "mean": round(mean, 2),
        }
    return False, {}
