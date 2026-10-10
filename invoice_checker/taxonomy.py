"""Fault taxonomy shared by the generator, the rule engine and the evaluator.

Every fault the checker can report has exactly one code here. The generator injects only codes
marked ``injectable`` and the evaluator scores only those, which keeps the ground truth and the
prediction space in lockstep -- a common source of bogus metrics in student projects.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class Severity(str, Enum):
    CRITICAL = "critical"   # invoice is legally defective / cannot be booked
    MAJOR = "major"         # must be corrected before sending
    MINOR = "minor"         # formal shortcoming, advisory
    INFO = "info"           # suspicious, needs a human look


class Detector(str, Enum):
    RULES = "rules"         # deterministic, no model involved
    LLM = "llm"             # requires language understanding
    HYBRID = "hybrid"       # rules produce candidates, model adjudicates


@dataclass(frozen=True)
class FaultSpec:
    code: str
    description: str
    severity: Severity
    detector: Detector
    injectable: bool = True
    legal_reference: str = ""


FAULTS: tuple[FaultSpec, ...] = (
    # --- mandatory content (German UStG s14(4)) -------------------------------------------------
    FaultSpec("MISSING_INVOICE_NUMBER", "Invoice number absent", Severity.CRITICAL, Detector.RULES,
              legal_reference="UStG s14(4) Nr.4"),
    FaultSpec("MISSING_INVOICE_DATE", "Invoice date absent", Severity.CRITICAL, Detector.RULES,
              legal_reference="UStG s14(4) Nr.3"),
    FaultSpec("MISSING_SUPPLIER_NAME", "Issuer name and address absent", Severity.CRITICAL,
              Detector.RULES, legal_reference="UStG s14(4) Nr.1"),
    FaultSpec("MISSING_SUPPLIER_VAT_ID", "Issuer VAT identification number absent", Severity.CRITICAL,
              Detector.RULES, legal_reference="UStG s14(4) Nr.2"),
    FaultSpec("MISSING_CUSTOMER_NAME", "Recipient name absent", Severity.MAJOR, Detector.RULES,
              legal_reference="UStG s14(4) Nr.1"),
    FaultSpec("MISSING_NET_AMOUNT", "Net amount absent where a split is required", Severity.CRITICAL,
              Detector.RULES, legal_reference="UStG s14(4) Nr.7"),
    FaultSpec("MISSING_VAT_RATE", "Tax rate absent where a split is required", Severity.MAJOR,
              Detector.RULES, legal_reference="UStG s14(4) Nr.7"),
    FaultSpec("MISSING_VAT_AMOUNT", "Tax amount absent where a split is required", Severity.MAJOR,
              Detector.RULES, legal_reference="UStG s14(4) Nr.7"),
    FaultSpec("MISSING_GROSS_AMOUNT", "Gross amount absent", Severity.CRITICAL, Detector.RULES,
              legal_reference="UStG s14(4) Nr.7"),
    FaultSpec("MISSING_PAYMENT_TERMS", "No due date or payment term stated", Severity.MAJOR,
              Detector.RULES, legal_reference="UStG s14(4) Nr.5"),
    FaultSpec("MISSING_SERVICE_DATE", "Neither service date nor service period stated", Severity.MAJOR,
              Detector.RULES, legal_reference="UStG s14(4) Nr.6"),

    # --- arithmetic ----------------------------------------------------------------------------
    FaultSpec("ARITHMETIC_GROSS_MISMATCH", "Net + VAT does not equal gross", Severity.CRITICAL,
              Detector.RULES),
    FaultSpec("ARITHMETIC_LINE_ITEMS_MISMATCH", "Line items do not sum to the stated net",
              Severity.CRITICAL, Detector.RULES),
    FaultSpec("ARITHMETIC_VAT_AMOUNT_WRONG", "VAT amount is inconsistent with net x rate",
              Severity.CRITICAL, Detector.RULES),

    # --- line-item pricing (work billed but not charged) ---------------------------------------
    # These catch the "10 hours of labour, priced at 0 EUR" family of mistakes: the quantity is
    # positive, so something was demonstrably delivered, but the money is missing or absurd.
    FaultSpec("LINE_AMOUNT_ZERO",
              "Line item has a positive quantity but a zero net amount", Severity.CRITICAL,
              Detector.RULES),
    FaultSpec("LINE_UNIT_PRICE_ZERO",
              "Line item has a positive quantity but a zero unit price", Severity.MAJOR,
              Detector.RULES),
    FaultSpec("LINE_QUANTITY_PRICE_INCONSISTENT",
              "Line net does not equal quantity x unit price", Severity.CRITICAL, Detector.RULES),
    FaultSpec("LINE_RATE_OUTLIER",
              "Effective unit price is far below the historical rate for this service",
              Severity.INFO, Detector.RULES),

    # --- identifiers ---------------------------------------------------------------------------
    FaultSpec("INVALID_VAT_ID_CHECKSUM", "VAT identification number fails its check digit",
              Severity.CRITICAL, Detector.RULES),
    FaultSpec("INVALID_IBAN_CHECKSUM", "IBAN fails the MOD-97 check", Severity.CRITICAL, Detector.RULES),
    FaultSpec("INVALID_IBAN_FORMAT", "IBAN has an impossible length or character set", Severity.MAJOR,
              Detector.RULES),
    FaultSpec("INVALID_INVOICE_NUMBER_FORMAT", "Invoice number breaks the issuing sequence scheme",
              Severity.MINOR, Detector.RULES),

    # --- dates ---------------------------------------------------------------------------------
    FaultSpec("DATE_IN_FUTURE", "Invoice date lies in the future", Severity.CRITICAL, Detector.RULES),
    FaultSpec("DATE_DELIVERY_AFTER_INVOICE", "Service date is later than the invoice date",
              Severity.MAJOR, Detector.RULES),
    FaultSpec("DATE_DUE_BEFORE_INVOICE", "Due date precedes the invoice date", Severity.MAJOR,
              Detector.RULES),
    FaultSpec("DATE_DUE_TERM_UNUSUAL", "Payment term is implausibly long or short", Severity.MINOR,
              Detector.RULES),

    # --- tax treatment -------------------------------------------------------------------------
    FaultSpec("REVERSE_CHARGE_NOTE_MISSING",
              "Zero-rated intra-EU supply without the required reverse-charge note",
              Severity.CRITICAL, Detector.RULES, legal_reference="UStG s14a(5)"),
    FaultSpec("VAT_RATE_UNEXPECTED", "Applied tax rate is not a permitted rate", Severity.CRITICAL,
              Detector.RULES, legal_reference="UStG s12"),
    FaultSpec("SMALL_AMOUNT_FIELDS_INCONSISTENT",
              "Small-amount invoice relief applied but fields are contradictory", Severity.MINOR,
              Detector.RULES, legal_reference="UStDV s33"),

    # --- commercial / master data --------------------------------------------------------------
    FaultSpec("UNKNOWN_SUPPLIER", "Supplier is not present in master data", Severity.MAJOR,
              Detector.RULES),
    FaultSpec("CURRENCY_MISMATCH", "Currency differs from the agreed contract currency",
              Severity.MAJOR, Detector.RULES),
    FaultSpec("UNKNOWN_CURRENCY", "Currency code is not a recognised ISO 4217 code", Severity.MAJOR,
              Detector.RULES),
    FaultSpec("NEGATIVE_AMOUNT", "A negative amount appears on an outgoing invoice", Severity.MAJOR,
              Detector.RULES),
    FaultSpec("AMOUNT_ABOVE_APPROVAL_THRESHOLD",
              "Gross amount exceeds the approval threshold for this cost centre", Severity.MAJOR,
              Detector.RULES),
    FaultSpec("AMOUNT_OUTLIER", "Gross amount is a statistical outlier for this supplier",
              Severity.INFO, Detector.RULES),
    FaultSpec("DUPLICATE_INVOICE", "Near-duplicate of an earlier invoice from the same supplier",
              Severity.CRITICAL, Detector.RULES),
    FaultSpec("IBAN_CHANGED_RECENTLY", "Bank details differ from master data", Severity.INFO,
              Detector.RULES),

    # --- extraction-layer health (never injected, always reported) ------------------------------
    FaultSpec("EXTRACTION_FAILED", "Fields could not be extracted reliably", Severity.INFO,
              Detector.LLM, injectable=False),
    FaultSpec("LLM_JUDGEMENT_FLAG", "Model flagged an ambiguous passage for human review",
              Severity.INFO, Detector.LLM, injectable=False),
)

FAULT_BY_CODE: dict[str, FaultSpec] = {spec.code: spec for spec in FAULTS}
INJECTABLE_CODES: tuple[str, ...] = tuple(spec.code for spec in FAULTS if spec.injectable)


def severity_of(code: str) -> Severity:
    spec = FAULT_BY_CODE.get(code)
    return spec.severity if spec else Severity.INFO


def describe(code: str) -> str:
    spec = FAULT_BY_CODE.get(code)
    return spec.description if spec else code
