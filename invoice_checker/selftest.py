"""Built-in checks: the parts of the system that must be provably correct.

These are deliberately assertions about *algorithms and rule behaviour*, not about model output.
If the checksum code or the arithmetic rules are wrong, every metric downstream is meaningless, so
they are the first thing to run and the first thing to show a supervisor.
"""
from __future__ import annotations

from datetime import date
from decimal import Decimal
from pathlib import Path

from . import checksums
from .config import THRESHOLDS
from .duplicates import find_duplicates
from .generate import (build_master_data, generate_corpus, render_document_text,
                       rule_fault_codes)
from .ingest import extract_text_from_pdf
from .models import Invoice, parse_decimal
from .pdfgen import write_pdf
from .rules import RulesEngine

PASS = "PASS"
FAIL = "FAIL"


def _check(results: list[tuple[str, bool, str]], name: str, condition: bool, detail: str = "") -> None:
    results.append((name, bool(condition), detail))
    status = PASS if condition else FAIL
    suffix = f"  ({detail})" if detail and not condition else ""
    print(f"[{status}] {name}{suffix}")


def run_selftest() -> list[tuple[str, bool, str]]:
    results: list[tuple[str, bool, str]] = []

    # --- checksums ------------------------------------------------------------------------------
    _check(results, "IBAN DE89370400440532013000 is valid",
           checksums.iban_is_valid("DE89370400440532013000"))
    _check(results, "IBAN with a corrupted check digit is rejected",
           not checksums.iban_is_valid("DE89370400440532013001"))
    _check(results, "IBAN with the wrong length is rejected",
           not checksums.iban_is_valid("DE893704004405320130"))
    generated_iban = checksums.make_iban_de("12030000", "0020205151")
    _check(results, "generated IBAN passes its own validator",
           checksums.iban_is_valid(generated_iban), generated_iban)

    _check(results, "DE123456788 is a valid German VAT ID",
           checksums.vat_id_checksum_de("DE123456788"))
    _check(results, "German VAT ID with a corrupted digit is rejected",
           not checksums.vat_id_checksum_de("DE123456789"))
    generated_vat = checksums.make_vat_id_de("12345678")
    _check(results, "generated VAT ID passes its own validator",
           checksums.vat_id_checksum_de(generated_vat), generated_vat)
    _check(results, "permitted VAT rates are 0, 7 and 19 percent",
           checksums.vat_rate_is_permitted(0.19) and checksums.vat_rate_is_permitted(0.07)
           and checksums.vat_rate_is_permitted(0.0)
           and not checksums.vat_rate_is_permitted(0.16))

    # --- coercion -------------------------------------------------------------------------------
    _check(results, "German money format 1.234,56 parses to 1234.56",
           parse_decimal("1.234,56") == Decimal("1234.56"),
           str(parse_decimal("1.234,56")))
    _check(results, "English money format 1,234.56 parses to 1234.56",
           parse_decimal("1,234.56") == Decimal("1234.56"),
           str(parse_decimal("1,234.56")))
    _check(results, "currency symbol is stripped before parsing",
           parse_decimal("€ 190,00") == Decimal("190.00"))
    _check(results, "vat rate 19 is normalised to 0.19",
           Invoice.from_dict({"vat_rate": 19}).vat_rate == Decimal("0.19"))
    _check(results, "the literal string 'null' from a model is treated as absent",
           Invoice.from_dict({"invoice_number": "null"}).invoice_number is None)

    # --- constrained-output schema validator ----------------------------------------------------
    # These guard the extraction contract. A loosened validator silently turns model mistakes into
    # confident garbage, so each of these is checked explicitly.
    from .extract import INVOICE_JSON_SCHEMA
    from .llm import validate_against_schema

    valid_payload = {
        "invoice_number": "X", "invoice_date": "2024-01-01", "service_date": "",
        "supplier_name": "A", "supplier_vat_id": "DE1", "customer_name": "B",
        "customer_vat_id": "DE2", "net_amount": 1.0, "vat_rate": 0.19, "vat_amount": 0.19,
        "gross_amount": 1.19, "currency": "EUR", "due_date": "", "iban": "",
    }
    _check(results, "schema accepts a well-formed extraction",
           not validate_against_schema(valid_payload, INVOICE_JSON_SCHEMA))
    _check(results, "schema accepts empty strings for absent optional fields",
           not validate_against_schema(
               {**valid_payload, "service_date": "", "due_date": "", "iban": ""},
               INVOICE_JSON_SCHEMA))
    _check(results, "schema rejects a non-ISO date",
           any("YYYY-MM-DD" in error for error in validate_against_schema(
               {**valid_payload, "invoice_date": "14.03.2024"}, INVOICE_JSON_SCHEMA)))
    _check(results, "schema rejects a missing required field",
           any("required field missing" in error for error in validate_against_schema(
               {key: value for key, value in valid_payload.items() if key != "currency"},
               INVOICE_JSON_SCHEMA)))
    _check(results, "schema rejects an unexpected field",
           any("unexpected field" in error for error in validate_against_schema(
               {**valid_payload, "sneaky": 1}, INVOICE_JSON_SCHEMA)))
    _check(results, "schema rejects a number delivered as a string",
           any("expected number" in error for error in validate_against_schema(
               {**valid_payload, "net_amount": "1.00"}, INVOICE_JSON_SCHEMA)))
    _check(results, "schema rejects null for a non-nullable field",
           bool(validate_against_schema(
               {**valid_payload, "invoice_number": None}, INVOICE_JSON_SCHEMA)))

    # --- PDF text layer -------------------------------------------------------------------------
    lines = ["Rechnung Nr. INV-1", "Gesamtbetrag brutto: 1.190,00 EUR",
             "Pruefung: Umlaute äöüß und Euro €"]
    scratch = Path(__file__).resolve().parent.parent / ".selftest_tmp"
    scratch.mkdir(parents=True, exist_ok=True)
    try:
        pdf_path = scratch / "probe.pdf"
        write_pdf(pdf_path, lines)
        extracted = extract_text_from_pdf(pdf_path)
        _check(results, "PDF text layer round-trips every line",
               all(line in extracted for line in lines),
               extracted.replace("\n", " | ")[:160])
        _check(results, "PDF extraction result is not empty", bool(extracted.strip()))
    finally:
        for leftover in scratch.glob("*"):
            leftover.unlink(missing_ok=True)
        scratch.rmdir()

    # --- rule engine ----------------------------------------------------------------------------
    master = build_master_data(seed=20240517)
    engine = RulesEngine(master)
    profile = master["suppliers"][0]

    clean = Invoice(
        invoice_number="NOR-2024-0001", invoice_date=date(2024, 5, 2),
        service_date=date(2024, 4, 30), supplier_name=profile["name"],
        supplier_address=profile["address"], supplier_vat_id=profile["vat_id"],
        customer_name=master["our_company"]["name"], customer_vat_id=master["our_company"]["vat_id"],
        net_amount=Decimal("1000.00"), vat_rate=Decimal("0.19"), vat_amount=Decimal("190.00"),
        gross_amount=Decimal("1190.00"), currency="EUR", payment_terms_days=30,
        due_date=date(2024, 6, 1), iban=profile["iban"],
    )
    clean_codes = [fault.code for fault in engine.check_invoice(clean)]
    _check(results, "a correct invoice produces no faults", not clean_codes, str(clean_codes))

    broken = Invoice.from_dict(clean.to_dict())
    broken.gross_amount = Decimal("1200.00")
    broken_codes = {fault.code for fault in engine.check_invoice(broken)}
    _check(results, "net + VAT mismatch is detected", "ARITHMETIC_GROSS_MISMATCH" in broken_codes,
           str(sorted(broken_codes)))

    wrong_vat = Invoice.from_dict(clean.to_dict())
    wrong_vat.vat_amount = Decimal("160.00")
    wrong_vat.gross_amount = Decimal("1160.00")
    wrong_codes = {fault.code for fault in engine.check_invoice(wrong_vat)}
    _check(results, "VAT amount inconsistent with net x rate is detected",
           "ARITHMETIC_VAT_AMOUNT_WRONG" in wrong_codes, str(sorted(wrong_codes)))

    future = Invoice.from_dict(clean.to_dict())
    future.invoice_date = date(2024, 12, 1)
    future.due_date = date(2025, 1, 1)
    _check(results, "future-dated invoice is detected",
           "DATE_IN_FUTURE" in {fault.code for fault in engine.check_invoice(future)})

    unsigned = Invoice.from_dict(clean.to_dict())
    unsigned.supplier_vat_id = None
    unsigned.supplier_tax_number = None
    _check(results, "missing VAT identification number is detected",
           "MISSING_SUPPLIER_VAT_ID" in {fault.code for fault in engine.check_invoice(unsigned)})

    # A negative amount must not be "fixed" by the arithmetic rule: it is a separate finding.
    negative = Invoice.from_dict(clean.to_dict())
    negative.gross_amount = Decimal("-1190.00")
    negative_codes = {fault.code for fault in engine.check_invoice(negative)}
    _check(results, "negative gross amount is detected", "NEGATIVE_AMOUNT" in negative_codes)
    _check(results, "negative gross amount also breaks the arithmetic chain",
           "ARITHMETIC_GROSS_MISMATCH" in negative_codes)

    # Small-amount relief (UStDV s33): gross-only is acceptable, but a partial split is not.
    small = Invoice.from_dict(clean.to_dict())
    small.net_amount = small.vat_amount = small.vat_rate = None
    small.gross_amount = Decimal("180.00")
    small_codes = rule_fault_codes(small, master, THRESHOLDS)
    _check(results, "gross-only small-amount invoice raises no missing-field fault",
           not [code for code in small_codes if code.startswith("MISSING_")], str(small_codes))
    partial = Invoice.from_dict(small.to_dict())
    partial.net_amount = Decimal("168.22")
    _check(results, "partial tax split on a small-amount invoice is flagged",
           "SMALL_AMOUNT_FIELDS_INCONSISTENT"
           in {fault.code for fault in engine.check_invoice(partial)})

    # --- line-item pricing: work delivered but not charged for ----------------------------------
    from .models import LineItem
    from .rules import build_rate_index

    def with_lines(items, **overrides):
        base = Invoice.from_dict(clean.to_dict())
        base.line_items = items
        for key, value in overrides.items():
            setattr(base, key, value)
        return base

    hourly = [
        LineItem(description="Softwareentwicklung", quantity=Decimal("10"),
                 unit_price=Decimal("95.00"), line_net=Decimal("950.00")),
    ]

    # The motivating defect: 10 hours of labour, none of it charged.
    zero_billed = with_lines([
        LineItem(description="Softwareentwicklung", quantity=Decimal("10"),
                 unit_price=Decimal("0.00"), line_net=Decimal("0.00")),
    ], net_amount=Decimal("0.00"), vat_amount=Decimal("0.00"), gross_amount=Decimal("0.00"))
    zero_codes = {fault.code for fault in engine.check_invoice(zero_billed)}
    _check(results, "10 hours billed at a zero line amount is detected",
           "LINE_AMOUNT_ZERO" in zero_codes, str(sorted(zero_codes)))
    _check(results, "10 hours billed at a zero unit price is detected",
           "LINE_UNIT_PRICE_ZERO" in zero_codes, str(sorted(zero_codes)))

    # A positive quantity with a *missing* price must still be caught, because the field is what a
    # human would fix and an extraction layer may simply have dropped it.
    missing_price = with_lines([
        LineItem(description="Beratung", quantity=Decimal("10"), unit_price=None,
                 line_net=Decimal("0.00")),
    ])
    _check(results, "positive quantity with a zero net but no price is still flagged",
           "LINE_AMOUNT_ZERO" in {fault.code for fault in engine.check_invoice(missing_price)})

    # A genuine free-of-charge line (quantity zero) must NOT be flagged: over-reporting legitimate
    # goodwill lines is the failure mode that would make the checker unusable.
    goodwill = with_lines([
        LineItem(description="Material", quantity=Decimal("0"), unit_price=Decimal("0.00"),
                 line_net=Decimal("0.00")),
    ], net_amount=Decimal("0.00"), vat_amount=Decimal("0.00"), gross_amount=Decimal("0.00"))
    goodwill_codes = {fault.code for fault in engine.check_invoice(goodwill)}
    _check(results, "a zero-quantity goodwill line is not reported as a pricing fault",
           not {"LINE_AMOUNT_ZERO", "LINE_UNIT_PRICE_ZERO"} & goodwill_codes,
           str(sorted(goodwill_codes)))

    # quantity x unit price must agree with the line net.
    inconsistent = with_lines([
        LineItem(description="Wartung", quantity=Decimal("10"), unit_price=Decimal("80.00"),
                 line_net=Decimal("80.00")),
    ], net_amount=Decimal("80.00"), vat_amount=Decimal("15.20"), gross_amount=Decimal("95.20"))
    _check(results, "line net inconsistent with quantity x unit price is detected",
           "LINE_QUANTITY_PRICE_INCONSISTENT"
           in {fault.code for fault in engine.check_invoice(inconsistent)})

    # A decimal-point error: the rate is a tenth of the historical median.
    history = [
        with_lines([LineItem(description="Projektmanagement", quantity=Decimal("1"),
                             unit_price=Decimal("110.00"), line_net=Decimal("110.00"))])
        for _ in range(6)
    ]
    index = build_rate_index(history)
    _check(results, "rate index learns a median from history",
           index.get("projektmanagement", {}).get("median") == 110.0, str(index))
    undercharged = with_lines([
        LineItem(description="Projektmanagement", quantity=Decimal("10"),
                 unit_price=Decimal("11.00"), line_net=Decimal("110.00")),
    ], net_amount=Decimal("110.00"), vat_amount=Decimal("20.90"), gross_amount=Decimal("130.90"))
    _check(results, "a rate an order of magnitude below the historical median is flagged",
           "LINE_RATE_OUTLIER"
           in {fault.code for fault in engine.check_invoice(undercharged, rate_index=index)})

    # The same line at the normal rate must stay clean: the rule must not fire on everything.
    _check(results, "the historical rate itself is not flagged",
           "LINE_RATE_OUTLIER"
           not in {fault.code for fault in engine.check_invoice(history[0], rate_index=index)})
    _check(results, "a correctly priced invoice with line items produces no pricing fault",
           not {"LINE_AMOUNT_ZERO", "LINE_UNIT_PRICE_ZERO", "LINE_QUANTITY_PRICE_INCONSISTENT"}
           & {fault.code for fault in engine.check_invoice(with_lines(hourly))})

    # --- service: German report and durable state -------------------------------------------------
    from datetime import date as _date

    from .i18n_de import _BUILDERS, de_amount, de_date, de_percent, explain
    from .models import CheckResult, Fault
    from .service_report import _legal_de, render_report, report_filename
    from .service_state import StateStore

    _check(results, "German amount formatting uses dot-thousands and comma-decimals",
           de_amount("1234.5") == "1.234,50", de_amount("1234.5"))
    _check(results, "German amount formatting includes the currency when given",
           de_amount("1190", "EUR") == "1.190,00 EUR", de_amount("1190", "EUR"))
    _check(results, "German date formatting is DD.MM.YYYY",
           de_date("2024-03-14") == "14.03.2024" and de_date(_date(2024, 3, 14)) == "14.03.2024")
    _check(results, "German percentage formatting drops trailing zeros",
           de_percent("0.19") == "19 %" and de_percent("0.07") == "7 %",
           f"{de_percent('0.19')} / {de_percent('0.07')}")
    _check(results, "a legal reference is rendered the German way",
           _legal_de("UStG s14(4) Nr.4") == "§ 14 Abs. 4 Nr. 4 UStG",
           _legal_de("UStG s14(4) Nr.4"))
    _check(results, "a small-amount reference is rendered correctly",
           _legal_de("UStDV s33") == "§ 33 UStDV", _legal_de("UStDV s33"))

    from .taxonomy import FAULT_BY_CODE as _FAULTS

    unexplained = sorted(set(_FAULTS) - set(_BUILDERS))
    _check(results, "every fault code has a German explanation", not unexplained, str(unexplained))

    # A builder that throws must degrade to a generic sentence, never lose the finding.
    sample_evidence = {
        "net": "100.00", "vat": "19.00", "expected_gross": "119.00", "stated_gross": "120.00",
        "difference": "1.00", "currency": "EUR", "quantity": "10", "unit_price": "0",
        "line_net": "0", "description": "Beratung", "line": "1", "iban": "DE00",
        "vat_id": "DE1", "value": "X", "pattern": "P", "rate": "0.19", "supplier": "A",
        "gross": "1", "threshold": "2", "limit": "250", "term_days": "180", "max_days": "90",
    }
    generic = [code for code in sorted(_BUILDERS)
               if "Regelcode" in explain(code, sample_evidence, severity="critical")]
    _check(results, "German explanations are specific rather than generic fallbacks",
           not generic, str(generic))

    fault = Fault(code="LINE_AMOUNT_ZERO", severity="critical",
                  message="positive quantity, zero amount",
                  evidence={"line": 2, "description": "Beratung", "quantity": "10",
                            "line_net": "0.00", "currency": "EUR"})
    report_invoice = Invoice.from_dict({
        "invoice_number": "R-1", "invoice_date": "2024-03-14", "supplier_name": "ACME GmbH",
        "supplier_vat_id": "DE123456788", "customer_name": "Kunde GmbH",
        "net_amount": "1000.00", "vat_rate": "0.19", "vat_amount": "190.00",
        "gross_amount": "1190.00", "currency": "EUR", "line_items": [],
    })
    report_text = render_report(CheckResult(doc_id="rechnung.pdf", faults=[fault]),
                                report_invoice, source_name="rechnung.pdf",
                                folder_label="eingang", extraction="text+regex",
                                processing_seconds=1.25)
    missing_needles = [needle for needle in
                       ("PRÜFBERICHT", "ERGEBNIS: FEHLERHAFT", "rechnung.pdf", "Kritisch",
                        "Position 2", "Beratung", "0,00 EUR", "Bitte")
                       if needle not in report_text]
    _check(results, "German report states verdict, finding, offending values and the fix",
           not missing_needles, str(missing_needles))
    _check(results, "report filename is derived from the invoice name",
           report_filename("rechnung-42.pdf") == "rechnung-42.fehlerbericht.txt",
           report_filename("rechnung-42.pdf"))

    # State store: dedupe on (path, hash); an edited file must count as new.
    scratch_dir = Path(__file__).resolve().parent.parent / ".selftest_state"
    scratch_dir.mkdir(parents=True, exist_ok=True)
    db_path = scratch_dir / "state.sqlite3"
    doc_path = scratch_dir / "doc.txt"
    try:
        db_path.unlink(missing_ok=True)
        doc_path.write_text("first version", encoding="utf-8")
        with StateStore(db_path) as store:
            sha_one = StateStore.hash_file(doc_path)
            _check(results, "an unseen document has no prior record",
                   store.previous(doc_path, sha_one) is None)
            store.record(path=doc_path, sha256=sha_one, folder="test", original_name="doc.txt",
                         verdict="faulty", fault_codes=["LINE_AMOUNT_ZERO"], severity="critical",
                         destination="fehlerhaft/doc.txt", report_path="fehlerhaft/doc.report.txt",
                         duration_ms=12)
            _check(results, "a processed document is recognised on the next pass",
                   store.previous(doc_path, sha_one) is not None)
            doc_path.write_text("second, corrected version", encoding="utf-8")
            sha_two = StateStore.hash_file(doc_path)
            _check(results, "an edited document counts as new and is re-checked",
                   sha_one != sha_two and store.previous(doc_path, sha_two) is None)
            _check(results, "hashing identical content is stable",
                   StateStore.hash_file(doc_path) == sha_two)
    finally:
        for leftover in scratch_dir.glob("*"):
            leftover.unlink(missing_ok=True)
        scratch_dir.rmdir()

    # --- duplicates -----------------------------------------------------------------------------
    original = Invoice.from_dict(clean.to_dict())
    clone = Invoice.from_dict(clean.to_dict())
    clone.invoice_number = "NOR-2024-0099"
    findings = find_duplicates([("a", original), ("b", clone)], strategy="fuzzy")
    _check(results, "same supplier, amount and date with a new number is a near-duplicate",
           any(finding.doc_id == "b" for finding in findings), str(findings))
    exact = Invoice.from_dict(clean.to_dict())
    findings = find_duplicates([("a", original), ("b", exact)], strategy="exact")
    _check(results, "identical invoice numbers are duplicate findings",
           any(finding.doc_id == "b" for finding in findings))

    # --- corpus-level false-positive control ----------------------------------------------------
    documents = generate_corpus(120, seed=99, fault_rate=0.0, include_duplicates=False,
                                master=master)
    false_positives: dict[str, int] = {}
    for generated in documents:
        invoice = Invoice.from_dict(generated.clean.to_dict())
        for fault in engine.check_invoice(invoice):
            false_positives[fault.code] = false_positives.get(fault.code, 0) + 1
    _check(results, "clean corpus produces no false positives",
           not false_positives, str(false_positives))

    _check(results, "generated documents render a non-empty body",
           all(len(generated.doc_text) > 200 for generated in documents[:10]))

    return [(name, ok, detail) for name, ok, detail in results]
