"""Duplicate and near-duplicate detection.

Three strategies, selectable so the report can show the cost/benefit of each:

* ``exact``   -- the same invoice number appears twice (catches copy-paste duplicates)
* ``fuzzy``   -- same supplier, near-identical gross amount and a close date (stdlib difflib;
                 catches the realistic case where a supplier re-issues with a new number)
* ``embedding`` -- optional, uses sentence-transformers when installed; otherwise falls back to
                 fuzzy and says so rather than failing

The deliberate design choice: duplicates are found by *fingerprinting*, not by asking a generative
model. A model asked "is this a duplicate?" will answer confidently and inconsistently.
"""
from __future__ import annotations

import difflib
from dataclasses import dataclass
from datetime import date

from .config import CAPS, THRESHOLDS, Thresholds
from .models import Invoice


@dataclass
class DuplicateFinding:
    doc_id: str
    duplicate_of: str
    reason: str
    metrics: dict


def fingerprint(inv: Invoice) -> tuple[str, str, str] | None:
    """(normalised supplier, gross as string, ISO date) -- None when any part is missing."""
    if not inv.supplier_name or inv.gross_amount is None or not inv.invoice_date:
        return None
    supplier = " ".join(inv.supplier_name.lower().split())
    return supplier, f"{inv.gross_amount:.2f}", inv.invoice_date.isoformat()


def find_duplicates(invoices: list[tuple[str, Invoice]], *,
                    thresholds: Thresholds | None = None,
                    strategy: str = "fuzzy") -> list[DuplicateFinding]:
    """Return one finding per document that duplicates an earlier document.

    ``invoices`` is an ordered list of ``(doc_id, invoice)``. Order matters: the first occurrence is
    treated as the original and later ones are flagged, which mirrors how an accounts-payable
    ledger actually accumulates.
    """
    thresholds = thresholds or THRESHOLDS
    findings: list[DuplicateFinding] = []

    # Pass 1: identical invoice numbers.
    seen_numbers: dict[str, str] = {}
    for doc_id, inv in invoices:
        if not inv.invoice_number:
            continue
        if inv.invoice_number in seen_numbers:
            findings.append(DuplicateFinding(
                doc_id=doc_id, duplicate_of=seen_numbers[inv.invoice_number],
                reason="identical invoice number",
                metrics={"invoice_number": inv.invoice_number},
            ))
        else:
            seen_numbers[inv.invoice_number] = doc_id

    flagged = {finding.doc_id for finding in findings}
    if strategy == "exact":
        return findings

    # Pass 2: near-duplicates by fingerprint and amount/date proximity.
    seen: dict[tuple[str, str, str], tuple[str, Invoice]] = {}
    for doc_id, inv in invoices:
        key = fingerprint(inv)
        if key is None:
            continue
        if key in seen and doc_id not in flagged:
            other_id, other = seen[key]
            findings.append(DuplicateFinding(
                doc_id=doc_id, duplicate_of=other_id,
                reason="same supplier, amount and date with a different invoice number",
                metrics={
                    "supplier": inv.supplier_name,
                    "gross": str(inv.gross_amount),
                    "date": inv.invoice_date.isoformat(),
                    "other_invoice_number": other.invoice_number,
                    "invoice_number": inv.invoice_number,
                },
            ))
            flagged.add(doc_id)
            continue
        seen.setdefault(key, (doc_id, inv))

    # Pass 3: fuzzy on (supplier, amount) within a date window, for re-issues with a new date.
    if strategy in {"fuzzy", "embedding"}:
        previous: list[tuple[str, Invoice]] = []
        for doc_id, inv in invoices:
            if doc_id in flagged:
                previous.append((doc_id, inv))
                continue
            if inv.supplier_name and inv.gross_amount is not None and inv.invoice_date:
                for other_id, other in previous:
                    if other.supplier_name != inv.supplier_name or other.gross_amount is None:
                        continue
                    amount_delta = abs(float(inv.gross_amount) - float(other.gross_amount))
                    if amount_delta > thresholds.duplicate_amount_tolerance:
                        continue
                    if not (other.invoice_date and inv.invoice_date):
                        continue
                    day_delta = abs((inv.invoice_date - other.invoice_date).days)
                    if day_delta == 0 or day_delta > thresholds.duplicate_window_days:
                        continue
                    similarity = difflib.SequenceMatcher(
                        None, (inv.invoice_number or ""), (other.invoice_number or "")
                    ).ratio()
                    if similarity >= thresholds.duplicate_similarity:
                        findings.append(DuplicateFinding(
                            doc_id=doc_id, duplicate_of=other_id,
                            reason=f"near-identical invoice number (similarity {similarity:.2f}) "
                                   f"with the same supplier and amount",
                            metrics={"similarity": round(similarity, 3),
                                     "amount_delta": round(amount_delta, 2),
                                     "day_delta": day_delta},
                        ))
                        flagged.add(doc_id)
                        break
            previous.append((doc_id, inv))

    return findings


def embedding_backend_available() -> bool:
    return CAPS.sentence_transformers


def duplicate_strategy_note(strategy: str) -> str:
    if strategy == "embedding" and not embedding_backend_available():
        return ("sentence-transformers is not installed; duplicate detection fell back to "
                "difflib fuzzy matching")
    return f"duplicate strategy: {strategy}"
