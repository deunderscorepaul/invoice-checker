"""Check-digit algorithms and ISO code tables.

Pure stdlib so the whole rule engine runs anywhere. These are the checks that make an LLM
unnecessary for identifier validation -- and that a generative model reliably gets wrong.
"""
from __future__ import annotations

import re

IBAN_LENGTHS = {
    "DE": 22, "AT": 20, "CH": 21, "FR": 27, "NL": 18, "BE": 16, "IT": 27, "ES": 24,
    "PL": 28, "CZ": 24, "DK": 18, "SE": 24, "FI": 18, "NO": 15, "PT": 25, "IE": 22,
    "LU": 20, "GR": 27, "HU": 28, "RO": 24, "SK": 24, "SI": 19, "BG": 22, "HR": 21,
    "EE": 20, "LV": 21, "LT": 20, "MT": 31, "CY": 28, "GB": 22, "US": 0,
}

ISO_CURRENCIES = {
    "EUR", "USD", "GBP", "CHF", "SEK", "NOK", "DKK", "PLN", "CZK", "HUF", "RON",
    "BGN", "HRK", "TRY", "JPY", "CNY", "CAD", "AUD", "NZD", "INR", "ZAR", "MXN", "BRL",
}

# Rates permitted under UStG s12 (19% standard, 7% reduced) plus 0% for reverse charge / exports.
PERMITTED_VAT_RATES = {0.0, 0.07, 0.19}


def vat_id_checksum_de(vat_id: str) -> bool:
    """Validate a German USt-IdNr (ISO 7064 MOD 11,10 style, as used for DE VAT IDs).

    Reference procedure: take digits 1-9 as the body, produce a check digit with the
    product-sum method, compare it with digit 10.
    """
    cleaned = re.sub(r"[^0-9A-Za-z]", "", vat_id or "").upper()
    if not cleaned.startswith("DE"):
        return False
    digits = cleaned[2:]
    if not digits.isdigit() or len(digits) != 9:
        return False
    product = 10
    for char in digits[:8]:
        total = (int(char) + product) % 10
        if total == 0:
            total = 10
        product = (total * 2) % 11
    check = (11 - product) % 10
    return check == int(digits[8])


def vat_id_check_digit_de(body8: str) -> int:
    """Inverse of :func:`vat_id_checksum_de` -- used by the synthetic generator."""
    product = 10
    for char in body8[:8]:
        total = (int(char) + product) % 10
        if total == 0:
            total = 10
        product = (total * 2) % 11
    return (11 - product) % 10


def make_vat_id_de(body8: str) -> str:
    """Build a valid DE VAT id from the first 8 digits.

    The check digit is computed over digits 1-8 and occupies position 9, giving the canonical
    ``DE`` plus 9 digits layout. An earlier implementation placed it at position 10, producing IDs
    that failed the validator -- caught by the clean-corpus false-positive control in the selftest.
    """
    body8 = re.sub(r"\D", "", body8)[:8].ljust(8, "0")
    return f"DE{body8}{vat_id_check_digit_de(body8)}"


def iban_mod97(iban: str) -> int:
    """Return the MOD-97 remainder of an IBAN (1 == valid)."""
    cleaned = re.sub(r"\s+", "", iban or "").upper()
    rearranged = cleaned[4:] + cleaned[:4]
    digits = "".join(str(int(ch, 36)) for ch in rearranged if ch.isalnum())
    if not digits:
        return -1
    remainder = 0
    for char in digits:
        remainder = (remainder * 10 + int(char)) % 97
    return remainder


def iban_is_valid(iban: str) -> bool:
    cleaned = re.sub(r"\s+", "", iban or "").upper()
    if len(cleaned) < 5:
        return False
    country = cleaned[:2]
    if not country.isalpha():
        return False
    expected = IBAN_LENGTHS.get(country)
    if expected and len(cleaned) != expected:
        return False
    if not re.fullmatch(r"[A-Z]{2}[0-9]{2}[A-Z0-9]+", cleaned):
        return False
    return iban_mod97(cleaned) == 1


def iban_check_digits(country: str, bban: str) -> str:
    """Compute the two check digits for a country + BBAN (used by the generator)."""
    rearranged = f"{bban}{country}00".upper()
    digits = "".join(str(int(ch, 36)) if ch.isalnum() else "" for ch in rearranged)
    remainder = 0
    for char in digits:
        remainder = (remainder * 10 + int(char)) % 97
    check = 98 - remainder
    return f"{check:02d}"


def make_iban_de(bank_code8: str, account10: str) -> str:
    bban = f"{bank_code8}{account10}"
    return f"DE{iban_check_digits('DE', bban)}{bban}"


def iban_country_length_ok(iban: str) -> bool:
    cleaned = re.sub(r"\s+", "", iban or "").upper()
    expected = IBAN_LENGTHS.get(cleaned[:2])
    return bool(expected) and len(cleaned) == expected


def currency_is_iso(code: str | None) -> bool:
    return bool(code) and code.strip().upper() in ISO_CURRENCIES


def vat_rate_is_permitted(rate: float | None, tolerance: float = 1e-6) -> bool:
    if rate is None:
        return True
    return any(abs(rate - allowed) <= tolerance for allowed in PERMITTED_VAT_RATES)
