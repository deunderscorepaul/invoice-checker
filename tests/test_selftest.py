"""Pytest wrappers around the built-in checks, so CI runs the same assertions as ``selftest``.

The checks are generated once per session: running the whole suite twice would rebuild the
false-positive control corpus for nothing.
"""
from __future__ import annotations

import pytest

from invoice_checker import selftest as selftest_module


@pytest.fixture(scope="session")
def check_results():
    return {name: ok for name, ok, _ in selftest_module.run_selftest()}


def test_checksums_and_rules_all_pass(check_results):
    failures = [name for name, ok in check_results.items() if not ok]
    assert not failures, "failed checks: " + "; ".join(failures)


def test_no_false_positives_on_clean_corpus(check_results):
    assert check_results.get("clean corpus produces no false positives") is True
