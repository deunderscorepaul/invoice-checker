"""Central configuration: paths, optional-dependency detection, runtime knobs."""
from __future__ import annotations

import importlib.util
import json
import os
from dataclasses import dataclass, field, asdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
SYNTH_DIR = DATA_DIR / "synth"
OUT_DIR = DATA_DIR / "out"
MASTER_FILE = DATA_DIR / "master_data.json"

DEFAULT_LLM_MODEL = os.environ.get("INVOICE_CHECKER_MODEL", "qwen2.5:7b-instruct")
DEFAULT_OLLAMA_URL = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434")
DEFAULT_SEED = 20240517


def has_module(name: str) -> bool:
    """True if a module is importable, without importing it."""
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


@dataclass
class Capabilities:
    pytesseract: bool = field(default_factory=lambda: has_module("pytesseract"))
    pypdf: bool = field(default_factory=lambda: has_module("pypdf"))
    sentence_transformers: bool = field(default_factory=lambda: has_module("sentence_transformers"))
    pandas: bool = field(default_factory=lambda: has_module("pandas"))

    def as_dict(self) -> dict:
        return asdict(self)

    def summary(self) -> str:
        lines = ["optional dependency status:"]
        for key, ok in self.as_dict().items():
            lines.append(f"  {'OK   ' if ok else 'MISS '} {key}")
        if not self.pytesseract:
            lines.append("  note: OCR disabled - scanned images will be skipped, digital PDFs still work")
        if not self.sentence_transformers:
            lines.append("  note: duplicate detection will use difflib fuzzy matching")
        return "\n".join(lines)


CAPS = Capabilities()


@dataclass
class Thresholds:
    """Tunable decision points, kept in one place so they can be ablated in the report."""
    # Arithmetic
    amount_tolerance: float = 0.01          # currency rounding tolerance (EUR)
    # Small-amount invoice relief, German UStDV s33: <= 250 EUR gross may omit net/VAT split
    small_amount_limit: float = 250.0
    # Duplicates
    duplicate_amount_tolerance: float = 1.00
    duplicate_window_days: int = 120
    duplicate_similarity: float = 0.90
    # Outliers
    outlier_zscore: float = 3.0
    # Fallback for suppliers with too few invoices for a meaningful standard deviation.
    outlier_ratio: float = 5.0
    outlier_min_samples: int = 3
    # --- line-item pricing ----------------------------------------------------------------------
    # A tiny line net is treated as "billed at zero" only when the *net* itself rounds away, because
    # a legitimate 0.03 EUR line is not a pricing error. A positive net with an absurd effective
    # rate is reported separately as LINE_RATE_OUTLIER.
    line_net_zero_threshold: float = 0.005
    # Effective rate below this fraction of the historical median for the same service is flagged.
    # 0.25 tolerates generous discounts while catching decimal-point and omission errors.
    line_rate_ratio: float = 0.25
    line_rate_min_samples: int = 5
    # Extraction
    extraction_confidence_floor: float = 0.60
    # Dates
    max_future_days: int = 0                # an invoice dated in the future is a fault
    # Longest commercially plausible payment term. Must be above the generator's normal range
    # (14-60 days) and below the 180 days it injects as a fault.
    max_payment_term_days: int = 90
    max_due_after_invoice_days: int = 365


THRESHOLDS = Thresholds()


def ensure_dirs() -> None:
    for path in (DATA_DIR, SYNTH_DIR, OUT_DIR):
        path.mkdir(parents=True, exist_ok=True)


def dump_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def load_json(path: Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def build_and_load_master(path: Path | None = None) -> dict:
    """Load supplier/master data, generating and persisting a default set if none exists.

    The service needs *some* master data to check suppliers, approval thresholds and contract
    currencies against. Shipping a sensible default means the service works out of the box; a real
    deployment replaces ``master_data.json`` with its own supplier list.
    """
    from .generate import build_master_data

    target = Path(path) if path else MASTER_FILE
    if target.exists():
        try:
            return load_json(target)
        except (OSError, json.JSONDecodeError):
            pass
    master = build_master_data(DEFAULT_SEED)
    dump_json(target, master)
    return master


def write_jsonl(path: Path, records) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1
    return count


def read_jsonl(path: Path):
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)
