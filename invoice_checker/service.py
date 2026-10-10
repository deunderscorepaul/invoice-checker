"""Folder-watching service.

Ablauf pro Datei:
  1. Datei in .processing/ verschieben (vor dem Lesen, damit nichts verloren geht)
  2. Text lesen, Felder extrahieren, Regeln prüfen
  3. fehlerhaft -> fehlerhaft/ + deutscher Bericht, sonst liegen lassen

Bereits geprüfte Dateien werden über (Pfad, sha256) in SQLite erkannt. Wird eine
Rechnung korrigiert und erneut abgelegt, ändert sich der Hash und sie wird neu
geprüft -- genau das ist der Sinn des Workflows.
"""
from __future__ import annotations

import logging
import shutil
import signal
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from .config import MASTER_FILE, build_and_load_master
from .extract import extract_with_fallback
from .ingest import IngestError, ingest_document
from .llm import LLMError, OllamaClient
from .models import CheckResult, Invoice
from .rules import RulesEngine, build_rate_index
from .service_config import ServiceConfig, WatchFolder, load_config
from .service_report import render_report, report_filename
from .service_state import StateStore
from .taxonomy import INJECTABLE_CODES

PROCESSING_DIRNAME = ".processing"
UNREADABLE_CODES = {"EXTRACTION_FAILED"}


@dataclass
class DocumentOutcome:
    path: Path
    original_name: str
    verdict: str                      # clean | faulty | skipped | error
    faults: list = field(default_factory=list)
    check: CheckResult | None = None
    invoice: Invoice | None = None
    destination: Path | None = None
    report_path: Path | None = None
    detail: str = ""
    duration_ms: int = 0
    extraction: str = ""

    @property
    def codes(self) -> list[str]:
        return [fault.code for fault in self.faults]


# --- helpers ------------------------------------------------------------------------------------

def enable_utf8_console() -> None:
    """stdout/stderr auf UTF-8 stellen, sonst kommen Umlaute auf der Windows-Konsole kaputt an."""
    import sys

    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError, OSError):
            # Kein umkonfigurierbarer Stream (z.B. in Tests umgeleitet) -- dann eben nicht.
            pass


def setup_logging(config: ServiceConfig, *, verbose: bool = True) -> logging.Logger:
    enable_utf8_console()
    logger = logging.getLogger("invoice_checker.service")
    logger.setLevel(logging.DEBUG if verbose else logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s  %(levelname)-7s %(message)s",
                                  datefmt="%Y-%m-%d %H:%M:%S")
    console = logging.StreamHandler()
    console.setFormatter(formatter)
    logger.addHandler(console)

    if config.log_path:
        config.log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(config.log_path, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    return logger


def wait_until_stable(path: Path, *, checks: int, interval: float, timeout: float,
                      logger: logging.Logger | None = None) -> bool:
    """Wartet, bis Größe und Änderungszeit unverändert bleiben. Sonst lesen wir halbe Dateien."""
    deadline = time.monotonic() + timeout
    last: tuple[int, float] | None = None
    stable = 0
    while time.monotonic() < deadline:
        try:
            stat = path.stat()
        except FileNotFoundError:
            return False
        current = (stat.st_size, stat.st_mtime)
        if current == last:
            stable += 1
            if stable >= checks:
                return True
        else:
            stable = 0
            last = current
        time.sleep(interval)
    if logger:
        logger.warning("file did not settle within %.0fs: %s", timeout, path.name)
    return False


def _unique_destination(directory: Path, name: str) -> Path:
    """Nicht überschreiben: 'a.pdf' -> 'a (2).pdf'."""
    candidate = directory / name
    if not candidate.exists():
        return candidate
    stem, suffix = candidate.stem, candidate.suffix
    for counter in range(2, 1000):
        candidate = directory / f"{stem} ({counter}){suffix}"
        if not candidate.exists():
            return candidate
    raise RuntimeError(f"could not find a free filename for {name} in {directory}")


def _move(src: Path, dst: Path) -> Path:
    dst.parent.mkdir(parents=True, exist_ok=True)
    target = _unique_destination(dst.parent, dst.name)
    shutil.move(str(src), str(target))
    return target


def _human_bytes(count: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if count < 1024 or unit == "GB":
            return f"{count:.0f} {unit}" if unit == "B" else f"{count:.1f} {unit}"
        count /= 1024.0
    return f"{count:.1f} GB"


# --- the service --------------------------------------------------------------------------------

class InvoiceWatcherService:
    def __init__(self, config: ServiceConfig, *, logger: logging.Logger | None = None):
        self.config = config
        self.log = logger or setup_logging(config)
        self.master = build_and_load_master(config.master_path or MASTER_FILE)
        self.engine = RulesEngine(self.master)
        self.state = StateStore(config.state_path)
        self.client: OllamaClient | None = None
        self.rate_index: dict = {}
        self._stop = False
        self.totals = {"clean": 0, "faulty": 0, "skipped": 0, "error": 0}

        if config.extraction != "regex":
            self.client = OllamaClient(config.model, host=config.host, num_ctx=config.num_ctx,
                                       timeout=config.timeout)
            healthy, message = self.client.health()
            self.log.info("LLM extraction enabled: %s", message)
            if not healthy:
                self.log.warning("LLM unavailable (%s); falling back to the regex extractor", message)
                self.client = None

        # Rates for the LINE_RATE_OUTLIER rule. Learned from the master data folder's history if
        # present, otherwise from nothing — the rule simply stays quiet rather than guessing.
        self.rate_index = self._load_rate_index()

    # --- rate index -----------------------------------------------------------------------------
    def _load_rate_index(self) -> dict:
        """Durchschnittspreise aus den bereits akzeptierten Rechnungen.

        Bewusst nur aus sauberen Belegen: sonst würde eine falsch bepreiste Position den
        Maßstab mitbestimmen, an dem sie gemessen wird.
        """
        invoices: list[Invoice] = []
        try:
            rows = self.state.connection.execute(
                "SELECT path FROM processed WHERE verdict = 'clean' ORDER BY id DESC LIMIT 2000"
            ).fetchall()
        except Exception:
            return {}
        for row in rows:
            candidate = Path(row["path"])
            if not candidate.exists():
                continue
            try:
                text, _ = ingest_document(candidate, allow_ocr=True)
                if not text.strip():
                    continue
                outcome = extract_with_fallback(text, None, strategy="regex")
                invoices.append(outcome.invoice)
            except (IngestError, OSError, LLMError):
                continue
        if not invoices:
            return {}
        index = build_rate_index(invoices)
        self.log.info("rate index built from %d accepted invoices (%d services)",
                      len(invoices), len(index))
        return index

    # --- startup recovery -----------------------------------------------------------------------
    def recover_processing_folders(self) -> int:
        """Reste aus einem abgebrochenen Lauf zurück in den Eingangsordner legen."""
        recovered = 0
        for folder in self.config.watch:
            processing = folder.input_dir / PROCESSING_DIRNAME
            if not processing.exists():
                continue
            for leftover in sorted(processing.iterdir()):
                if leftover.is_dir():
                    continue
                target = _move(leftover, folder.input_dir / leftover.name)
                self.log.warning("recovered an interrupted document: %s", target.name)
                recovered += 1
        return recovered

    # --- main loop ------------------------------------------------------------------------------
    def request_stop(self, *_args) -> None:
        self._stop = True
        self.log.info("stop requested; finishing the current document")

    def install_signal_handlers(self) -> None:
        for name in ("SIGINT", "SIGTERM"):
            handler = getattr(signal, name, None)
            if handler is None:
                continue
            try:
                signal.signal(handler, self.request_stop)
            except (ValueError, OSError):
                # Not on the main thread (embedded use); the caller drives stop() instead.
                pass

    def run_forever(self, *, once: bool = False, max_cycles: int | None = None) -> dict:
        self.install_signal_handlers()
        self._announce_start()
        recovered = self.recover_processing_folders()
        if recovered:
            self.log.info("%d document(s) recovered from an interrupted run", recovered)

        cycles = 0
        idle_cycles = 0
        try:
            while not self._stop:
                processed = self.run_once()
                cycles += 1
                if processed == 0:
                    idle_cycles += 1
                else:
                    idle_cycles = 0
                if once or (max_cycles is not None and cycles >= max_cycles):
                    break
                if self._stop:
                    break
                if idle_cycles and idle_cycles % 30 == 0:
                    self.log.info("idle for %d cycles; %d document(s) checked so far",
                                  idle_cycles, sum(self.totals.values()))
                # In kleinen Schritten schlafen, damit ein Stop sofort greift.
                self._sleep(self.config.poll_seconds)
        finally:
            self._announce_stop()
            self.state.close()
        return self.totals

    def _sleep(self, seconds: float) -> None:
        deadline = time.monotonic() + seconds
        while not self._stop and time.monotonic() < deadline:
            time.sleep(min(0.5, max(0.0, deadline - time.monotonic())))

    def _announce_start(self) -> None:
        self.log.info("=" * 72)
        self.log.info("Rechnungsprüfung startet (Extraktion=%s)", self.config.extraction)
        for folder in self.config.watch:
            self.log.info("  Prüfordner: %s", folder.input_dir)
            self.log.info("    fehlerfrei -> %s", folder.approved_dir)
            self.log.info("    abgelehnt  -> %s", folder.failures_dir)
        self.log.info("  Zustandsdatei: %s", self.config.state_path)
        self.log.info("  Abfrageintervall: %ss", self.config.poll_seconds)

    def _announce_stop(self) -> None:
        self.log.info("Dienst gestoppt — freigegeben=%d abgelehnt=%d übersprungen=%d Fehler=%d",
                      self.totals["clean"], self.totals["faulty"],
                      self.totals["skipped"], self.totals["error"])

    # --- one pass over every watched folder -----------------------------------------------------
    def run_once(self) -> int:
        processed = 0
        for folder in self.config.watch:
            processed += self._scan_folder(folder)
        return processed

    def _scan_folder(self, folder: WatchFolder) -> int:
        folder.input_dir.mkdir(parents=True, exist_ok=True)
        processed = 0
        for candidate in self._pending_files(folder):
            if self._stop:
                break
            outcome = self._process_one(folder, candidate)
            if outcome is None:
                continue
            if outcome.verdict == "skipped":
                # Counted here rather than in _process_one, so the "documents checked" figure the
                # user sees means "newly checked" and the skip count means "already seen".
                self.totals["skipped"] = self.totals.get("skipped", 0) + 1
                continue
            processed += 1
        return processed

    def _pending_files(self, folder: WatchFolder) -> list[Path]:
        """Nur Dateien auf oberster Ebene. Unterordner werden nie durchsucht."""
        try:
            entries = sorted(folder.input_dir.iterdir())
        except OSError as exc:
            self.log.error("cannot read %s: %s", folder.input_dir, exc)
            return []
        pending = []
        for entry in entries:
            if entry.is_dir() or entry.name.startswith("."):
                continue
            if entry.suffix.lower() not in self.config.extensions:
                continue
            pending.append(entry)
        return pending

    # --- one document ---------------------------------------------------------------------------
    def _process_one(self, folder: WatchFolder, path: Path) -> DocumentOutcome | None:
        started = time.perf_counter()
        original_name = path.name
        try:
            sha = StateStore.hash_file(path)
        except OSError as exc:
            self.log.error("cannot read %s: %s", original_name, exc)
            return None

        previous = self.state.previous(path, sha)
        if previous is not None:
            self.log.debug("already processed, unchanged: %s (%s)", original_name, previous.verdict)
            return DocumentOutcome(path=path, original_name=original_name, verdict="skipped",
                                   detail=f"already processed as {previous.verdict} "
                                          f"on {previous.processed_at}")

        if not wait_until_stable(path, checks=self.config.stable_checks,
                                 interval=self.config.stable_interval,
                                 timeout=self.config.stable_timeout, logger=self.log):
            self.log.warning("skipping (still changing): %s", original_name)
            return None

        # Erst reservieren, dann lesen. Ein Absturz lässt die Datei dann in .processing
        # liegen statt halb verarbeitet im Eingang.
        processing_dir = folder.input_dir / PROCESSING_DIRNAME
        processing_dir.mkdir(parents=True, exist_ok=True)
        claimed = _unique_destination(processing_dir, original_name)
        try:
            shutil.move(str(path), str(claimed))
        except OSError as exc:
            self.log.error("cannot claim %s: %s", original_name, exc)
            return None

        self.log.info("checking %s", original_name)
        outcome = self._evaluate(folder, claimed, original_name)
        outcome.duration_ms = int((time.perf_counter() - started) * 1000)

        self._route(folder, outcome)

        self.state.record(
            path=path, sha256=sha, folder=folder.label, original_name=original_name,
            verdict=outcome.verdict, fault_codes=outcome.codes,
            severity=outcome.faults[0].severity if outcome.faults else "",
            destination=str(outcome.destination or ""),
            report_path=str(outcome.report_path or ""),
            duration_ms=outcome.duration_ms, detail=outcome.detail,
        )
        self.totals[outcome.verdict] = self.totals.get(outcome.verdict, 0) + 1
        return outcome

    def _evaluate(self, folder: WatchFolder, path: Path, original_name: str) -> DocumentOutcome:
        outcome = DocumentOutcome(path=path, original_name=original_name, verdict="clean")
        try:
            text, ingest_method = ingest_document(path, allow_ocr=True)
        except IngestError as exc:
            invoice = Invoice()
            text, ingest_method = "", "failed"
            outcome.detail = str(exc)

        if not text.strip():
            # Kein Rechnungsfehler, aber die Datei soll auch nicht ewig im Eingang liegen.
            invoice = Invoice()
            outcome.invoice = invoice
            outcome.extraction = ingest_method
            check = CheckResult(doc_id=original_name, extracted=False)
            check.faults.append(self.engine.fault(
                "EXTRACTION_FAILED",
                "No fields could be extracted from this document",
                method=ingest_method,
            ))
            outcome.check = check
            outcome.faults = check.faults
            outcome.verdict = "faulty" if folder.move_unreadable_to_failures else "error"
            outcome.detail = outcome.detail or f"no text recovered ({ingest_method})"
            return outcome

        strategy = self.config.resolve_extraction(folder)
        try:
            extracted = extract_with_fallback(text, self.client, strategy=strategy)
        except LLMError as exc:
            outcome.detail = f"extraction error: {exc}"
            extracted = extract_with_fallback(text, None, strategy="regex")

        invoice = extracted.invoice
        outcome.invoice = invoice
        outcome.extraction = f"{ingest_method}+{extracted.method}"

        faults = self.engine.check_invoice(invoice, rate_index=self.rate_index or None)
        check = CheckResult(doc_id=original_name, faults=faults, extracted=True,
                            llm_used=self.client is not None and "llm" in extracted.method)
        outcome.check = check
        outcome.faults = faults
        outcome.verdict = "faulty" if faults else "clean"
        if extracted.warnings:
            outcome.detail = "; ".join(extracted.warnings[:3])
        return outcome

    # --- routing --------------------------------------------------------------------------------
    def _route(self, folder: WatchFolder, outcome: DocumentOutcome) -> None:
        if self.config.dry_run:
            # Nur melden, Datei zurücklegen.
            target = folder.failures_dir if outcome.verdict == "faulty" else \
                (folder.approved_dir if outcome.verdict == "clean" else folder.failures_dir)
            outcome.destination = _move(outcome.path, folder.input_dir / outcome.original_name)
            self.log.info("  [dry run] %s → %s  (%s)", outcome.original_name, target.name,
                          ", ".join(outcome.codes) or "keine Feststellungen")
            return

        if outcome.verdict == "faulty":
            destination = self._route_to_failures(folder, outcome)
            outcome.destination = destination
            return

        if outcome.verdict == "clean":
            approved_dir = folder.approved_dir
            approved_dir.mkdir(parents=True, exist_ok=True)
            outcome.destination = _move(outcome.path, approved_dir / outcome.original_name)
            self.log.info("  OK → %s", outcome.destination.name)
            return

        # Fehler, die keine Rechnungsfehler sind, kommen in den Fehlerordner.
        destination_dir = folder.failures_dir if folder.move_unreadable_to_failures \
            else folder.input_dir
        outcome.destination = _move(outcome.path, destination_dir / outcome.original_name)
        self.log.warning("  unreadable → %s (%s)", outcome.destination.name, outcome.detail)

    def _route_to_failures(self, folder: WatchFolder, outcome: DocumentOutcome) -> Path:
        failures_dir = folder.failures_dir
        failures_dir.mkdir(parents=True, exist_ok=True)

        destination = _move(outcome.path, failures_dir / outcome.original_name)
        report_text = render_report(
            outcome.check, outcome.invoice, source_name=outcome.original_name,
            folder_label=folder.label, extraction=outcome.extraction,
            processing_seconds=outcome.duration_ms / 1000 if outcome.duration_ms else None,
        )
        report_path = _unique_destination(failures_dir, report_filename(outcome.original_name))
        report_path.write_text(report_text, encoding="utf-8")
        outcome.report_path = report_path

        codes = ", ".join(outcome.codes)
        self.log.warning("  ABGELEHNT → %s  [%s]", destination.name, codes)
        self.log.warning("              Bericht: %s", report_path.name)
        return destination
