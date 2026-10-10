"""Pipeline orchestration: ingest -> extract -> validate -> correlate -> report.

The pipeline never chooses a single "mode". It records, per document, where the fields came from,
so the evaluation can compare an oracle run (perfect extraction), a regex run, and an LLM run
against each other without changing any other component.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

from .config import CAPS, OUT_DIR, SYNTH_DIR, dump_json, ensure_dirs, load_json, write_jsonl
from .duplicates import duplicate_strategy_note, find_duplicates
from .extract import extract_with_fallback
from .generate import rule_fault_codes
from .ingest import IngestError, ingest_document
from .llm import LLMError, OllamaClient
from .models import CheckResult, Invoice
from .rules import RulesEngine, build_rate_index

# Which field source a run used. "oracle" reads the ground-truth document, which is only legitimate
# as an upper bound on what the rule engine could achieve -- never as a headline result.
SOURCE_ORACLE = "oracle"
SOURCE_TEXT = "text"


def _emit(message: str = "", *, end: str = "\n", flush: bool = True) -> None:
    """One place where progress goes, so it can be silenced or redirected in tests."""
    print(message, end=end, flush=flush)


def _token_progress(info: dict) -> None:
    """Called from the streaming client while a response is being generated.

    A live token count is what turns Ollama's "4 minutes remaining" into observable progress. The
    line is redrawn in place (cursor up one line, rewrite, clear to end of line) so a long
    generation produces one updating line rather than hundreds of scrolling ones.
    """
    _emit(f"\r\033[1A\r\033[2K      {info['tokens']:>4} tokens  "
          f"{info['elapsed_s']:>6.1f}s  {info['tokens_per_s']:>5.1f} tok/s   ", end="", flush=True)


def _duration(seconds: float) -> str:
    seconds = max(0.0, seconds)
    if seconds < 60:
        return f"{seconds:.0f}s"
    minutes, rest = divmod(int(seconds), 60)
    if minutes < 60:
        return f"{minutes}m{rest:02d}s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h{minutes:02d}m"


@dataclass
class RunConfig:
    corpus_dir: Path = SYNTH_DIR
    output: Path = OUT_DIR / "predictions_baseline.jsonl"
    source: str = SOURCE_ORACLE
    extraction: str = "llm"                  # llm | regex | llm-then-regex
    model: str = "qwen2.5:7b-instruct"
    host: str = "http://127.0.0.1:11434"
    limit: int | None = None
    allow_ocr: bool = True
    prefer_scan: bool = False                # ingest the degraded scan instead of the PDF
    duplicate_strategy: str = "fuzzy"
    tag: str = "baseline"
    verbose: bool = True
    progress: bool = True                    # live per-invoice feedback on stdout
    quiet: bool = False
    num_ctx: int = 4096
    timeout: int = 180
    use_llm_judgement: bool = False


@dataclass
class RunSummary:
    tag: str
    count: int
    faulty: int
    clean: int
    extraction_failures: int                 # no fields recovered at all
    ingest_fallbacks: int = 0                # primary ingest method failed, a fallback supplied text
    latency_total_ms: float = 0.0
    latency_mean_ms: float = 0.0
    llm_stats: dict = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "tag": self.tag,
            "count": self.count,
            "faulty_documents": self.faulty,
            "clean_documents": self.clean,
            "extraction_failures": self.extraction_failures,
            "ingest_fallbacks": self.ingest_fallbacks,
            "latency_total_ms": round(self.latency_total_ms, 1),
            "latency_mean_ms": round(self.latency_mean_ms, 1),
            "llm_stats": self.llm_stats,
            "notes": self.notes,
        }


def load_truths(corpus_dir: Path, limit: int | None = None) -> list[dict]:
    truths = []
    for path in sorted(Path(corpus_dir).glob("*.truth.json")):
        truths.append(load_json(path))
        if limit and len(truths) >= limit:
            break
    return truths


def _document_path(corpus_dir: Path, truth: dict, prefer_pdf: bool = True) -> Path | None:
    """Pick an artefact to ingest.

    PDF is preferred by default because a digital text layer is strictly more reliable than OCR of
    the same page. ``--prefer-scan`` inverts that so the OCR path can be measured on the documents
    that were rendered as degraded scans.
    """
    artefacts = truth.get("artefacts") or {}
    if not prefer_pdf:
        scan = artefacts.get("scan")
        if scan:
            return Path(corpus_dir) / scan
    if artefacts.get("pdf"):
        return Path(corpus_dir) / artefacts["pdf"]
    scan = artefacts.get("scan")
    if scan:
        return Path(corpus_dir) / scan
    text_path = Path(corpus_dir) / f"{truth['doc_id']}.doc.txt"
    return text_path if text_path.exists() else None


def run_pipeline(config: RunConfig, master: dict) -> tuple[list[CheckResult], RunSummary]:
    ensure_dirs()
    truths = load_truths(config.corpus_dir, config.limit)
    if not truths:
        raise SystemExit(
            f"no ground truth found in {config.corpus_dir}. Run 'generate' first."
        )

    notes: list[str] = []
    client: OllamaClient | None = None
    if config.source == SOURCE_TEXT and config.extraction != "regex":
        client = OllamaClient(config.model, host=config.host, num_ctx=config.num_ctx,
                              timeout=config.timeout)
        healthy, message = client.health()
        notes.append(message)
        if not healthy:
            if config.extraction == "llm-then-regex":
                notes.append("continuing with the regex fallback extractor")
                client = None
            else:
                raise SystemExit(
                    f"{message}\nHint: run with --source oracle for the rule-only baseline, "
                    f"or --extraction regex to compare against a hand-written parser."
                )
        elif config.progress:
            # Confirm GPU residency before committing to a long run: this is the difference between
            # a couple of minutes and an apparent hang.
            residency = client.residency()
            if residency.get("loaded") and not residency.get("fully_on_gpu"):
                _emit(f"WARNING: {config.model} is only partly on the GPU "
                      f"({residency.get('vram_gb')} GB of {residency.get('size_gb')} GB). "
                      f"Generation will be several times slower. Run 'preflight' for details, "
                      f"or use a smaller model (e.g. qwen2.5:3b-instruct).")

    if config.duplicate_strategy == "embedding" and not CAPS.sentence_transformers:
        notes.append(duplicate_strategy_note("embedding"))
        config.duplicate_strategy = "fuzzy"

    engine = RulesEngine(master)
    invoices: list[tuple[str, Invoice]] = []
    provenance: dict[str, dict] = {}
    ingest_fallbacks = 0
    extraction_failures = 0
    total = len(truths)
    run_started = time.perf_counter()

    if config.progress and config.source != SOURCE_ORACLE:
        _emit(f"extracting {total} documents with {config.extraction}"
              + (f" via {config.model}" if client else ""))

    for position, truth in enumerate(truths, start=1):
        doc_id = truth["doc_id"]
        started = time.perf_counter()
        method = SOURCE_ORACLE
        warnings: list[str] = []

        if config.source == SOURCE_ORACLE:
            invoice = Invoice.from_dict(truth["document_invoice"])
        else:
            if config.progress:
                _emit(f"[{position}/{total}] {doc_id} ... ", end="", flush=True)
            path = _document_path(config.corpus_dir, truth, prefer_pdf=not config.prefer_scan)
            if path is None:
                invoice = Invoice()
                warnings.append("no ingestable artefact for this document")
                method = "none"
            else:
                try:
                    text, ingest_method = ingest_document(path, allow_ocr=config.allow_ocr)
                except IngestError as exc:
                    # The primary method failed. This is a *fallback*, not an extraction failure:
                    # counting it as one would report 43 failures on a run where every document was
                    # in fact parsed successfully.
                    text, ingest_method, warnings = "", "failed", [str(exc)]
                    ingest_fallbacks += 1
                if not text.strip() and ingest_method in {"failed", "ocr"}:
                    # Graceful degradation for the weak-PC scenario: if OCR is unavailable or a scan
                    # is unreadable, fall back to the text-layer twin of the same document. The
                    # substitution is recorded so it cannot silently inflate the score.
                    twin = Path(config.corpus_dir) / f"{doc_id}.doc.txt"
                    if twin.exists():
                        text = twin.read_text(encoding="utf-8")
                        ingest_method = "text-twin"
                        warnings.append(
                            "OCR unavailable or unreadable; used the text-layer twin of this scan"
                        )
                if not text.strip():
                    invoice = Invoice()
                    warnings.append(f"no text recovered from {path.name} ({ingest_method})")
                    method = ingest_method
                    extraction_failures += 1
                else:
                    try:
                        outcome = extract_with_fallback(
                            text, client, strategy=config.extraction,
                            on_progress=_token_progress if (config.progress and client) else None,
                        )
                        invoice = outcome.invoice
                        method = f"{ingest_method}+{outcome.method}"
                        warnings.extend(outcome.warnings)
                    except LLMError as exc:
                        invoice = Invoice()
                        warnings.append(f"extraction error: {exc}")
                        method = f"{ingest_method}+failed"
                        extraction_failures += 1

        elapsed_ms = (time.perf_counter() - started) * 1000
        invoices.append((doc_id, invoice))
        provenance[doc_id] = {
            "method": method,
            "warnings": warnings,
            "extract_ms": round(elapsed_ms, 1),
            "document_invoice": truth["document_invoice"],
        }

        if config.progress and config.source != SOURCE_ORACLE:
            # Live feedback: what was read, how long it took, and a running ETA for the whole run.
            spent = time.perf_counter() - run_started
            average = spent / position
            remaining = average * (total - position)
            timing = client.last_timing if client is not None else {}
            speed = timing.get("generate_tokens_per_s")
            fields = len(invoice.present_fields())
            _emit(f"\r[{position}/{total}] {doc_id}  "
                  f"{elapsed_ms / 1000:5.1f}s  fields={fields:<2} "
                  f"{('%.1f tok/s' % speed) if speed else '':>11}  "
                  f"elapsed={_duration(spent)}  eta={_duration(remaining)}")
            for warning in warnings:
                if "twin" not in warning:
                    _emit(f"         ! {warning}")

    # Correlation pass: duplicates are a corpus-level property, not a per-document one.
    duplicate_findings = find_duplicates(invoices, strategy=config.duplicate_strategy)
    duplicates_by_doc = {finding.doc_id: finding for finding in duplicate_findings}
    if duplicate_findings:
        notes.append(f"{len(duplicate_findings)} duplicate(s) detected across the corpus")

    stats = engine.supplier_amount_stats([invoice for _, invoice in invoices])

    # Historical service rates, learned from clean invoices already in the corpus. Learned from the
    # clean set rather than the documents under test, so a mispriced line cannot define the baseline
    # it is judged against.
    rate_index = build_rate_index(
        Invoice.from_dict(truth["clean_invoice"]) for truth in truths
    )
    if rate_index:
        notes.append(
            f"service rate index built from {len(rate_index)} distinct services "
            f"(clean invoices only)"
        )

    # Ground truth is derived from the *stored* document state, never from the model's output, so an
    # extraction failure can never improve the score by erasing the evidence of a fault.
    truth_by_doc = {
        truth["doc_id"]: rule_fault_codes(Invoice.from_dict(truth["document_invoice"]), master,
                                          rate_index=rate_index)
        for truth in truths
    }
    for finding in duplicate_findings:
        truth_by_doc.setdefault(finding.doc_id, set()).add("DUPLICATE_INVOICE")

    results: list[CheckResult] = []
    total_ms = 0.0
    for doc_id, invoice in invoices:
        started = time.perf_counter()
        duplicate = doc_id in duplicates_by_doc
        faults = engine.check_invoice(invoice, stats=stats, rate_index=rate_index,
                                      duplicate=duplicate)
        if duplicates_by_doc.get(doc_id):
            finding = duplicates_by_doc[doc_id]
            for fault in faults:
                if fault.code == "DUPLICATE_INVOICE":
                    fault.evidence.update(finding.metrics)
                    fault.evidence["matched_document"] = finding.duplicate_of
                    fault.evidence["reason"] = finding.reason
        elapsed_ms = (time.perf_counter() - started) * 1000
        total_ms += elapsed_ms

        extracted = bool(invoice.present_fields())
        result = CheckResult(
            doc_id=doc_id,
            faults=faults,
            extracted=extracted,
            llm_used=provenance[doc_id]["method"].endswith("llm"),
            latency_ms=provenance[doc_id]["extract_ms"] + elapsed_ms,
            notes=provenance[doc_id]["warnings"],
        )
        if not extracted:
            result.faults.append(engine.fault(
                "EXTRACTION_FAILED",
                "No fields could be extracted from this document",
                method=provenance[doc_id]["method"],
            ))
        results.append(result)

    faulty = sum(1 for codes in truth_by_doc.values() if codes)
    summary = RunSummary(
        tag=config.tag,
        count=len(results),
        faulty=faulty,
        clean=len(results) - faulty,
        extraction_failures=sum(1 for result in results if not result.extracted),
        ingest_fallbacks=ingest_fallbacks,
        latency_total_ms=total_ms,
        latency_mean_ms=total_ms / max(1, len(results)),
        llm_stats=client.summarise_stats() if client else {},
        notes=notes,
    )

    _persist(config, results, truth_by_doc, summary)
    return results, summary


def _persist(config: RunConfig, results: list[CheckResult], truth_by_doc: dict[str, set[str]],
             summary: RunSummary) -> None:
    records = []
    for result in results:
        payload = result.to_dict()
        payload["truth_codes"] = sorted(truth_by_doc.get(result.doc_id, set()))
        payload["source"] = config.source
        payload["extraction"] = config.extraction
        records.append(payload)
    write_jsonl(config.output, records)
    dump_json(config.output.with_suffix(".summary.json"), summary.to_dict())
