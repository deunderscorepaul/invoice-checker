"""Command line interface for the invoice checker."""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from . import __version__
from .config import (CAPS, DEFAULT_LLM_MODEL, DEFAULT_OLLAMA_URL, DEFAULT_SEED, MASTER_FILE,
                     OUT_DIR, SYNTH_DIR, ensure_dirs, dump_json, load_json)
from .generate import build_master_data, generate_corpus, write_corpus
from .models import Invoice
from .pipeline import RunConfig, SOURCE_ORACLE, SOURCE_TEXT, load_truths, run_pipeline
from .rules import RulesEngine

EXIT_OK = 0
EXIT_FAILURE = 1


def _load_master(seed: int = DEFAULT_SEED) -> dict:
    if MASTER_FILE.exists():
        return load_json(MASTER_FILE)
    master = build_master_data(seed)
    dump_json(MASTER_FILE, master)
    return master


# --- commands -----------------------------------------------------------------------------------

def cmd_lint(_args) -> int:
    print(f"invoice-checker {__version__}")
    print(f"python: {sys.version.split()[0]}")
    print(CAPS.summary())
    print()
    print("importable check (informational):")
    for module in ("zlib", "decimal", "difflib", "urllib.request", "PIL", "pytest"):
        from .config import has_module

        print(f"  {'OK   ' if has_module(module) else 'MISS '} {module}")
    return EXIT_OK


def cmd_generate(args) -> int:
    ensure_dirs()
    master = build_master_data(args.seed)
    dump_json(MASTER_FILE, master)
    print(f"master data: {len(master['suppliers'])} suppliers -> {MASTER_FILE}")

    started = time.perf_counter()
    documents = generate_corpus(args.n, seed=args.seed, fault_rate=args.fault_rate,
                                include_duplicates=not args.no_duplicates, master=master)
    corpus_dir = Path(args.out) if args.out else SYNTH_DIR
    manifest = write_corpus(documents, corpus_dir, render_pdf=not args.no_pdf,
                            scan_ratio=args.scans)
    elapsed = time.perf_counter() - started

    print(f"generated {manifest['count']} documents in {elapsed:.2f}s -> {corpus_dir}")
    print(f"  faulty documents : {manifest['faulty']} "
          f"({manifest['faulty'] / max(1, manifest['count']):.1%})")
    print(f"  PDFs written     : {manifest['pdfs']}")
    print(f"  scan images      : {manifest['scans']}")
    if args.scans and not CAPS.pytesseract:
        print("  note: pytesseract is not installed, so the scan images cannot be OCR'd yet")

    from collections import Counter

    counter: Counter[str] = Counter()
    for document in documents:
        counter.update(document.truth_codes)
    print("  fault distribution:")
    for code, count in counter.most_common():
        print(f"    {code:<40}{count:>5}")
    return EXIT_OK


def cmd_run(args) -> int:
    master = _load_master()
    if args.source == "auto":
        source = SOURCE_ORACLE if args.baseline else SOURCE_TEXT
    else:
        source = SOURCE_ORACLE if args.source == "oracle" else SOURCE_TEXT
    extraction = "regex" if args.baseline else args.extraction

    tag = args.tag or ("baseline" if args.baseline else f"llm-{extraction}")
    output = Path(args.out) if args.out else OUT_DIR / f"predictions_{tag}.jsonl"

    config = RunConfig(
        corpus_dir=Path(args.corpus) if args.corpus else SYNTH_DIR,
        output=output,
        source=source,
        extraction=extraction,
        model=args.model,
        host=args.host,
        limit=args.limit,
        allow_ocr=not args.no_ocr,
        prefer_scan=args.prefer_scan,
        duplicate_strategy=args.duplicates,
        tag=tag,
        progress=not args.quiet,
        quiet=args.quiet,
        num_ctx=args.num_ctx,
        timeout=args.timeout,
    )
    print(f"run tag={tag} source={source} extraction={extraction}"
          + (f" model={args.model}" if source == SOURCE_TEXT and extraction != "regex" else "")
          + (" [preferring scan images]" if args.prefer_scan else ""))
    if source == SOURCE_TEXT and extraction != "regex":
        print(f"num_ctx={args.num_ctx} timeout={args.timeout}s "
              f"(raise --timeout if generation is slow; lower --num-ctx to cut VRAM)")
    results, summary = run_pipeline(config, master)

    print(f"documents processed : {summary.count} ({summary.faulty} faulty / {summary.clean} clean)")
    print(f"faults found        : {sum(len(r.faults) for r in results)}")
    print(f"extraction failures : {summary.extraction_failures}")
    if summary.ingest_fallbacks:
        print(f"ingest fallbacks    : {summary.ingest_fallbacks} "
              f"(primary method failed, fallback supplied the text)")
    print(f"latency             : mean {summary.latency_mean_ms:.1f} ms/invoice")
    if summary.llm_stats:
        stats = summary.llm_stats
        print(f"llm usage           : {stats['calls']} calls, "
              f"{stats['prompt_tokens']} prompt + {stats['completion_tokens']} completion tokens, "
              f"avg {stats['avg_ms']} ms/call")
    for note in summary.notes:
        print(f"note: {note}")
    print(f"predictions written : {output}")
    return EXIT_OK


def cmd_eval(args) -> int:
    from .evaluate import compare_reports, evaluate, load_report, render_console, save

    if args.all:
        reports = []
        for path in sorted(OUT_DIR.glob("predictions_*.jsonl")):
            report = evaluate(path)
            save(report)
            reports.append(report)
            print(render_console(report))
            print()
        if len(reports) > 1:
            print(compare_reports(reports))
        return EXIT_OK

    path = Path(args.pred) if args.pred else OUT_DIR / "predictions_baseline.jsonl"
    if not path.exists():
        print(f"no predictions at {path}. Run 'run --baseline' first.", file=sys.stderr)
        return EXIT_FAILURE
    report = evaluate(path, tag=args.tag)
    print(render_console(report))
    saved = save(report)
    print(f"\nreport written: {saved}")
    if args.csv:
        from .report import write_code_table_csv

        print(f"code table written: {write_code_table_csv(report, Path(args.csv))}")
    return EXIT_OK


def cmd_bench(args) -> int:
    """Time the pipeline and, when possible, report peak GPU memory.

    The VRAM figure is what makes the 'runs on consumer hardware' claim measurable rather than
    aspirational. It is read from nvidia-smi, so it works whether or not torch is installed.
    """
    import subprocess

    from .evaluate import evaluate

    master = _load_master()
    truths = load_truths(SYNTH_DIR, args.limit)
    if not truths:
        print("no corpus; run 'generate' first", file=sys.stderr)
        return EXIT_FAILURE

    engine = RulesEngine(master)
    invoices = [(truth["doc_id"], Invoice.from_dict(truth["document_invoice"])) for truth in truths]

    started = time.perf_counter()
    for doc_id, invoice in invoices:
        engine.check_invoice(invoice)
    rule_ms = (time.perf_counter() - started) * 1000

    started = time.perf_counter()
    from .ingest import extract_text_from_pdf

    text_chars = 0
    for truth in truths:
        pdf = (truth.get("artefacts") or {}).get("pdf")
        if pdf:
            text_chars += len(extract_text_from_pdf(SYNTH_DIR / pdf))
    ingest_ms = (time.perf_counter() - started) * 1000

    count = len(invoices)
    print(f"documents            : {count}")
    print(f"rule engine          : {rule_ms:.1f} ms total, {rule_ms / count:.2f} ms/invoice")
    print(f"pdf text extraction  : {ingest_ms:.1f} ms total, {ingest_ms / count:.2f} ms/invoice")
    print(f"extracted characters : {text_chars} ({text_chars / count:.0f} per document)")

    try:
        output = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,memory.used",
             "--format=csv,noheader"],
            capture_output=True, text=True, timeout=15, check=False,
        )
        if output.returncode == 0 and output.stdout.strip():
            print("gpu                  : " + output.stdout.strip().replace("\n", " | "))
    except (FileNotFoundError, subprocess.SubprocessError):
        print("gpu                  : nvidia-smi unavailable")

    if args.predictions:
        report = evaluate(Path(args.predictions), tag="bench")
        print(f"micro F1             : {report.micro['f1']:.3f}")
    return EXIT_OK


def cmd_preflight(args) -> int:
    """Prove where the model runs before committing to a long extraction run."""
    from .llm import OllamaClient
    from .preflight import format_report, probe

    client = OllamaClient(args.model, host=args.host, num_ctx=args.num_ctx, timeout=args.timeout)
    healthy, message = client.health()
    if not healthy:
        print(message, file=sys.stderr)
        return EXIT_FAILURE

    print(f"probing {args.model} with a single tiny generation ...")
    result = probe(client)
    print(format_report(args.model, client, result))

    speed = result.get("tokens_per_s") or 0
    if speed:
        estimate = 200 / speed
        print()
        print(f"projected cost    : ~{estimate:.1f}s/invoice at ~200 output tokens, "
              f"so ~{estimate * 300 / 60:.0f} minutes for a 300-invoice run")
        if not (result.get("residency") or {}).get("fully_on_gpu", True):
            print("                    (this estimate will only hold if the model fits in VRAM)")
    return EXIT_OK if result.get("ok") else EXIT_FAILURE


def cmd_dashboard(args) -> int:
    """Render the results dashboard to a single self-contained HTML file."""
    from .dashboard import build_dashboard

    out_dir = Path(args.out_dir) if args.out_dir else OUT_DIR
    target = Path(args.output) if args.output else out_dir / "dashboard.html"
    if args.text:
        from .dashboard import terminal_summary

        print(terminal_summary(out_dir))
        return EXIT_OK

    _, written = build_dashboard(out_dir, Path(args.corpus) if args.corpus else SYNTH_DIR)
    if args.output:
        target.write_text(written.read_text(encoding="utf-8"), encoding="utf-8")
        written = target
    size_kb = written.stat().st_size / 1024
    print(f"dashboard written: {written}  ({size_kb:.0f} KB, self-contained HTML)")
    print("open it in any browser; no server and no network access required")
    if args.open:
        import webbrowser

        webbrowser.open(written.resolve().as_uri())
        print("opened in your default browser")
    return EXIT_OK


def cmd_summary(args) -> int:
    """Compact text version of the dashboard for a terminal."""
    from .dashboard import terminal_summary

    out_dir = Path(args.out_dir) if args.out_dir else OUT_DIR
    print(terminal_summary(out_dir))
    return EXIT_OK


def cmd_service(args) -> int:
    """Prüfordner überwachen und Rechnungen einsortieren."""
    import json

    from .service import InvoiceWatcherService, enable_utf8_console
    from .service_config import load_config, sample_config

    enable_utf8_console()

    if args.init:
        target = Path(args.init)
        if target.exists() and not args.force:
            print(f"Konfiguration existiert schon: {target} (--force zum Überschreiben)",
                  file=sys.stderr)
            return EXIT_FAILURE
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(sample_config(target.parent), indent=2, ensure_ascii=False),
                          encoding="utf-8")
        print(f"Vorlage geschrieben nach {target}")
        print(f"Prüfordner anpassen, dann:  service --config {target}")
        return EXIT_OK

    if not args.config:
        print("--config fehlt (oder --init zum Anlegen)", file=sys.stderr)
        return EXIT_FAILURE

    config = load_config(Path(args.config))
    if args.once:
        config.poll_seconds = 1
    if args.dry_run:
        config.dry_run = True

    service = InvoiceWatcherService(config)
    try:
        totals = service.run_forever(once=args.once)
    except KeyboardInterrupt:
        print("\nabgebrochen")
        return 130

    print()
    print(f"geprüft in diesem Lauf      : {sum(totals.values())}")
    print(f"  freigegeben (approved)    : {totals.get('clean', 0)}")
    print(f"  abgelehnt (fehlerhaft)    : {totals.get('faulty', 0)}")
    print(f"  unlesbar / Fehler         : {totals.get('error', 0)}")
    print(f"  übersprungen (schon geprüft): {totals.get('skipped', 0)}")
    return EXIT_OK


def cmd_selftest(_args) -> int:
    from .selftest import run_selftest

    results = run_selftest()
    failures = [name for name, ok, _ in results if not ok]
    if failures:
        print(f"\n{len(failures)} of {len(results)} check(s) FAILED:", file=sys.stderr)
        for name in failures:
            print(f"  - {name}", file=sys.stderr)
        return EXIT_FAILURE
    print(f"\nall {len(results)} checks passed")
    return EXIT_OK


# --- argument parsing ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="invoice_checker",
        description="Hybrid local-AI checker for outgoing invoices.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("lint", help="report optional dependency availability").set_defaults(func=cmd_lint)
    sub.add_parser("selftest", help="run built-in checks on checksums and rules").set_defaults(
        func=cmd_selftest)

    pre = sub.add_parser("preflight",
                         help="check the model fits in VRAM and measure its speed before a long run")
    pre.add_argument("--model", default=DEFAULT_LLM_MODEL)
    pre.add_argument("--host", default=DEFAULT_OLLAMA_URL)
    pre.add_argument("--num-ctx", type=int, default=4096, dest="num_ctx")
    pre.add_argument("--timeout", type=int, default=300)
    pre.set_defaults(func=cmd_preflight)

    gen = sub.add_parser("generate", help="build the labelled synthetic corpus")
    gen.add_argument("-n", "--n", type=int, default=200, help="number of invoices (default 200)")
    gen.add_argument("--fault-rate", type=float, default=0.6, help="share of faulty documents")
    gen.add_argument("--seed", type=int, default=DEFAULT_SEED)
    gen.add_argument("--scans", type=float, default=0.0,
                     help="share of documents also rendered as scan images, e.g. 0.2")
    gen.add_argument("--no-pdf", action="store_true", help="skip PDF rendering")
    gen.add_argument("--no-duplicates", action="store_true", help="disable duplicate injection")
    gen.add_argument("--out", default=None, help="output directory (default data/synth)")
    gen.set_defaults(func=cmd_generate)

    run = sub.add_parser("run", help="run the checker over the corpus")
    run.add_argument("--baseline", action="store_true",
                     help="rules only: oracle fields, no model (the baseline in the report)")
    run.add_argument("--source", choices=["auto", "oracle", "text"], default="auto",
                     help="oracle reads ground-truth fields; text ingests real documents")
    run.add_argument("--extraction", choices=["llm", "regex", "llm-then-regex"], default="llm")
    run.add_argument("--model", default=DEFAULT_LLM_MODEL)
    run.add_argument("--host", default=DEFAULT_OLLAMA_URL)
    run.add_argument("--corpus", default=None)
    run.add_argument("--out", default=None)
    run.add_argument("--tag", default=None)
    run.add_argument("--limit", type=int, default=None)
    run.add_argument("--no-ocr", action="store_true")
    run.add_argument("--prefer-scan", action="store_true",
                     help="ingest the degraded scan image instead of the PDF text layer (OCR path)")
    run.add_argument("--quiet", action="store_true", help="suppress per-invoice progress output")
    run.add_argument("--num-ctx", type=int, default=4096, dest="num_ctx",
                     help="model context window; lower uses less VRAM (default 4096)")
    run.add_argument("--timeout", type=int, default=180,
                     help="per-request timeout in seconds (default 180)")
    run.add_argument("--duplicates", choices=["exact", "fuzzy", "embedding"], default="fuzzy")
    run.set_defaults(func=cmd_run)

    ev = sub.add_parser("eval", help="score predictions against ground truth")
    ev.add_argument("--pred", default=None, help="predictions jsonl (default baseline)")
    ev.add_argument("--tag", default=None)
    ev.add_argument("--all", action="store_true", help="evaluate every predictions file found")
    ev.add_argument("--csv", default=None, help="also write the per-code table to this CSV file")
    ev.set_defaults(func=cmd_eval)

    bench = sub.add_parser("bench", help="time the rule engine and report GPU memory")
    bench.add_argument("--limit", type=int, default=None)
    bench.add_argument("--predictions", default=None)
    bench.set_defaults(func=cmd_bench)

    dash = sub.add_parser("dashboard",
                          help="build a self-contained HTML dashboard of all scored runs")
    dash.add_argument("--out-dir", default=None, dest="out_dir",
                      help="directory holding predictions_*.jsonl (default data/out)")
    dash.add_argument("--output", default=None, help="write the HTML to this path")
    dash.add_argument("--corpus", default=None, help="corpus directory holding manifest.json")
    dash.add_argument("--open", action="store_true", help="open the result in a browser")
    dash.add_argument("--text", action="store_true",
                      help="print a compact text summary instead of writing HTML")
    dash.set_defaults(func=cmd_dashboard)

    summ = sub.add_parser("summary", help="print a compact text summary of all scored runs")
    summ.add_argument("--out-dir", default=None, dest="out_dir")
    summ.set_defaults(func=cmd_summary)

    svc = sub.add_parser("service",
                         help="Prüfordner überwachen und Rechnungen nach approved/ oder "
                              "fehlerhaft/ verschieben (mit deutschem Bericht)")
    svc.add_argument("--config", default=None, help="Pfad zur JSON-Konfiguration")
    svc.add_argument("--init", default=None, metavar="PATH",
                     help="Vorlage nach PATH schreiben und beenden")
    svc.add_argument("--once", action="store_true",
                     help="anstehende Dateien einmal abarbeiten, dann beenden")
    svc.add_argument("--dry-run", action="store_true",
                     help="nur prüfen und melden, keine Datei verschieben")
    svc.add_argument("--force", action="store_true", help="vorhandene Konfiguration überschreiben")
    svc.set_defaults(func=cmd_service)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except SystemExit:
        raise
    except KeyboardInterrupt:
        print("\nabgebrochen", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
