# Ausgangsrechnungsprüfung

Hybride, komplett lokale Prüfung von **Ausgangsrechnungen** auf Consumer-Hardware
(entwickelt auf 16 GB RAM / RTX 3060 Ti 8 GB).

Grundgedanke: **das kleine lokale LLM wird nur dort eingesetzt, wo Sprache verstanden werden muss
(unstrukturierte Extraktion und Erklärung). Alle rechtlichen, rechnerischen und formalen Prüfungen
macht deterministischer Code.** Ein LLM darf niemals eine Spalte Zahlen addieren.

Im Betrieb als Ordnerdienst: Rechnungen in den Prüfordner legen, der Dienst sortiert sie
automatisch nach `approved/` oder `fehlerhaft/` — abgelehnte mit deutschem Fehlerbericht.

```
PDF / Scan / Text
      │
      ├─ 1. ingest      Textebene im PDF  oder  OCR-Fallback       (ingest.py)
      ├─ 2. extract     lokales LLM -> JSON nach Schema            (llm.py, extract.py)
      ├─ 3. validate    deterministische Regelprüfung              (rules.py)
      ├─ 4. correlate   Dubletten / Ausreißer                      (duplicates.py)
      └─ 5. report      Feststellungen, Schweregrad, Rechtsbezug   (service_report.py, report.py)
```

## Quick start (bundled runtime, no install)

```powershell
# from the invoice-checker directory
$py = "C:\Users\JUUL\.dsh\dsh-runtimes\dsh-primary-runtime\dependencies\python\python.exe"

& $py -m invoice_checker selftest          # verify checksums + rules work
& $py -m invoice_checker generate --n 200  # build a labelled synthetic corpus
& $py -m invoice_checker run --baseline    # rules-only run (no LLM needed)
& $py -m invoice_checker eval              # per-fault-type precision / recall / F1
& $py -m invoice_checker lint              # which optional deps are present
```

Add the LLM path (requires [Ollama](https://ollama.com) running locally):

```powershell
ollama pull qwen2.5:7b-instruct            # or qwen2.5:3b-instruct on weaker hardware
& $py -m invoice_checker preflight         # FIRST: is the model on the GPU, and how fast?
& $py -m invoice_checker run --limit 10    # small run first; progress + ETA are printed
& $py -m invoice_checker eval --pred data/out/predictions_llm.jsonl --tag llm
```

**If a run seems to hang**, stop it and run `preflight`. A model that does not fit in VRAM is partly
executed on the CPU and becomes many times slower, which Ollama reports only as a completion estimate
that never moves. `preflight` says whether the model is fully resident in VRAM and what its real
token rate is. Fixes, in order: `--model qwen2.5:3b-instruct`, lower `--num-ctx`, close other GPU apps.

## Why the LLM is the small part

| Fault class | Decided by | LLM needed? |
|---|---|---|
| net + VAT ≠ gross, line items ≠ net | Decimal arithmetic | **No** |
| **Work billed at zero: "10 h of labour" with a 0 EUR price or amount** | line arithmetic | **No** |
| **Line net ≠ quantity × unit price** | line arithmetic | **No** |
| **Rate far below the historical median for that service** | indexed rate table | **No** |
| Umsatzsteuer-ID check digit (DE) | ISO 7064-style checksum | **No** |
| IBAN validity | MOD-97 | **No** |
| Mandatory fields present | schema check | **No** |
| Future-dated, delivery > invoice date, due < invoice | date logic | **No** |
| §14 UStG payment-term / reverse-charge note issues | rule table | **No** |
| Unknown supplier, currency mismatch, negative amounts | master data | **No** |
| Duplicate invoice (near, not exact) | fuzzy / embedding similarity | borderline only |
| **Reading fields and table rows out of a messy document** | small local LLM | **Yes** |
| **Explaining a rejection in plain language** | small local LLM | **Yes** |

That is the whole argument of the project: measure the rules-only baseline against the LLM-backed
baseline and show where each one wins. 35 of the 37 fault codes are decided without a model.

## Commands

| Command | Purpose |
|---|---|
| `selftest` | 58 built-in checks: checksums, arithmetic, dates, line pricing, schema, German reports, state |
| `preflight [--model NAME]` | Check the model is on the GPU and measure its token rate before a long run |
| `generate --n N [--fault-rate 0.6] [--scans 0.2]` | Synthetic corpus + ground truth + optional scan-style images |
| `run [--baseline] [--limit N] [--model NAME] [--quiet] [--num-ctx N]` | Run the pipeline with live progress and ETA |
| `eval [--pred FILE] [--tag NAME] [--all] [--csv FILE]` | Per-fault precision/recall/F1 + latency, optional CSV export |
| `lint` | Report which optional dependencies are importable |
| `bench` | Time the rule engine over the corpus, and report GPU memory |
| `dashboard [--open] [--text] [--out-dir DIR]` | Build a self-contained HTML results dashboard |
| `summary` | Print a compact text version of the dashboard |
| `service --init PATH` | Write a starting service config |
| `service --config FILE [--once] [--dry-run]` | Watch a folder and file faulty invoices with a German report |

## Rechnungsprüfung als Dienst

```powershell
& $py -m invoice_checker service --init service.json    # Vorlage anlegen
& $py -m invoice_checker service --config service.json  # läuft dauerhaft
& $py -m invoice_checker service --config service.json --once      # ein Durchlauf
& $py -m invoice_checker service --config service.json --dry-run   # nur melden, nichts verschieben
```

`watch[].input` auf den Prüfordner zeigen lassen. Rechnungen, die dort abgelegt werden,
werden automatisch einsortiert:

```
rechnungen/pruefordner/            <- Rechnungen hier ablegen
├── approved/                      <- geprüft und in Ordnung
├── fehlerhaft/                    <- abgelehnt, jeweils mit .fehlerbericht.txt
└── .processing/                   <- intern, nicht anfassen
```

Der Prüfordner ist nach dem Lauf wieder leer; jede Rechnung liegt danach in genau einem der
beiden Zielordner.

```json
{
  "watch": [
    { "name": "pruefordner", "input": "rechnungen/pruefordner",
      "approved": "approved", "failures": "fehlerhaft" }
  ],
  "poll_seconds": 10,
  "extraction": "regex",
  "model": "qwen2.5:7b-instruct",
  "state_path": "service_state.sqlite3",
  "log_path": "service.log"
}
```

### Was der Dienst garantiert

| Garantie | Umsetzung |
|---|---|
| Keine Rechnung geht verloren | Datei wird **vor** dem Lesen in `.processing/` verschoben; Reste werden beim Start automatisch zurückgeholt |
| Keine Rechnung wird doppelt geprüft | SQLite-Schlüssel **(Pfad, sha256)**; eine korrigierte und erneut abgelegte Rechnung wird aber neu geprüft |
| Keine halb geschriebene Datei wird gelesen | Größe und Änderungszeit müssen über zwei Abfragen gleich bleiben |
| Die Originaldatei bleibt unverändert | Es wird nur verschoben, der Inhalt ist danach byte-identisch |
| Nichts wird stillschweigend übersprungen | Unlesbare Belege landen ebenfalls in `fehlerhaft/`, mit entsprechendem Bericht |

`extraction` steht auf `regex`: unter einer Sekunde pro Rechnung, ohne Abhängigkeiten. Mit
`llm-then-regex` nutzt der Dienst das lokale Modell — robuster bei ungewöhnlichen Layouts, aber
~6 s pro Rechnung, also vorher `preflight` laufen lassen.

### Der deutsche Fehlerbericht

Eine Textdatei pro abgelehnter Rechnung, direkt neben der Rechnung. Aufbau: Ergebnis und
Schweregrad, dann jede Feststellung mit den beanstandeten Werten, dann was zu tun ist, dann die
Rechtsgrundlage.

```
ERGEBNIS: FEHLERHAFT – 3 Feststellung(en)
  Schweregrad: Kritisch
  Die Rechnung ist in dieser Form nicht buchungsfähig und darf nicht versendet werden.

  2. LINE_AMOUNT_ZERO  (Kritisch)
  Position 1 „Material und Verbrauchsmaterial“: Es werden 4 Einheiten berechnet,
  der Positionsbetrag ist jedoch 0,00. Eine erbrachte Leistung darf nicht mit 0,00
  berechnet werden. Bitte den Einzelpreis oder den Positionsbetrag ergänzen.

  Festgestellte Werte:
  Menge:                        4
  Einzelpreis:                  29,48
  Positionsbetrag:              0,00
```

Alle 37 Fehlercodes haben einen handgeschriebenen deutschen Text; `selftest` prüft das, ein neuer
Code ohne Erklärung fällt also sofort auf.

### Dauerbetrieb

Bewusst eine Polling-Schleife und keine Dateisystem-Benachrichtigung: nur Standardbibliothek,
identisches Verhalten unter Windows und Linux, und unempfindlich gegen das Zusammenfassen von
Events, durch das Watch-Tools auf Netzlaufwerken Dateien verpassen. Für den unbeaufsichtigten
Betrieb in der Aufgabenplanung eintragen (Trigger: *bei Systemstart*) oder mit NSSM als Dienst.
`SIGINT`/`SIGTERM` werden sauber behandelt: die aktuelle Rechnung wird fertig geprüft, dann endet
der Dienst.

## Dashboard

```powershell
 $py -m invoice_checker dashboard --open   # writes data/out/dashboard.html and opens it
 $py -m invoice_checker summary            # same numbers, in the terminal
```

One **self-contained HTML file** — inline SVG, no server, no CDN, no JavaScript dependencies. It
opens offline, prints to PDF, and can be attached to a report as-is. Every chart is paired with the
underlying table so the numbers are readable without the graphics.

It shows: corpus composition, headline gauges per variant, a cross-variant comparison table, a
document/fault outcome breakdown, per-fault-code precision/recall/F1 sorted worst-first, latency
distributions with a quality-vs-cost scatter, detector attribution, and an extraction-health panel.

Metrics are **recomputed from the prediction files** with the same code the `eval` command uses, so
the dashboard cannot disagree with the CLI. It discovers every `predictions_*.jsonl` in `data/out`;
move unwanted runs into a subdirectory (e.g. `data/out/archive/`) and they drop out of the report.

## Data layout

```
data/synth/                     generated corpus (deterministic given --seed)
  inv-0001.clean.txt            ground-truth-clean rendered document text
  inv-0001.doc.txt              what the checker should ingest (may contain faults)
  inv-0001.pdf                  digital PDF (text layer)
  inv-0001.truth.json           {fault_codes: [...], clean: {...}, doc: {...}}
data/out/
  predictions_baseline.jsonl    one record per invoice: predicted faults + evidence
  eval_baseline.json            metrics
```

## Optional dependencies (auto-detected, never required)

| Package | Unlocks | Fallback if absent |
|---|---|---|
| `pytesseract` + Tesseract binary | OCR of `--scans` images | images are skipped with a warning; everything else still runs |
| `pypdf` | robustness on third-party PDFs | built-in extractor handles uncompressed/Flate text streams |
| `sentence-transformers` | embedding-based duplicate detection | `difflib` fuzzy matching (baseline) |

The core pipeline is **stdlib-only**: no numpy, no pydantic, no reportlab. The synthetic PDFs are
written by a ~200-line generator so the project has zero heavyweight requirements.

## Evaluation protocol

Fault injection is *labelled*, so scoring is exact rather than judgemental:

- corpus is generated from a seed, faults are applied programmatically, ground truth is recorded
- metrics are reported **per fault type** (macro-averaged) — aggregate accuracy is meaningless at a
  60% base rate
- `extraction_failed` and `not_evaluated` outcomes are reported separately, never hidden inside the
  accuracy figure
- latency and (optionally) peak VRAM are recorded per invoice so the "runs on consumer hardware"
  claim is measured, not asserted

### Measured results (300 documents, seed 20240517, RTX 3060 Ti / 8 GB)

| Variant | Field source | Micro P | Micro R | Micro F1 | Macro F1 | Document F1 | Latency |
|---|---|---|---|---|---|---|---|
| `baseline` | oracle ground-truth fields | 0.995 | 1.000 | **0.998** | 0.971 | **1.000** | <1 ms |
| `regex` | regex parser over real PDF text | 0.879 | 0.870 | **0.874** | 0.677 | **0.941** | 3 ms |
| `llm` (8-doc sample) | qwen2.5:7b-instruct Q4_K_M | 0.909 | 0.833 | **0.870** | 0.727 | 0.727 | 6.6 s |

Deterministic stages over the same 300 real PDFs: text extraction **1.09 ms/invoice**, rule engine
**0.06 ms/invoice**. Reproduce with the commands below; the corpus is deterministic for a seed.

> The `baseline` row is a **coverage reference, not an accuracy claim** — ground truth is derived by
> running the same rule engine over the stored document, so recall is 1.000 by construction. Its
> purpose is to establish the ceiling and isolate the cost of imperfect extraction, which is exactly
> the 0.998 → 0.874 gap. See [DESIGN.md](DESIGN.md) for the full reasoning, the correctness
> controls, and the bugs those controls caught.
