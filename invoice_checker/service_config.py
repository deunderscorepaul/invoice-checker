"""Service-Konfiguration aus einer JSON-Datei.

Bewusst Datei statt Kommandozeile: ein Dienst, der unbeaufsichtigt läuft, braucht
seine Einstellungen an einer Stelle. Relative Pfade gelten ab dem Ordner der
Konfigurationsdatei, damit man beides zusammen verschieben kann.

Rechnungen landen im Prüfordner und werden von dort automatisch nach ``approved``
oder ``fehlerhaft`` verschoben.

```json
{
  "watch": [
    {
      "name": "pruefordner",
      "input": "C:/Rechnungen/Pruefordner",
      "approved": "approved",
      "failures": "fehlerhaft",
      "report_language": "de"
    }
  ],
  "poll_seconds": 10,
  "extraction": "regex",
  "model": "qwen2.5:7b-instruct"
}
```
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_EXTENSIONS = (".pdf", ".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".txt")


@dataclass
class WatchFolder:
    """Ein Prüfordner mit den beiden Zielordnern darunter."""
    input_dir: Path
    name: str = ""
    approved_dirname: str = "approved"
    failures_dirname: str = "fehlerhaft"
    report_language: str = "de"
    extraction: str | None = None             # überstimmt die Voreinstellung des Dienstes
    move_unreadable_to_failures: bool = True

    @property
    def approved_dir(self) -> Path:
        return self.input_dir / self.approved_dirname

    @property
    def failures_dir(self) -> Path:
        return self.input_dir / self.failures_dirname

    @property
    def label(self) -> str:
        return self.name or self.input_dir.name


@dataclass
class ServiceConfig:
    watch: list[WatchFolder] = field(default_factory=list)
    poll_seconds: int = 10
    stable_checks: int = 2                    # consecutive identical size+mtime readings
    stable_interval: float = 1.0
    stable_timeout: float = 120.0             # give up waiting for a file to settle
    max_retries: int = 2                      # retries for a *transient* extraction error
    extraction: str = "regex"                 # regex | llm | llm-then-regex
    model: str = "qwen2.5:7b-instruct"
    host: str = "http://127.0.0.1:11434"
    num_ctx: int = 4096
    timeout: int = 300
    extensions: tuple[str, ...] = DEFAULT_EXTENSIONS
    state_path: Path = Path("service_state.sqlite3")
    master_path: Path | None = None           # supplier master data (defaults to data/master_data.json)
    log_path: Path | None = None
    reprocess_if_changed: bool = True
    dry_run: bool = False

    def resolve_extraction(self, folder: WatchFolder) -> str:
        return folder.extraction or self.extraction


def load_config(path: Path) -> ServiceConfig:
    """JSON lesen. Relative Pfade gelten ab dem Ordner der Datei."""
    path = Path(path).resolve()
    if not path.exists():
        raise SystemExit(f"config not found: {path}")
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SystemExit(f"config is not valid JSON: {exc}") from exc

    base = path.parent

    def resolve(value: str | None) -> Path | None:
        if not value:
            return None
        candidate = Path(value)
        return candidate if candidate.is_absolute() else (base / candidate)

    watch: list[WatchFolder] = []
    for entry in raw.get("watch", []):
        input_dir = resolve(entry.get("input"))
        if input_dir is None:
            raise SystemExit("every 'watch' entry needs an 'input' folder")
        watch.append(WatchFolder(
            input_dir=input_dir,
            name=entry.get("name", ""),
            approved_dirname=entry.get("approved", "approved"),
            failures_dirname=entry.get("failures", "fehlerhaft"),
            report_language=entry.get("report_language", raw.get("report_language", "de")),
            extraction=entry.get("extraction"),
            move_unreadable_to_failures=entry.get("move_unreadable_to_failures", True),
        ))

    if not watch:
        raise SystemExit("config contains no 'watch' folders")

    extensions = raw.get("extensions")
    return ServiceConfig(
        watch=watch,
        poll_seconds=int(raw.get("poll_seconds", 10)),
        stable_checks=int(raw.get("stable_checks", 2)),
        stable_interval=float(raw.get("stable_interval", 1.0)),
        stable_timeout=float(raw.get("stable_timeout", 120.0)),
        max_retries=int(raw.get("max_retries", 2)),
        extraction=raw.get("extraction", "regex"),
        model=raw.get("model", "qwen2.5:7b-instruct"),
        host=raw.get("host", "http://127.0.0.1:11434"),
        num_ctx=int(raw.get("num_ctx", 4096)),
        timeout=int(raw.get("timeout", 300)),
        extensions=tuple(extensions) if extensions else DEFAULT_EXTENSIONS,
        state_path=resolve(raw.get("state_path", "service_state.sqlite3")),
        master_path=resolve(raw.get("master_path")),
        log_path=resolve(raw.get("log_path")),
        reprocess_if_changed=raw.get("reprocess_if_changed", True),
        dry_run=bool(raw.get("dry_run", False)),
    )


def sample_config(base: Path) -> dict:
    """Vorlage für ``service --init``."""
    return {
        "_comment": "Rechnungspruefung. Pfade gelten relativ zu dieser Datei.",
        "watch": [
            {
                "name": "pruefordner",
                "input": "rechnungen/pruefordner",
                "approved": "approved",
                "failures": "fehlerhaft",
                "report_language": "de"
            }
        ],
        "poll_seconds": 10,
        "extraction": "regex",
        "model": "qwen2.5:7b-instruct",
        "host": "http://127.0.0.1:11434",
        "state_path": "service_state.sqlite3",
        "log_path": "service.log",
        "dry_run": False
    }
