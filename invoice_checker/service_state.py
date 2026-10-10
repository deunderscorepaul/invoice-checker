"""Zustandsspeicher (SQLite), damit nach einem Neustart nichts doppelt geprüft wird.

Schlüssel ist (Pfad, sha256). Nur der Pfad würde eine korrigierte Rechnung
ignorieren, nur der Hash würde eine unveränderte Kopie unter neuem Namen
überspringen. Zeilen werden nie gelöscht, die Tabelle ist also gleichzeitig die
Historie.
"""
from __future__ import annotations

import hashlib
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS processed (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    path          TEXT NOT NULL,
    sha256        TEXT NOT NULL,
    folder        TEXT NOT NULL,
    original_name TEXT NOT NULL,
    processed_at  TEXT NOT NULL,
    verdict       TEXT NOT NULL,          -- clean | faulty | unreadable | error | skipped
    fault_codes   TEXT NOT NULL DEFAULT '',
    severity      TEXT NOT NULL DEFAULT '',
    destination   TEXT NOT NULL DEFAULT '',
    report_path   TEXT NOT NULL DEFAULT '',
    duration_ms   INTEGER NOT NULL DEFAULT 0,
    detail        TEXT NOT NULL DEFAULT '',
    UNIQUE(path, sha256)
);
CREATE INDEX IF NOT EXISTS idx_processed_path ON processed(path);
CREATE INDEX IF NOT EXISTS idx_processed_at ON processed(processed_at);
"""


@dataclass
class PreviousRun:
    verdict: str
    processed_at: str
    fault_codes: str
    report_path: str


class StateStore:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(str(self.path))
        self.connection.row_factory = sqlite3.Row
        # WAL übersteht einen harten Abbruch besser als das Standardjournal.
        self.connection.execute("PRAGMA journal_mode=WAL")
        self.connection.execute("PRAGMA synchronous=NORMAL")
        self.connection.executescript(SCHEMA)
        self.connection.commit()

    def close(self) -> None:
        try:
            self.connection.close()
        except sqlite3.Error:
            pass

    def __enter__(self) -> "StateStore":
        return self

    def __exit__(self, *_exc) -> None:
        self.close()

    @staticmethod
    def hash_file(path: Path, chunk: int = 1 << 20) -> str:
        digest = hashlib.sha256()
        with Path(path).open("rb") as handle:
            while True:
                block = handle.read(chunk)
                if not block:
                    break
                digest.update(block)
        return digest.hexdigest()

    def previous(self, path: Path, sha256: str) -> PreviousRun | None:
        row = self.connection.execute(
            "SELECT verdict, processed_at, fault_codes, report_path FROM processed "
            "WHERE path = ? AND sha256 = ? ORDER BY id DESC LIMIT 1",
            (str(path), sha256),
        ).fetchone()
        if row is None:
            return None
        return PreviousRun(row["verdict"], row["processed_at"], row["fault_codes"],
                           row["report_path"])

    def latest_for_path(self, path: Path) -> PreviousRun | None:
        row = self.connection.execute(
            "SELECT verdict, processed_at, fault_codes, report_path FROM processed "
            "WHERE path = ? ORDER BY id DESC LIMIT 1",
            (str(path),),
        ).fetchone()
        if row is None:
            return None
        return PreviousRun(row["verdict"], row["processed_at"], row["fault_codes"],
                           row["report_path"])

    def record(self, *, path: Path, sha256: str, folder: str, original_name: str,
               verdict: str, fault_codes: list[str], severity: str, destination: str,
               report_path: str, duration_ms: int, detail: str = "") -> None:
        self.connection.execute(
            "INSERT OR REPLACE INTO processed "
            "(path, sha256, folder, original_name, processed_at, verdict, fault_codes, "
            " severity, destination, report_path, duration_ms, detail) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
            (str(path), sha256, folder, original_name,
             datetime.now(timezone.utc).isoformat(timespec="seconds"),
             verdict, ",".join(sorted(set(fault_codes))), severity, str(destination),
             str(report_path), int(duration_ms), detail[:500]),
        )
        self.connection.commit()

    def counts(self) -> dict[str, int]:
        rows = self.connection.execute(
            "SELECT verdict, COUNT(*) AS n FROM processed GROUP BY verdict"
        ).fetchall()
        return {row["verdict"]: row["n"] for row in rows}

    def recent(self, limit: int = 10) -> list[sqlite3.Row]:
        return self.connection.execute(
            "SELECT original_name, verdict, severity, fault_codes, processed_at "
            "FROM processed ORDER BY id DESC LIMIT ?",
            (limit,),
        ).fetchall()
